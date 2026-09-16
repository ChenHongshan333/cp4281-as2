#!/usr/bin/env python3
"""Run one reproducible TRELLIS.2 image-conditioned generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


MODEL_REVISIONS = {
    "microsoft/TRELLIS.2-4B": "af44b45f2e35a493886929c6d786e563ec68364d",
    "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "briaai/RMBG-2.0": "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
}
TRELLIS_SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--trellis-source", required=True, type=Path)
    parser.add_argument("--envmap", required=True, type=Path)
    parser.add_argument("--experiment-spec", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="1024_cascade",
    )
    parser.add_argument("--decimation-target", type=int, default=500_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--render-resolution", type=int, default=768)
    parser.add_argument("--render-views", type=int, default=4)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def build_neutral_studio_environment(device: torch.device) -> torch.Tensor:
    """Create a deterministic lat-long HDR environment for PBR previews."""
    height, width = 256, 512
    latitude, longitude = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-torch.pi, torch.pi, width, device=device),
        indexing="ij",
    )

    # A soft warm ceiling provides even base illumination.
    vertical = 0.22 + 0.28 * ((1.0 - latitude) * 0.5)
    environment = vertical[..., None] * torch.tensor(
        [1.0, 0.98, 0.95], dtype=torch.float32, device=device
    )

    def add_softbox(
        center_longitude: float,
        center_latitude: float,
        longitude_spread: float,
        latitude_spread: float,
        intensity: float,
        color: tuple[float, float, float],
    ) -> None:
        nonlocal environment
        distance = (
            ((longitude - center_longitude) / longitude_spread) ** 2
            + ((latitude - center_latitude) / latitude_spread) ** 2
        )
        weight = torch.exp(-0.5 * distance)[..., None]
        environment = environment + weight * intensity * torch.tensor(
            color, dtype=torch.float32, device=device
        )

    add_softbox(-0.8, -0.35, 0.42, 0.24, 3.0, (1.0, 0.91, 0.82))
    add_softbox(1.35, -0.10, 0.65, 0.38, 1.1, (0.72, 0.82, 1.0))
    add_softbox(2.65, 0.10, 0.35, 0.50, 0.7, (1.0, 1.0, 1.0))
    environment = environment.contiguous()
    if environment.shape != (height, width, 3):
        raise RuntimeError(
            f"Unexpected procedural environment shape: {environment.shape}"
        )
    if not torch.isfinite(environment).all():
        raise RuntimeError("Procedural environment contains non-finite values")
    return environment


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def install_dinov3_transformers5_compatibility(
    pipeline: Trellis2ImageTo3DPipeline,
) -> dict[str, Any]:
    """Preserve TRELLIS.2's pre-norm DINOv3 features across Transformers APIs."""
    extractor = pipeline.image_cond_model
    model = extractor.model

    candidates = (
        ("layer", model),
        ("model.layer", getattr(model, "model", None)),
        ("encoder.layer", getattr(model, "encoder", None)),
    )
    for layer_path, holder in candidates:
        if holder is not None and hasattr(holder, "layer"):
            layers = holder.layer
            break
    else:
        raise RuntimeError(
            "Cannot locate the DINOv3 encoder layers in the installed "
            f"Transformers API ({type(model).__name__})"
        )

    if layer_path == "layer":
        return {
            "name": "dinov3_transformers5_layer_path",
            "applied": False,
            "layer_path": layer_path,
            "layer_count": len(layers),
        }

    def extract_features_compat(image: torch.Tensor) -> torch.Tensor:
        image = image.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = model.embeddings(image, bool_masked_pos=None)
        position_embeddings = model.rope_embeddings(image)
        for layer_module in layers:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        # TRELLIS.2 expects parameter-free normalization of pre-norm states.
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    extractor.extract_features = extract_features_compat
    return {
        "name": "dinov3_transformers5_layer_path",
        "applied": True,
        "layer_path": layer_path,
        "layer_count": len(layers),
        "upstream_reference": "microsoft/TRELLIS.2 pull request 156 and commit 3f4faad",
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    trellis_source = args.trellis_source.resolve()
    envmap_path = args.envmap.resolve()
    experiment_spec_path = (
        args.experiment_spec.resolve() if args.experiment_spec is not None else None
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not (model_path / "pipeline.json").is_file():
        raise FileNotFoundError(f"TRELLIS.2 snapshot is incomplete: {model_path}")
    if not envmap_path.is_file():
        raise FileNotFoundError(f"Environment map not found: {envmap_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    source_digest = sha256(input_path)
    experiment_case: dict[str, Any] | None = None
    experiment_schema_version: int | None = None
    if experiment_spec_path is not None:
        if not experiment_spec_path.is_file():
            raise FileNotFoundError(
                f"Experiment specification not found: {experiment_spec_path}"
            )
        experiment_spec = json.loads(experiment_spec_path.read_text(encoding="utf-8"))
        experiment_schema_version = experiment_spec.get("schema_version")
        matching_cases = [
            case
            for case in experiment_spec.get("image_conditioned_cases", [])
            if case.get("case_id") == args.case_id
        ]
        if len(matching_cases) != 1:
            raise RuntimeError(
                f"Expected exactly one experiment case for {args.case_id}; "
                f"found {len(matching_cases)}"
            )
        experiment_case = matching_cases[0]
        if experiment_case.get("status") != "ready":
            raise RuntimeError(
                f"Experiment case {args.case_id} is not ready: "
                f"{experiment_case.get('status')}"
            )
        if experiment_case.get("source_sha256") != source_digest:
            raise RuntimeError(
                f"Experiment-spec SHA-256 does not match input for {args.case_id}"
            )

    output_dir.mkdir(parents=True)

    metadata_path = output_dir / "metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "case_id": args.case_id,
        "status": "running",
        "started_at": now_iso(),
        "source_input": str(input_path),
        "source_sha256": source_digest,
        "background_removal_applied": True,
        "seed": args.seed,
        "generation_parameters": {
            "pipeline_type": args.pipeline_type,
            "sparse_structure_sampler": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "guidance_interval": [0.6, 1.0],
                "rescale_t": 5.0,
            },
            "shape_slat_sampler": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "guidance_interval": [0.6, 1.0],
                "rescale_t": 3.0,
            },
            "texture_slat_sampler": {
                "steps": 12,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "guidance_interval": [0.6, 0.9],
                "rescale_t": 3.0,
            },
            "decimation_target": args.decimation_target,
            "texture_size": args.texture_size,
            "render_resolution": args.render_resolution,
            "render_views": args.render_views,
        },
        "model_revisions": MODEL_REVISIONS,
        "trellis_source_revision": git_revision(trellis_source),
        "runtime": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "pillow": package_version("Pillow"),
            "transformers": package_version("transformers"),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "timings_seconds": {},
        "output_files": {},
    }
    if experiment_case is not None:
        metadata["experiment_spec"] = {
            "path": str(experiment_spec_path),
            "schema_version": experiment_schema_version,
        }
        metadata["source_provenance"] = experiment_case.get("source_provenance")
        metadata["challenge"] = experiment_case.get("challenge")
        metadata["selection_rationale"] = experiment_case.get(
            "selection_rationale"
        )
        metadata["step4_candidate"] = experiment_case.get("step4_candidate")
    write_json(metadata_path, metadata)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if metadata["trellis_source_revision"] != TRELLIS_SOURCE_REVISION:
            raise RuntimeError(
                "Unexpected TRELLIS.2 source revision: "
                f"{metadata['trellis_source_revision']}"
            )

        initialization_start = time.perf_counter()
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_path))
        pipeline.cuda()
        dinov3_compatibility = install_dinov3_transformers5_compatibility(pipeline)
        metadata["compatibility_patches"] = [dinov3_compatibility]
        print(
            "DINOv3 encoder layers: "
            f"{dinov3_compatibility['layer_path']} "
            f"({dinov3_compatibility['layer_count']} layers); "
            f"compatibility patch applied={dinov3_compatibility['applied']}"
        )
        cuda_sync()
        metadata["timings_seconds"]["shared_model_initialization"] = elapsed(
            initialization_start
        )

        with Image.open(input_path) as opened:
            stored_size = list(opened.size)
            original_mode = opened.mode
            exif_orientation = opened.getexif().get(274, 1)
            image = ImageOps.exif_transpose(opened).copy()
        metadata["source_image"] = {
            "stored_size": stored_size,
            "oriented_size": list(image.size),
            "mode": original_mode,
            "exif_orientation": exif_orientation,
        }
        metadata["background_removal_applied"] = not (
            image.mode == "RGBA"
            and np.any(np.asarray(image)[:, :, 3] != 255)
        )

        case_start = time.perf_counter()
        cuda_sync()
        preprocessing_start = time.perf_counter()
        processed = pipeline.preprocess_image(image)
        cuda_sync()
        metadata["timings_seconds"]["preprocessing"] = elapsed(
            preprocessing_start
        )
        processed_path = output_dir / "preprocessed.png"
        processed.save(processed_path)

        torch.cuda.reset_peak_memory_stats()
        cuda_sync()
        generation_start = time.perf_counter()
        mesh = pipeline.run(
            processed,
            seed=args.seed,
            preprocess_image=False,
            pipeline_type=args.pipeline_type,
        )[0]
        cuda_sync()
        metadata["timings_seconds"]["trellis_generation"] = elapsed(
            generation_start
        )
        metadata["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        mesh.simplify(16_777_216)

        cuda_sync()
        export_start = time.perf_counter()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb_path = output_dir / "asset.glb"
        glb.export(glb_path, extension_webp=True)
        cuda_sync()
        metadata["timings_seconds"]["glb_export"] = elapsed(export_start)
        metadata["timings_seconds"]["per_case_end_to_end"] = elapsed(case_start)

        cuda_sync()
        render_start = time.perf_counter()
        environment = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
        if environment is not None:
            environment = cv2.cvtColor(environment, cv2.COLOR_BGR2RGB)
            envmap = EnvMap(
                torch.tensor(environment, dtype=torch.float32, device="cuda")
            )
            renders = render_utils.render_snapshot(
                mesh,
                resolution=args.render_resolution,
                r=2,
                fov=36,
                nviews=args.render_views,
                envmap=envmap,
            )
            render_key = "shaded"
            metadata["rendering"] = {
                "mode": "pbr_shaded",
                "envmap": str(envmap_path),
                "envmap_loaded": True,
            }
        else:
            print(
                "WARNING: OpenCV could not read the EXR environment map; "
                "using a procedural neutral studio environment for PBR previews."
            )
            environment = build_neutral_studio_environment(mesh.vertices.device)
            envmap = EnvMap(environment)
            renders = render_utils.render_snapshot(
                mesh,
                resolution=args.render_resolution,
                r=2,
                fov=36,
                nviews=args.render_views,
                envmap=envmap,
            )
            render_key = "shaded"
            metadata["rendering"] = {
                "mode": "pbr_shaded_procedural_neutral",
                "requested_envmap": str(envmap_path),
                "envmap_loaded": False,
                "procedural_envmap": True,
                "procedural_envmap_shape": list(environment.shape),
                "procedural_envmap_value_range": [
                    float(environment.min().item()),
                    float(environment.max().item()),
                ],
                "warning": "OpenCV returned None while reading the EXR file.",
            }
        cuda_sync()
        metadata["timings_seconds"]["preview_rendering"] = elapsed(render_start)
        view_paths = []
        for index, frame in enumerate(renders[render_key]):
            view_path = output_dir / f"view_{index:02d}.png"
            Image.fromarray(frame).save(view_path)
            view_paths.append(str(view_path))

        metadata["mesh"] = {
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "voxel_size": float(mesh.voxel_size),
        }
        metadata["output_files"] = {
            "preprocessed_image": str(processed_path),
            "glb": str(glb_path),
            "render_key": render_key,
            "rendered_views": view_paths,
        }
        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        write_json(metadata_path, metadata)
        print("STEP 3A GENERATION: PASSED")
        print(f"GLB: {glb_path}")
        print(f"Metadata: {metadata_path}")
    except Exception as error:
        metadata["status"] = "failed"
        metadata["finished_at"] = now_iso()
        metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
