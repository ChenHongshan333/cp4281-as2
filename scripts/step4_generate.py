#!/usr/bin/env python3
"""Run one reproducible TRELLIS.2 Step 4 trajectory extraction case."""

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
from pathlib import Path
from typing import Any

try:
    from scripts.step4_capture import (
        STAGE1_RESCALE_T,
        STAGE1_STEPS,
        STAGE1_TARGETS,
        STAGE2_RESCALE_T,
        STAGE2_STEPS,
        STAGE2_TARGETS,
        CapturedInvocation,
        SamplerCapture,
    )
    from scripts.step4_decode import (
        RenderCamera,
        decode_stage1_invocation,
        decode_stage2_invocation,
        frame_manifest,
        select_high_resolution_shape_invocation,
    )
except ModuleNotFoundError:
    from step4_capture import (
        STAGE1_RESCALE_T,
        STAGE1_STEPS,
        STAGE1_TARGETS,
        STAGE2_RESCALE_T,
        STAGE2_STEPS,
        STAGE2_TARGETS,
        CapturedInvocation,
        SamplerCapture,
    )
    from step4_decode import (
        RenderCamera,
        decode_stage1_invocation,
        decode_stage2_invocation,
        frame_manifest,
        select_high_resolution_shape_invocation,
    )


TRELLIS_SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
MODEL_REVISIONS = {
    "microsoft/TRELLIS.2-4B": "af44b45f2e35a493886929c6d786e563ec68364d",
    "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": (
        "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    ),
    "briaai/RMBG-2.0": "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
}
PIPELINE_TYPE = "1024_cascade"
SEED = 42
STRUCTURE_RESOLUTION = 32
EXPECTED_CASE_IDS = (
    "img02_hard_surface",
    "img04_organic_shape",
    "img05_complex_shape",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-spec",
        type=Path,
        default=Path("inputs/step3/experiment_spec.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--trellis-source", type=Path)
    parser.add_argument("--render-resolution", type=int, default=512)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def resolve_under(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Path escapes project root: {path}") from error
    return resolved


def read_experiment_spec(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("status") != "step3_complete":
        raise ValueError(f"Step 3 is not complete: {spec.get('status')}")
    return spec


def step4_candidates(spec: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        case
        for case in spec.get("image_conditioned_cases", [])
        if case.get("step4_candidate") is True
    ]
    case_ids = tuple(case.get("case_id") for case in candidates)
    if case_ids != EXPECTED_CASE_IDS:
        raise ValueError(
            f"Unexpected Step 4 candidates: {case_ids}; expected {EXPECTED_CASE_IDS}"
        )
    return candidates


def load_case(spec: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [
        case for case in step4_candidates(spec) if case.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Step 4 candidate {case_id!r}; found {len(matches)}"
        )
    if matches[0].get("status") != "completed":
        raise ValueError(f"Step 3 case is not completed: {matches[0]}")
    return matches[0]


def state_filename(position: int, timestep: float) -> str:
    if position < 0:
        raise ValueError(position)
    return f"state_{position:02d}_t{timestep:.6f}.pt"


def frame_filename(position: int, timestep: float) -> str:
    if position < 0:
        raise ValueError(position)
    return f"frame_{position:02d}_t{timestep:.6f}.png"


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def invocation_manifest(
    invocation: CapturedInvocation,
    state_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": invocation.stage,
        "invocation_index": invocation.invocation_index,
        "model_label": invocation.model_label,
        "steps": invocation.steps,
        "rescale_t": invocation.rescale_t,
        "returned_timesteps": list(invocation.returned_timesteps),
        "selected_states": state_records,
    }


def save_invocation_states(
    invocation: CapturedInvocation,
    directory: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    import torch

    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for position, state in enumerate(invocation.selected_states):
        path = directory / state_filename(position, state.timestep)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "stage": invocation.stage,
                "invocation_index": invocation.invocation_index,
                "model_label": invocation.model_label,
                "state_index": state.state_index,
                "timestep": state.timestep,
                "target_timestep": state.target_timestep,
                "payload": state.payload,
            },
            temporary,
        )
        temporary.replace(path)
        records.append(
            {
                "position": position,
                "state_index": state.state_index,
                "timestep": state.timestep,
                "target_timestep": state.target_timestep,
                "path": relative_path(path, output_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def save_decoded_frames(
    frames: list[Any],
    directory: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for position, frame in enumerate(frames):
        path = directory / frame_filename(position, frame.timestep)
        Image.fromarray(frame.image).save(path)
        record = frame_manifest(frame)
        record.update(
            {
                "position": position,
                "path": relative_path(path, output_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        records.append(record)
    return records


def model_label_resolver(pipeline: Any):
    identities = {id(model): name for name, model in pipeline.models.items()}

    def resolve(model: Any) -> str:
        label = identities.get(id(model))
        if label is None:
            raise ValueError(f"Sampler received an unknown model: {type(model)!r}")
        return label

    return resolve


def run_gpu_case(args: argparse.Namespace) -> None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from PIL import Image, ImageOps

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils

    try:
        from scripts.step3a_generate import (
            build_neutral_studio_environment,
            install_dinov3_transformers5_compatibility,
        )
    except ModuleNotFoundError:
        from step3a_generate import (
            build_neutral_studio_environment,
            install_dinov3_transformers5_compatibility,
        )

    required = {
        "case_id": args.case_id,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "trellis_source": args.trellis_source,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing runtime arguments: {missing}")
    if args.render_resolution <= 0:
        raise ValueError("render-resolution must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    probe = torch.ones(1, device="cuda")
    if not probe.is_cuda:
        raise RuntimeError("CUDA tensor probe failed")

    project_root = args.project_root.resolve()
    spec_path = resolve_under(project_root, args.experiment_spec)
    model_path = args.model_path.resolve()
    trellis_source = args.trellis_source.resolve()
    output_root = resolve_under(project_root, args.output_dir)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_root}")
    if not (model_path / "pipeline.json").is_file():
        raise FileNotFoundError(model_path / "pipeline.json")
    if not trellis_source.is_dir():
        raise FileNotFoundError(trellis_source)
    source_revision = git_revision(trellis_source)
    if source_revision != TRELLIS_SOURCE_REVISION:
        raise RuntimeError(f"Unexpected TRELLIS.2 revision: {source_revision}")

    spec = read_experiment_spec(spec_path)
    case = load_case(spec, args.case_id)
    input_path = resolve_under(project_root, Path(case["source_input"]))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_digest = sha256(input_path)
    if input_digest != case["source_sha256"]:
        raise RuntimeError(
            f"Source SHA-256 mismatch: {input_digest} != {case['source_sha256']}"
        )

    output_root.mkdir(parents=True)
    metadata_path = output_root / "metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage": "Step 4 - Visualise the diffusion process",
        "status": "running",
        "case_id": args.case_id,
        "started_at": now_iso(),
        "source_input": str(input_path),
        "source_sha256": input_digest,
        "source_provenance": case["source_provenance"],
        "challenge": case["challenge"],
        "selection_rationale": case["selection_rationale"],
        "seed": SEED,
        "pipeline_type": PIPELINE_TYPE,
        "model_revisions": MODEL_REVISIONS,
        "trellis_source_revision": source_revision,
        "capture_policy": {
            "captured_value": "pred_x_t (the post-update x_{t_prev})",
            "stage1": {
                "steps": STAGE1_STEPS,
                "rescale_t": STAGE1_RESCALE_T,
                "target_range": [0.5, 0.0],
                "selection_targets": list(STAGE1_TARGETS),
            },
            "stage2": {
                "steps": STAGE2_STEPS,
                "rescale_t": STAGE2_RESCALE_T,
                "target_range": [1.0, 0.0],
                "selection_targets": list(STAGE2_TARGETS),
                "formal_render_invocation": "shape_slat_flow_model_1024",
            },
            "note": (
                "Stage 1 uses 24 rather than 12 Euler steps, so this Step 4 "
                "rerun is deterministic but is not claimed to be byte-identical "
                "to the accepted Step 3 asset."
            ),
        },
        "runtime": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "timings_seconds": {},
    }
    write_json(metadata_path, metadata)

    try:
        with Image.open(input_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        input_copy = output_root / "input.png"
        image.save(input_copy)

        initialization_start = time.perf_counter()
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_path))
        pipeline.cuda()
        compatibility = install_dinov3_transformers5_compatibility(pipeline)
        torch.cuda.synchronize()
        metadata["timings_seconds"]["model_initialization"] = elapsed(
            initialization_start
        )
        metadata["compatibility_patches"] = [compatibility]

        torch.cuda.synchronize()
        preprocessing_start = time.perf_counter()
        processed = pipeline.preprocess_image(image)
        torch.cuda.synchronize()
        metadata["timings_seconds"]["preprocessing"] = elapsed(
            preprocessing_start
        )
        processed_path = output_root / "preprocessed.png"
        processed.save(processed_path)

        resolve_model_label = model_label_resolver(pipeline)
        with SamplerCapture(
            pipeline.sparse_structure_sampler,
            stage="stage1_sparse_structure",
            targets=STAGE1_TARGETS,
            minimum_timestep=0.0,
            maximum_timestep=0.5,
            model_label=resolve_model_label,
        ) as stage1_capture, SamplerCapture(
            pipeline.shape_slat_sampler,
            stage="stage2_shape_slat",
            targets=STAGE2_TARGETS,
            minimum_timestep=0.0,
            maximum_timestep=1.0,
            model_label=resolve_model_label,
        ) as stage2_capture:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            generation_start = time.perf_counter()
            outputs, latents = pipeline.run(
                processed,
                seed=SEED,
                preprocess_image=False,
                pipeline_type=PIPELINE_TYPE,
                sparse_structure_sampler_params={"steps": STAGE1_STEPS},
                shape_slat_sampler_params={"steps": STAGE2_STEPS},
                tex_slat_sampler_params={"steps": 12},
                return_latent=True,
            )
            torch.cuda.synchronize()
            metadata["timings_seconds"]["generation_and_capture"] = elapsed(
                generation_start
            )
            metadata["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()

        if len(stage1_capture.invocations) != 1:
            raise RuntimeError(
                f"Expected one Stage 1 invocation, got {len(stage1_capture.invocations)}"
            )
        if len(stage2_capture.invocations) != 2:
            raise RuntimeError(
                "Expected the 1024 cascade to invoke the shape sampler twice; "
                f"got {len(stage2_capture.invocations)}"
            )
        stage1_invocation = stage1_capture.invocations[0]
        stage2_lr_invocation = stage2_capture.invocations[0]
        stage2_hr_invocation = select_high_resolution_shape_invocation(
            stage2_capture.invocations
        )
        if stage2_lr_invocation.model_label != "shape_slat_flow_model_512":
            raise RuntimeError(
                f"Unexpected first shape pass: {stage2_lr_invocation.model_label}"
            )

        shape_resolution = int(latents[2])
        del latents
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one final mesh, got {len(outputs)}")
        final_mesh = outputs[0]
        final_mesh.simplify(16_777_216)
        environment = build_neutral_studio_environment(final_mesh.vertices.device)
        envmap = EnvMap(environment)
        final_renders = render_utils.render_snapshot(
            final_mesh,
            resolution=args.render_resolution,
            r=2,
            fov=36,
            nviews=1,
            envmap=envmap,
        )
        final_render_path = output_root / "final_render.png"
        Image.fromarray(final_renders["shaded"][0]).save(final_render_path)
        metadata["final_mesh"] = {
            "vertices": int(final_mesh.vertices.shape[0]),
            "faces": int(final_mesh.faces.shape[0]),
            "shape_resolution": shape_resolution,
            "render_mode": "pbr_shaded_procedural_neutral",
        }
        del outputs, final_mesh, final_renders, envmap, environment
        torch.cuda.empty_cache()

        capture_start = time.perf_counter()
        stage1_state_records = save_invocation_states(
            stage1_invocation,
            output_root / "stage1" / "states",
            output_root,
        )
        stage2_lr_state_records = save_invocation_states(
            stage2_lr_invocation,
            output_root / "stage2_lr" / "states",
            output_root,
        )
        stage2_hr_state_records = save_invocation_states(
            stage2_hr_invocation,
            output_root / "stage2" / "states",
            output_root,
        )
        metadata["timings_seconds"]["save_captured_states"] = elapsed(
            capture_start
        )

        camera = RenderCamera(resolution=args.render_resolution)
        torch.cuda.synchronize()
        stage1_decode_start = time.perf_counter()
        stage1_frames = decode_stage1_invocation(
            pipeline,
            stage1_invocation,
            structure_resolution=STRUCTURE_RESOLUTION,
            camera=camera,
        )
        torch.cuda.synchronize()
        metadata["timings_seconds"]["stage1_decode_and_render"] = elapsed(
            stage1_decode_start
        )

        torch.cuda.synchronize()
        stage2_decode_start = time.perf_counter()
        stage2_frames = decode_stage2_invocation(
            pipeline,
            stage2_hr_invocation,
            shape_resolution=shape_resolution,
            camera=camera,
        )
        torch.cuda.synchronize()
        metadata["timings_seconds"]["stage2_decode_and_render"] = elapsed(
            stage2_decode_start
        )

        stage1_frame_records = save_decoded_frames(
            stage1_frames,
            output_root / "stage1" / "frames",
            output_root,
        )
        stage2_frame_records = save_decoded_frames(
            stage2_frames,
            output_root / "stage2" / "frames",
            output_root,
        )

        metadata.update(
            {
                "status": "completed",
                "finished_at": now_iso(),
                "inputs": {
                    "original": relative_path(input_copy, output_root),
                    "preprocessed": relative_path(processed_path, output_root),
                },
                "final_render": {
                    "path": relative_path(final_render_path, output_root),
                    "size_bytes": final_render_path.stat().st_size,
                    "sha256": sha256(final_render_path),
                },
                "captures": {
                    "stage1": invocation_manifest(
                        stage1_invocation, stage1_state_records
                    ),
                    "stage2_low_resolution_evidence": invocation_manifest(
                        stage2_lr_invocation, stage2_lr_state_records
                    ),
                    "stage2_formal_high_resolution": invocation_manifest(
                        stage2_hr_invocation, stage2_hr_state_records
                    ),
                },
                "rendered_frames": {
                    "stage1": stage1_frame_records,
                    "stage2": stage2_frame_records,
                },
                "acceptance": {
                    "stage1_frame_count": len(stage1_frame_records),
                    "stage2_frame_count": len(stage2_frame_records),
                    "all_true_timesteps_recorded": True,
                    "formal_stage2_is_high_resolution_pass": True,
                    "labelled_strips_created": False,
                },
            }
        )
        if len(stage1_frame_records) != 5 or len(stage2_frame_records) != 5:
            raise RuntimeError("Step 4 case did not produce five frames per stage")
        write_json(metadata_path, metadata)
        print("STEP 4 SINGLE-CASE EXTRACTION: PASSED")
        print(f"Output: {output_root}")
        print(f"Metadata: {metadata_path}")
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_json(metadata_path, metadata)
        raise


def run_self_test(spec_path: Path, project_root: Path) -> None:
    resolved_spec = resolve_under(project_root.resolve(), spec_path)
    spec = read_experiment_spec(resolved_spec)
    candidates = step4_candidates(spec)
    assert tuple(case["case_id"] for case in candidates) == EXPECTED_CASE_IDS
    for case in candidates:
        loaded = load_case(spec, case["case_id"])
        assert loaded is case
        source = resolve_under(project_root.resolve(), Path(case["source_input"]))
        assert source.is_file(), source
        assert sha256(source) == case["source_sha256"]
    assert state_filename(0, 0.5) == "state_00_t0.500000.pt"
    assert frame_filename(4, 0.0) == "frame_04_t0.000000.png"
    print("Candidates:", ", ".join(EXPECTED_CASE_IDS))
    print("Candidate input hashes: PASSED")
    print("Output naming policy: PASSED")
    print("STEP 4 SINGLE-CASE RUNNER SELF-TEST: PASSED")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.experiment_spec, args.project_root)
        return
    run_gpu_case(args)


if __name__ == "__main__":
    main()
