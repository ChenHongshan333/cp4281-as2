#!/usr/bin/env python3
"""Generate one reproducible FLUX.1-schnell intermediate image for Step 3b."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import FluxPipeline
from PIL import Image


FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
FLUX_MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--experiment-spec", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
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


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_case(spec_path: Path, case_id: str) -> tuple[str, dict[str, Any]]:
    with spec_path.open(encoding="utf-8") as handle:
        spec = json.load(handle)

    text_conditioned = spec.get("text_conditioned")
    if not isinstance(text_conditioned, dict):
        raise ValueError("experiment_spec.json has no text_conditioned object")

    template = text_conditioned.get("prompt_template")
    if not isinstance(template, str) or "{subject_description}" not in template:
        raise ValueError("Prompt template must contain {subject_description}")

    matches = [
        case
        for case in text_conditioned.get("cases", [])
        if case.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one text case {case_id!r}; found {len(matches)}")

    case = matches[0]
    subject = case.get("subject_description")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError(f"Case {case_id!r} has no subject_description")

    expanded_prompt = template.format(subject_description=subject)
    if "{" in expanded_prompt or "}" in expanded_prompt:
        raise ValueError(f"Unresolved placeholder in expanded prompt: {expanded_prompt}")
    return template, case


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not args.experiment_spec.is_file():
        raise FileNotFoundError(args.experiment_spec)
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not (args.model_path / "model_index.json").is_file():
        raise FileNotFoundError(args.model_path / "model_index.json")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output_dir}")
    if args.height % 16 or args.width % 16:
        raise ValueError("FLUX image dimensions must be divisible by 16")
    if args.num_inference_steps != 4:
        raise ValueError("This experiment fixes FLUX.1-schnell at 4 inference steps")
    if args.guidance_scale != 0.0:
        raise ValueError("This experiment fixes FLUX.1-schnell guidance_scale at 0.0")


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    validate_args(args)
    template, case = load_case(args.experiment_spec, args.case_id)
    expanded_prompt = template.format(
        subject_description=case["subject_description"]
    )

    args.output_dir.mkdir(parents=True)
    image_path = args.output_dir / "intermediate.png"
    metadata_path = args.output_dir / "metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage": "Step 3b - FLUX text-to-image intermediate",
        "status": "running",
        "case_id": args.case_id,
        "started_at": now_iso(),
        "prompt_template": template,
        "subject_description": case["subject_description"],
        "expanded_prompt": expanded_prompt,
        "challenge": case.get("challenge"),
        "analysis_focus": case.get("analysis_focus"),
        "seed": args.seed,
        "generation_parameters": {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
            "dtype": "bfloat16",
            "memory_strategy": "model_cpu_offload",
        },
        "model": {
            "id": FLUX_MODEL_ID,
            "revision": FLUX_MODEL_REVISION,
            "snapshot_path": str(args.model_path),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "pillow": package_version("Pillow"),
            "gpu_name": torch.cuda.get_device_name(0),
        },
    }
    write_json(metadata_path, metadata)

    try:
        init_start = time.perf_counter()
        pipeline = FluxPipeline.from_pretrained(
            str(args.model_path),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(desc="FLUX.1-schnell")
        initialization_seconds = elapsed(init_start)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        torch.cuda.synchronize()
        generation_start = time.perf_counter()
        result = pipeline(
            prompt=expanded_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            generator=generator,
            output_type="pil",
        )
        torch.cuda.synchronize()
        text_to_image_seconds = elapsed(generation_start)
        peak_cuda_memory_bytes = torch.cuda.max_memory_allocated()

        if len(result.images) != 1 or not isinstance(result.images[0], Image.Image):
            raise RuntimeError("FLUX did not return exactly one PIL image")
        image = result.images[0].convert("RGB")
        if image.size != (args.width, args.height):
            raise RuntimeError(f"Unexpected output size: {image.size}")
        image.save(image_path, format="PNG")

        metadata.update(
            {
                "status": "completed",
                "finished_at": now_iso(),
                "timings_seconds": {
                    "shared_model_initialization": initialization_seconds,
                    "text_to_image": text_to_image_seconds,
                    "per_case_end_to_end": elapsed(run_start),
                },
                "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
                "output_files": {
                    "intermediate_image": image_path.name,
                    "intermediate_image_sha256": sha256(image_path),
                    "metadata": metadata_path.name,
                },
                "output_image": {
                    "format": "PNG",
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                },
            }
        )
        write_json(metadata_path, metadata)
        print("STEP 3B FLUX GENERATION: PASSED")
        print(f"Intermediate image: {image_path}")
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
                "timings_seconds": {"failed_after": elapsed(run_start)},
            }
        )
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
