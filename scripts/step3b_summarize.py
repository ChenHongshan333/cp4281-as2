#!/usr/bin/env python3
"""Validate and summarize a completed Step 3b text-conditioned batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CASE_IDS = [
    "txt01_desk_fan",
    "txt02_toolbox",
    "txt03_perfume_bottle",
    "txt04_root_sculpture",
    "txt05_lunar_rover",
]
FLUX_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"
TRELLIS_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--experiment-spec", required=True, type=Path)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty output file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    spec = read_json(args.experiment_spec.resolve())
    text_spec = spec["text_conditioned"]
    spec_cases = text_spec["cases"]
    assert [case["case_id"] for case in spec_cases] == EXPECTED_CASE_IDS
    assert text_spec["prompt_template_revision"] == 3
    assert text_spec["prompt_token_verification"]["status"] == "passed"
    assert text_spec["status"] in {
        "flux_smoke_complete_batch_ready",
        "batch_scripts_ready",
    }

    cases = []
    for spec_case in spec_cases:
        case_id = spec_case["case_id"]
        case_root = output_root / case_id
        flux_root = case_root / "flux"
        trellis_root = case_root / "trellis"
        flux_metadata_path = flux_root / "metadata.json"
        trellis_metadata_path = trellis_root / "metadata.json"
        intermediate_path = flux_root / "intermediate.png"
        glb_path = trellis_root / "asset.glb"

        flux_metadata = read_json(flux_metadata_path)
        trellis_metadata = read_json(trellis_metadata_path)
        expected_prompt = text_spec["prompt_template"].format(
            subject_description=spec_case["subject_description"]
        )

        assert flux_metadata["status"] == "completed", flux_metadata
        assert flux_metadata["case_id"] == case_id, flux_metadata
        assert flux_metadata["prompt_template"] == text_spec["prompt_template"]
        assert flux_metadata["expanded_prompt"] == expected_prompt
        assert flux_metadata["seed"] == text_spec["text_to_image"]["seed"]
        assert flux_metadata["model"]["revision"] == FLUX_REVISION
        assert flux_metadata["prompt_tokenization"]["clip_within_limit"] is True
        assert (
            flux_metadata["prompt_tokenization"]["t5_within_requested_limit"]
            is True
        )

        intermediate = checked_file(intermediate_path)
        assert (
            intermediate["sha256"]
            == flux_metadata["output_files"]["intermediate_image_sha256"]
        )

        assert trellis_metadata["status"] == "completed", trellis_metadata
        assert trellis_metadata["case_id"] == case_id, trellis_metadata
        assert trellis_metadata["source_sha256"] == intermediate["sha256"]
        assert trellis_metadata["background_removal_applied"] is True
        assert (
            trellis_metadata["model_revisions"]["microsoft/TRELLIS.2-4B"]
            == TRELLIS_REVISION
        )
        assert trellis_metadata["output_files"]["render_key"] == "shaded"
        assert trellis_metadata["rendering"]["mode"].startswith("pbr_shaded")

        glb = checked_file(glb_path)
        view_paths = sorted(trellis_root.glob("view_*.png"))
        assert len(view_paths) == 4, view_paths
        rendered_views = [checked_file(path) for path in view_paths]

        flux_timings = flux_metadata["timings_seconds"]
        trellis_timings = trellis_metadata["timings_seconds"]
        combined_primary = round(
            flux_timings["text_to_image"]
            + trellis_timings["per_case_end_to_end"],
            4,
        )
        timings = {
            "flux_model_initialization": flux_timings[
                "shared_model_initialization"
            ],
            "text_to_image": flux_timings["text_to_image"],
            "trellis_model_initialization": trellis_timings[
                "shared_model_initialization"
            ],
            "trellis_preprocessing": trellis_timings["preprocessing"],
            "trellis_generation": trellis_timings["trellis_generation"],
            "glb_export": trellis_timings["glb_export"],
            "preview_rendering": trellis_timings["preview_rendering"],
            "trellis_case_end_to_end": trellis_timings[
                "per_case_end_to_end"
            ],
            "combined_primary_end_to_end": combined_primary,
        }

        case_summary = {
            "schema_version": 1,
            "stage": "Step 3b - Text-conditioned generation",
            "status": "completed",
            "case_id": case_id,
            "prompt_authorship": text_spec["prompt_authorship"],
            "prompt_template_revision": text_spec["prompt_template_revision"],
            "prompt_template": text_spec["prompt_template"],
            "prompt_template_rationale": text_spec[
                "prompt_template_rationale"
            ],
            "subject_description": spec_case["subject_description"],
            "expanded_prompt": expected_prompt,
            "challenge": spec_case["challenge"],
            "selection_rationale": spec_case["selection_rationale"],
            "analysis_focus": spec_case["analysis_focus"],
            "text_to_image_limitation_candidate": case_id
            == "txt05_lunar_rover",
            "seed": flux_metadata["seed"],
            "prompt_tokenization": flux_metadata["prompt_tokenization"],
            "model_revisions": {
                "black-forest-labs/FLUX.1-schnell": FLUX_REVISION,
                **trellis_metadata["model_revisions"],
            },
            "gpu_name": {
                "flux": flux_metadata["environment"]["gpu_name"],
                "trellis": trellis_metadata["runtime"]["gpu_name"],
            },
            "background_removal_applied": trellis_metadata[
                "background_removal_applied"
            ],
            "timings_seconds": timings,
            "peak_cuda_memory_bytes": {
                "flux": flux_metadata["peak_cuda_memory_bytes"],
                "trellis": trellis_metadata["peak_cuda_memory_bytes"],
            },
            "rendering": trellis_metadata["rendering"],
            "output_files": {
                "intermediate": intermediate,
                "asset_glb": glb,
                "rendered_views": rendered_views,
                "flux_metadata": str(flux_metadata_path),
                "trellis_metadata": str(trellis_metadata_path),
            },
        }
        case_summary_path = case_root / "case_summary.json"
        write_json(case_summary_path, case_summary)
        case_summary["case_summary"] = str(case_summary_path)
        cases.append(case_summary)

    batch_summary = {
        "schema_version": 1,
        "stage": "Step 3b - Text-conditioned generation",
        "status": "completed",
        "slurm_job_id": args.slurm_job_id,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "case_count": len(cases),
        "prompt_template_revision": text_spec["prompt_template_revision"],
        "timing_policy": {
            "primary_metric": "combined_primary_end_to_end",
            "excludes_model_initialization": True,
            "excludes_preview_rendering": True,
        },
        "total_primary_generation_seconds": round(
            sum(
                case["timings_seconds"]["combined_primary_end_to_end"]
                for case in cases
            ),
            4,
        ),
        "text_to_image_limitation_candidate": "txt05_lunar_rover",
        "cases": cases,
    }
    summary_path = output_root / "batch_summary.json"
    write_json(summary_path, batch_summary)

    assert len(list(output_root.glob("*/flux/intermediate.png"))) == 5
    assert len(list(output_root.glob("*/trellis/asset.glb"))) == 5
    assert len(list(output_root.glob("*/trellis/view_*.png"))) == 20
    print(f"Batch summary: {summary_path}")
    for case in cases:
        print(
            case["case_id"],
            f'{case["timings_seconds"]["combined_primary_end_to_end"]:.4f}s',
            case["output_files"]["asset_glb"]["sha256"],
        )
    print("STEP 3B TEXT-CONDITIONED BATCH: PASSED")


if __name__ == "__main__":
    main()
