#!/usr/bin/env python3
"""Build deterministic labelled figures for the optional resolution comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


VIEW_NAMES = [f"view_{index:02d}.png" for index in range(4)]
DETAIL_CROPS = {
    "repeated shelves (view 02)": (220, 95, 550, 690),
    "panel seams (view 03)": (220, 95, 550, 690),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int) -> Any:
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def centred_text(draw: Any, box: tuple[int, int, int, int], text: str, face: Any) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
    width = right - left
    height = bottom - top
    x = box[0] + (box[2] - box[0] - width) / 2 - left
    y = box[1] + (box[3] - box[1] - height) / 2 - top
    draw.text((x, y), text, font=face, fill=(240, 243, 248))


def open_views(directory: Path) -> list[Any]:
    from PIL import Image

    images = []
    for name in VIEW_NAMES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        if image.size != (768, 768):
            raise ValueError(f"Unexpected image size for {path}: {image.size}")
        images.append(image)
    return images


def draw_overview(baseline: list[Any], highres: list[Any], output: Path) -> None:
    from PIL import Image, ImageDraw

    tile = 384
    left_label = 190
    top_label = 68
    gap = 12
    margin = 20
    canvas_width = margin * 2 + left_label + tile * 4 + gap * 3
    canvas_height = margin * 2 + top_label + tile * 2 + gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (23, 27, 34))
    draw = ImageDraw.Draw(canvas)
    header_font = font(24)
    row_font = font(25)

    for column in range(4):
        x = margin + left_label + column * (tile + gap)
        centred_text(
            draw,
            (x, margin, x + tile, margin + top_label),
            f"Matched view {column:02d}",
            header_font,
        )

    rows = (
        ("1024_cascade", baseline, (70, 145, 220)),
        ("1536_cascade", highres, (236, 145, 52)),
    )
    for row, (label, images, accent) in enumerate(rows):
        y = margin + top_label + row * (tile + gap)
        centred_text(draw, (margin, y, margin + left_label - 12, y + tile), label, row_font)
        for column, source in enumerate(images):
            x = margin + left_label + column * (tile + gap)
            resized = source.resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=accent, width=4)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def draw_details(baseline: list[Any], highres: list[Any], output: Path) -> None:
    from PIL import Image, ImageDraw

    crop_width = 420
    crop_height = 758
    left_label = 340
    top_label = 68
    gap = 14
    margin = 20
    canvas_width = margin * 2 + left_label + crop_width * 2 + gap
    canvas_height = margin * 2 + top_label + crop_height * 2 + gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (23, 27, 34))
    draw = ImageDraw.Draw(canvas)
    header_font = font(25)
    feature_font = font(23)

    for column, label in enumerate(("1024_cascade", "1536_cascade")):
        x = margin + left_label + column * (crop_width + gap)
        centred_text(
            draw,
            (x, margin, x + crop_width, margin + top_label),
            label,
            header_font,
        )

    details = (
        ("repeated shelves (view 02)", 2),
        ("panel seams (view 03)", 3),
    )
    for row, (label, view_index) in enumerate(details):
        y = margin + top_label + row * (crop_height + gap)
        centred_text(
            draw,
            (margin, y, margin + left_label - 12, y + crop_height),
            label,
            feature_font,
        )
        crop_box = DETAIL_CROPS[label]
        for column, source in enumerate((baseline[view_index], highres[view_index])):
            x = margin + left_label + column * (crop_width + gap)
            cropped = source.crop(crop_box).resize(
                (crop_width, crop_height), Image.Resampling.LANCZOS
            )
            canvas.paste(cropped, (x, y))
            accent = (70, 145, 220) if column == 0 else (236, 145, 52)
            draw.rectangle(
                (x, y, x + crop_width - 1, y + crop_height - 1),
                outline=accent,
                width=4,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def validate_metadata(baseline_dir: Path, highres_dir: Path) -> tuple[dict, dict]:
    baseline = load_json(baseline_dir / "metadata.json")
    highres = load_json(highres_dir / "metadata.json")
    if baseline["status"] != "completed" or highres["status"] != "completed":
        raise ValueError("Both source generations must be complete")
    if baseline["source_sha256"] != highres["source_sha256"]:
        raise ValueError("Source image hashes do not match")
    if baseline["seed"] != highres["seed"]:
        raise ValueError("Seeds do not match")
    if baseline["generation_parameters"]["pipeline_type"] != "1024_cascade":
        raise ValueError("Unexpected baseline pipeline")
    if highres["generation_parameters"]["pipeline_type"] != "1536_cascade":
        raise ValueError("Unexpected high-resolution pipeline")
    baseline_parameters = dict(baseline["generation_parameters"])
    highres_parameters = dict(highres["generation_parameters"])
    baseline_parameters.pop("pipeline_type")
    highres_parameters.pop("pipeline_type")
    if baseline_parameters != highres_parameters:
        raise ValueError("Non-resolution generation parameters do not match")
    return baseline, highres


def run(baseline_dir: Path, highres_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    baseline_metadata, highres_metadata = validate_metadata(baseline_dir, highres_dir)
    baseline_views = open_views(baseline_dir)
    highres_views = open_views(highres_dir)
    output_dir.mkdir(parents=True)

    overview_path = output_dir / "comparison_all_views.png"
    detail_path = output_dir / "comparison_detail_crops.png"
    draw_overview(baseline_views, highres_views, overview_path)
    draw_details(baseline_views, highres_views, detail_path)

    source_images = {}
    for pipeline, directory in (
        ("1024_cascade", baseline_dir),
        ("1536_cascade", highres_dir),
    ):
        source_images[pipeline] = {
            name: {
                "path": str(directory / name),
                "sha256": sha256(directory / name),
            }
            for name in VIEW_NAMES
        }

    manifest = {
        "schema_version": 1,
        "stage": "Optional extension - comparison figures",
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "case_id": "img05_complex_shape",
        "source_sha256": baseline_metadata["source_sha256"],
        "seed": baseline_metadata["seed"],
        "source_images": source_images,
        "operations": [
            "RGB conversion",
            "Lanczos downsampling for the overview",
            "identical pixel-coordinate crops for each resolution",
            "Lanczos resizing of detail crops",
            "labels and coloured borders",
        ],
        "detail_crop_boxes_xyxy": {
            label: list(box) for label, box in DETAIL_CROPS.items()
        },
        "outputs": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (overview_path, detail_path)
        },
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Overview: {overview_path}")
    print(f"Details: {detail_path}")
    print(f"Manifest: {manifest_path}")
    print("OPTIONAL HIGH-RESOLUTION COMPARISON FIGURES: PASSED")


def self_test() -> None:
    assert VIEW_NAMES == [
        "view_00.png",
        "view_01.png",
        "view_02.png",
        "view_03.png",
    ]
    assert set(DETAIL_CROPS) == {
        "repeated shelves (view 02)",
        "panel seams (view 03)",
    }
    for box in DETAIL_CROPS.values():
        left, top, right, bottom = box
        assert 0 <= left < right <= 768
        assert 0 <= top < bottom <= 768
    print("OPTIONAL COMPARISON FIGURE SELF-TEST: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--highres-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.baseline_dir is None or args.highres_dir is None or args.output_dir is None:
        raise SystemExit(
            "--baseline-dir, --highres-dir, and --output-dir are required"
        )
    run(
        args.baseline_dir.resolve(),
        args.highres_dir.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
