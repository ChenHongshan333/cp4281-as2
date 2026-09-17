#!/usr/bin/env python3
"""Create report-ready labelled diffusion strips from a Step 4 output.

The frame paths, true post-update timesteps, and measurements are read from
metadata.json.  Frame hashes are verified before rendering so labels cannot be
silently paired with the wrong images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PANEL_SIZE = 320
MARGIN = 36
GAP = 18
TITLE_HEIGHT = 62
LABEL_HEIGHT = 76
BACKGROUND = (246, 248, 251)
TEXT = (24, 31, 42)
SECONDARY_TEXT = (75, 85, 99)
BORDER = (203, 210, 220)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filenames = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for filename in filenames:
        if Path(filename).is_file():
            return ImageFont.truetype(filename, size=size)
    return ImageFont.load_default()


def safe_child(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes output directory: {relative_path}") from error
    return candidate


def centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text((center_x - width / 2, y), text, font=font, fill=fill)


def measurement_label(stage: str, frame: dict[str, Any]) -> str:
    measurement = frame.get("measurement", {})
    if stage == "stage1":
        value = measurement.get("occupied_voxels")
        if not isinstance(value, int):
            raise ValueError(f"Missing occupied_voxels measurement: {frame}")
        return f"{value:,} occupied voxels"
    value = measurement.get("faces")
    if not isinstance(value, int):
        raise ValueError(f"Missing faces measurement: {frame}")
    return f"{value:,} mesh faces"


def create_strip(
    output_root: Path,
    frames: list[dict[str, Any]],
    *,
    stage: str,
    title: str,
    destination: Path,
) -> dict[str, Any]:
    if len(frames) != 5:
        raise ValueError(f"Expected five {stage} frames, found {len(frames)}")

    width = 2 * MARGIN + len(frames) * PANEL_SIZE + (len(frames) - 1) * GAP
    height = 2 * MARGIN + TITLE_HEIGHT + PANEL_SIZE + LABEL_HEIGHT
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    timestep_font = load_font(22, bold=True)
    metric_font = load_font(17)

    draw.text((MARGIN, MARGIN - 3), title, font=title_font, fill=TEXT)
    panel_y = MARGIN + TITLE_HEIGHT

    for index, frame in enumerate(frames):
        relative_path = frame.get("path")
        expected_hash = frame.get("sha256")
        timestep = frame.get("timestep")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Incomplete frame manifest entry: {frame}")
        if not isinstance(timestep, (int, float)):
            raise ValueError(f"Missing true timestep: {frame}")

        frame_path = safe_child(output_root, relative_path)
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        actual_hash = sha256_file(frame_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frame hash mismatch for {relative_path}: "
                f"expected {expected_hash}, received {actual_hash}"
            )

        with Image.open(frame_path) as opened:
            source = opened.convert("RGB")
            panel = ImageOps.fit(
                source,
                (PANEL_SIZE, PANEL_SIZE),
                method=Image.Resampling.LANCZOS,
            )

        x = MARGIN + index * (PANEL_SIZE + GAP)
        canvas.paste(panel, (x, panel_y))
        draw.rectangle(
            (x, panel_y, x + PANEL_SIZE - 1, panel_y + PANEL_SIZE - 1),
            outline=BORDER,
            width=2,
        )
        center_x = x + PANEL_SIZE // 2
        centered_text(
            draw,
            center_x,
            panel_y + PANEL_SIZE + 10,
            f"t = {float(timestep):.6f}",
            timestep_font,
            TEXT,
        )
        centered_text(
            draw,
            center_x,
            panel_y + PANEL_SIZE + 40,
            measurement_label(stage, frame),
            metric_font,
            SECONDARY_TEXT,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.png")
    canvas.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, destination)
    return {
        "path": destination.relative_to(output_root).as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "panel_count": len(frames),
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def make_strips(output_root: Path, *, force: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve()
    metadata_path = output_root / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    if metadata.get("status") != "completed":
        raise ValueError(f"Step 4 output is not completed: {metadata.get('status')}")
    rendered = metadata.get("rendered_frames", {})
    stage1 = rendered.get("stage1")
    stage2 = rendered.get("stage2")
    if not isinstance(stage1, list) or not isinstance(stage2, list):
        raise ValueError("metadata.json does not contain both rendered frame lists")

    visualization_dir = output_root / "visualizations"
    stage1_path = visualization_dir / "stage1_diffusion_strip.png"
    stage2_path = visualization_dir / "stage2_diffusion_strip.png"
    if not force:
        existing = [path for path in (stage1_path, stage2_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing strips: {existing}")

    stage1_artifact = create_strip(
        output_root,
        stage1,
        stage="stage1",
        title="Stage 1: Sparse-Structure Denoising (true timestep decreases left to right)",
        destination=stage1_path,
    )
    stage2_artifact = create_strip(
        output_root,
        stage2,
        stage="stage2",
        title="Stage 2: High-Resolution Shape Denoising (true timestep decreases left to right)",
        destination=stage2_path,
    )

    metadata["labelled_strips"] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_by": "scripts/step4_make_strips.py",
        "stage1": stage1_artifact,
        "stage2": stage2_artifact,
    }
    acceptance = metadata.setdefault("acceptance", {})
    acceptance["labelled_strips_created"] = True
    atomic_write_json(metadata_path, metadata)
    return metadata["labelled_strips"]


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="step4-strip-self-test-") as temporary:
        root = Path(temporary)
        rendered: dict[str, list[dict[str, Any]]] = {"stage1": [], "stage2": []}
        for stage in ("stage1", "stage2"):
            frame_dir = root / stage / "frames"
            frame_dir.mkdir(parents=True)
            for position in range(5):
                path = frame_dir / f"frame_{position:02d}.png"
                Image.new(
                    "RGB",
                    (64, 64),
                    (30 + position * 35, 80, 160 - position * 20),
                ).save(path)
                measurement = (
                    {"occupied_voxels": 10 * (position + 1)}
                    if stage == "stage1"
                    else {"faces": 1000 * (position + 1)}
                )
                rendered[stage].append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                        "timestep": 1.0 - position / 4,
                        "measurement": measurement,
                    }
                )
        metadata = {
            "status": "completed",
            "rendered_frames": rendered,
            "acceptance": {"labelled_strips_created": False},
        }
        atomic_write_json(root / "metadata.json", metadata)
        artifacts = make_strips(root)
        updated = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        assert artifacts["stage1"]["panel_count"] == 5
        assert artifacts["stage2"]["panel_count"] == 5
        assert updated["acceptance"]["labelled_strips_created"] is True
        for stage in ("stage1", "stage2"):
            path = safe_child(root, artifacts[stage]["path"])
            with Image.open(path) as strip:
                assert strip.size == (
                    2 * MARGIN + 5 * PANEL_SIZE + 4 * GAP,
                    2 * MARGIN + TITLE_HEIGHT + PANEL_SIZE + LABEL_HEIGHT,
                )
    print("STEP 4 LABELLED STRIP SELF-TEST: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --self-test is used")
    artifacts = make_strips(args.output_dir, force=args.force)
    print("STEP 4 LABELLED STRIPS: PASSED")
    print(f"Stage 1: {artifacts['stage1']['path']}")
    print(f"Stage 2: {artifacts['stage2']['path']}")


if __name__ == "__main__":
    main()
