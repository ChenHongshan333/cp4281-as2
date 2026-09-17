#!/usr/bin/env python3
"""Diagnose Step 5 GLB composition and headless rendering support.

This script never loads a generative model.  It validates the selected Step 3
GLBs, composes their named geometry nodes with the transforms in scene_spec,
round-trips a temporary GLB, and probes the installed off-screen renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import platform
import shutil
import sys
import traceback
from pathlib import Path
from queue import Empty
from typing import Any


AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_child(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes project root: {relative_path}") from error
    return candidate


def read_spec(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("stage") != "Step 5 - Composite a scene":
        raise ValueError(f"Unexpected scene-spec stage: {spec.get('stage')}")
    return spec


def validate_spec(spec: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    assets = spec.get("assets")
    if not isinstance(assets, list):
        raise ValueError("scene_spec assets must be a list")
    minimum = int(spec["acceptance"]["minimum_generated_asset_count"])
    if len(assets) < minimum:
        raise ValueError(f"Expected at least {minimum} assets, found {len(assets)}")

    asset_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for asset in assets:
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError(f"Invalid asset_id: {asset}")
        if asset_id in asset_ids:
            raise ValueError(f"Duplicate asset_id: {asset_id}")
        asset_ids.add(asset_id)

        path = safe_child(project_root, asset["source_glb"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != asset["source_sha256"]:
            raise ValueError(
                f"Source hash mismatch for {asset_id}: "
                f"expected {asset['source_sha256']}, received {actual_hash}"
            )

        scale_rule = asset.get("scale_rule", {})
        axis = scale_rule.get("axis")
        if axis not in AXIS_INDEX:
            raise ValueError(f"Invalid scale axis for {asset_id}: {axis}")
        if float(scale_rule.get("target_extent_m", 0.0)) <= 0:
            raise ValueError(f"Invalid target extent for {asset_id}")
        rotation = asset.get("rotation_euler_degrees_xyz")
        translation = asset.get("translation_m")
        if not isinstance(rotation, list) or len(rotation) != 3:
            raise ValueError(f"Invalid rotation for {asset_id}")
        if not isinstance(translation, list) or len(translation) != 3:
            raise ValueError(f"Invalid translation for {asset_id}")

        validated.append({**asset, "resolved_path": path})
    return validated


def package_versions() -> dict[str, str | None]:
    packages = (
        "trimesh",
        "numpy",
        "Pillow",
        "pyglet",
        "PyOpenGL",
        "pyrender",
        "moderngl",
        "nvdiffrast",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def module_availability() -> dict[str, bool]:
    modules = (
        "trimesh",
        "numpy",
        "PIL",
        "pyglet",
        "OpenGL",
        "pyrender",
        "moderngl",
        "nvdiffrast",
    )
    return {name: importlib.util.find_spec(name) is not None for name in modules}


def system_commands() -> dict[str, str | None]:
    return {
        name: shutil.which(name)
        for name in ("blender", "xvfb-run", "ffmpeg", "nvidia-smi")
    }


def material_identity(geometry: Any) -> str | None:
    visual = getattr(geometry, "visual", None)
    material = getattr(visual, "material", None)
    name = getattr(material, "name", None)
    return str(name) if name is not None else None


def asset_outer_transform(trimesh: Any, np: Any, scene: Any, asset: dict[str, Any]) -> Any:
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    axis_index = AXIS_INDEX[asset["scale_rule"]["axis"]]
    source_extent = float(extents[axis_index])
    if source_extent <= 0:
        raise ValueError(f"Degenerate bounds for {asset['asset_id']}: {bounds}")
    uniform_scale = float(asset["scale_rule"]["target_extent_m"]) / source_extent

    centre_floor = np.array(
        [
            (bounds[0, 0] + bounds[1, 0]) / 2.0,
            bounds[0, 1],
            (bounds[0, 2] + bounds[1, 2]) / 2.0,
        ],
        dtype=np.float64,
    )
    normalise = trimesh.transformations.translation_matrix(-centre_floor)
    scale = np.eye(4, dtype=np.float64)
    scale[0, 0] = scale[1, 1] = scale[2, 2] = uniform_scale
    angles = np.radians(asset["rotation_euler_degrees_xyz"])
    rotation = trimesh.transformations.euler_matrix(*angles, axes="sxyz")
    placement = trimesh.transformations.translation_matrix(asset["translation_m"])
    return placement @ rotation @ scale @ normalise


def compose_diagnostic_scene(
    assets: list[dict[str, Any]],
    output_path: Path,
) -> tuple[dict[str, Any], Any]:
    import numpy as np
    import trimesh

    combined = trimesh.Scene(base_frame="world")
    asset_reports: list[dict[str, Any]] = []
    for asset in assets:
        source = trimesh.load(asset["resolved_path"], force="scene", process=False)
        bounds = np.asarray(source.bounds, dtype=np.float64)
        outer = asset_outer_transform(trimesh, np, source, asset)
        materials: set[str] = set()
        added = 0
        for node_name in source.graph.nodes_geometry:
            node_transform, geometry_name = source.graph[node_name]
            geometry = source.geometry[geometry_name]
            material_name = material_identity(geometry)
            if material_name:
                materials.add(material_name)
            combined.add_geometry(
                geometry.copy(),
                node_name=f"{asset['asset_id']}/{node_name}",
                geom_name=f"{asset['asset_id']}/{geometry_name}/{added}",
                transform=outer @ node_transform,
            )
            added += 1
        if added == 0:
            raise ValueError(f"No geometry nodes found in {asset['asset_id']}")
        asset_reports.append(
            {
                "asset_id": asset["asset_id"],
                "source_bounds": bounds.tolist(),
                "source_extents": (bounds[1] - bounds[0]).tolist(),
                "geometry_nodes": added,
                "material_names": sorted(materials),
                "outer_transform": outer.tolist(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(file_obj=output_path, file_type="glb")
    reloaded = trimesh.load(output_path, force="scene", process=False)
    report = {
        "asset_count": len(assets),
        "assets": asset_reports,
        "combined_geometry_count": len(combined.geometry),
        "combined_bounds": combined.bounds.tolist(),
        "combined_extents": combined.extents.tolist(),
        "export_path": str(output_path),
        "export_size_bytes": output_path.stat().st_size,
        "export_sha256": sha256_file(output_path),
        "reload_geometry_count": len(reloaded.geometry),
        "reload_bounds": reloaded.bounds.tolist(),
    }
    if report["reload_geometry_count"] != report["combined_geometry_count"]:
        raise ValueError(f"GLB round-trip geometry mismatch: {report}")
    return report, combined


def render_worker(glb_path: str, png_path: str, result_queue: Any) -> None:
    try:
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        import trimesh

        scene = trimesh.load(glb_path, force="scene", process=False)
        image = scene.save_image(resolution=(640, 360), visible=False)
        if not image:
            raise RuntimeError("trimesh.Scene.save_image returned no bytes")
        Path(png_path).write_bytes(image)
        result_queue.put(
            {
                "status": "completed",
                "size_bytes": Path(png_path).stat().st_size,
                "sha256": sha256_file(Path(png_path)),
            }
        )
    except Exception:
        result_queue.put(
            {
                "status": "failed",
                "traceback": traceback.format_exc(),
            }
        )


def probe_trimesh_render(glb_path: Path, png_path: Path) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=render_worker,
        args=(str(glb_path), str(png_path), result_queue),
    )
    process.start()
    process.join(timeout=45)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        return {"status": "timeout", "timeout_seconds": 45}
    try:
        result = result_queue.get(timeout=2)
    except Empty:
        result = {
            "status": "failed_without_result",
            "exit_code": process.exitcode,
        }
    result["exit_code"] = process.exitcode
    return result


def run_diagnostic(spec_path: Path, project_root: Path, output_dir: Path) -> dict[str, Any]:
    spec = read_spec(spec_path)
    assets = validate_spec(spec, project_root)
    versions = package_versions()
    modules = module_availability()
    if not modules["trimesh"] or not modules["numpy"]:
        raise RuntimeError(f"Composition dependencies are unavailable: {modules}")

    diagnostic_glb = output_dir / "diagnostic_scene.glb"
    composition, _ = compose_diagnostic_scene(assets, diagnostic_glb)
    render_path = output_dir / "trimesh_preview.png"
    render_result = probe_trimesh_render(diagnostic_glb, render_path)
    report = {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "PYGLET_HEADLESS": os.environ.get("PYGLET_HEADLESS"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "package_versions": versions,
        "module_availability": modules,
        "system_commands": system_commands(),
        "composition": composition,
        "trimesh_render_probe": render_result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backend_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def run_self_test(spec_path: Path, project_root: Path) -> None:
    spec = read_spec(spec_path)
    assets = validate_spec(spec, project_root)
    assert len(assets) == spec["acceptance"]["planned_generated_asset_count"]
    assert len(assets) >= spec["acceptance"]["minimum_generated_asset_count"]
    assert spec["coordinate_system"]["up_axis"] == "Y"
    assert spec["coordinate_system"]["units"] == "metres"
    print(f"Validated generated assets: {len(assets)}")
    print("STEP 5 SCENE SPEC SELF-TEST: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.spec, args.project_root)
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --self-test is used")
    report = run_diagnostic(args.spec, args.project_root.resolve(), args.output_dir)
    print(json.dumps(report, indent=2))
    print("STEP 5 BACKEND DIAGNOSTIC: COMPLETE")


if __name__ == "__main__":
    main()
