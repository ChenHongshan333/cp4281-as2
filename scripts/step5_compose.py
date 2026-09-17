#!/usr/bin/env python3
"""Compose the Step 5 scene, export one GLB, and render a CUDA preview."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from step5_backend_diag import (
    asset_outer_transform,
    read_spec,
    safe_child,
    sha256_file,
    validate_spec,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def normalize(vector: Any) -> Any:
    import numpy as np

    array = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(array))
    if length <= 1e-12:
        raise ValueError(f"Cannot normalize zero-length vector: {vector}")
    return array / length


def look_at_matrix(eye: Any, target: Any, up: Any = (0.0, 1.0, 0.0)) -> Any:
    import numpy as np

    eye_array = np.asarray(eye, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    forward = normalize(target_array - eye_array)
    right = normalize(np.cross(forward, np.asarray(up, dtype=np.float64)))
    camera_up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = camera_up
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(right, eye_array))
    view[1, 3] = -float(np.dot(camera_up, eye_array))
    view[2, 3] = float(np.dot(forward, eye_array))
    return view


def perspective_matrix(
    vertical_fov_degrees: float,
    aspect: float,
    near: float,
    far: float,
) -> Any:
    import numpy as np

    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError(f"Invalid field of view: {vertical_fov_degrees}")
    if aspect <= 0.0 or near <= 0.0 or far <= near:
        raise ValueError("Invalid perspective frustum")
    focal = 1.0 / math.tan(math.radians(vertical_fov_degrees) / 2.0)
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = focal / aspect
    projection[1, 1] = focal
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2.0 * far * near) / (near - far)
    projection[3, 2] = -1.0
    return projection


def transformed_bounds(bounds: Any, transform: Any) -> Any:
    import numpy as np

    lower, upper = np.asarray(bounds, dtype=np.float64)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        [corners, np.ones((len(corners), 1), dtype=np.float64)], axis=1
    )
    transformed = homogeneous @ np.asarray(transform, dtype=np.float64).T
    points = transformed[:, :3] / transformed[:, 3:4]
    return np.asarray([points.min(axis=0), points.max(axis=0)])


def material_image(material: Any) -> Any | None:
    for attribute in ("baseColorTexture", "image"):
        image = getattr(material, attribute, None)
        if image is not None:
            return image
    return None


def material_factor(material: Any) -> Any:
    import numpy as np

    for attribute in ("baseColorFactor", "main_color", "diffuse"):
        value = getattr(material, attribute, None)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size >= 3:
            if float(array.max()) > 1.0:
                array = array / 255.0
            if array.size == 3:
                array = np.concatenate([array, np.ones(1, dtype=np.float32)])
            return np.clip(array[:4], 0.0, 1.0)
    return np.ones(4, dtype=np.float32)


def vertex_colours(geometry: Any) -> tuple[Any, str]:
    import numpy as np
    from PIL import Image

    vertex_count = len(geometry.vertices)
    visual = geometry.visual
    vertex_values = getattr(visual, "vertex_colors", None)
    if getattr(visual, "kind", None) == "vertex" and vertex_values is not None:
        colours = np.asarray(vertex_values, dtype=np.float32)
        if colours.shape[0] == vertex_count:
            if float(colours.max()) > 1.0:
                colours = colours / 255.0
            if colours.shape[1] == 3:
                colours = np.concatenate(
                    [colours, np.ones((vertex_count, 1), dtype=np.float32)],
                    axis=1,
                )
            return np.clip(colours[:, :4], 0.0, 1.0), "vertex_color"

    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = material_image(material) if material is not None else None
    factor = material_factor(material) if material is not None else np.ones(4)
    if uv is not None and image is not None:
        texture = np.asarray(Image.fromarray(np.asarray(image)).convert("RGBA"))
        coordinates = np.asarray(uv, dtype=np.float64)
        if coordinates.shape == (vertex_count, 2):
            u = np.mod(coordinates[:, 0], 1.0)
            v = np.mod(coordinates[:, 1], 1.0)
            x = np.rint(u * (texture.shape[1] - 1)).astype(np.int64)
            y = np.rint((1.0 - v) * (texture.shape[0] - 1)).astype(np.int64)
            colours = texture[y, x].astype(np.float32) / 255.0
            colours *= factor[None, :]
            return np.clip(colours, 0.0, 1.0), "texture_sampled_at_vertices"

    colours = np.broadcast_to(factor[None, :], (vertex_count, 4)).copy()
    return np.clip(colours, 0.0, 1.0), "material_factor"


def geometry_chunk(geometry: Any, transform: Any, label: str) -> dict[str, Any]:
    import numpy as np
    import trimesh

    vertices = trimesh.transform_points(geometry.vertices, transform).astype(np.float32)
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    normals = np.asarray(geometry.vertex_normals, dtype=np.float64)
    normals = (np.linalg.inv(linear).T @ normals.T).T
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = (normals / np.maximum(lengths, 1e-12)).astype(np.float32)
    colours, colour_source = vertex_colours(geometry)
    faces = np.asarray(geometry.faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Non-triangular geometry in {label}: {faces.shape}")
    return {
        "label": label,
        "vertices": vertices,
        "normals": normals,
        "colours": colours[:, :3].astype(np.float32),
        "faces": faces,
        "colour_source": colour_source,
    }


def coloured_box(trimesh: Any, size: list[float], centre: list[float], colour: list[int]) -> Any:
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(centre)
    rgba = colour if len(colour) == 4 else [*colour, 255]
    mesh.visual.vertex_colors = rgba
    return mesh


def add_environment(
    scene: Any,
    chunks: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    import trimesh

    environment = spec["procedural_environment"]
    definitions: list[tuple[str, list[float], list[float], list[int]]] = [
        ("floor", environment["floor"]["size_m"], environment["floor"]["centre_m"], [72, 79, 88, 255]),
        ("back_wall", environment["back_wall"]["size_m"], environment["back_wall"]["centre_m"], [205, 211, 219, 255]),
        ("left_wall", environment["left_wall"]["size_m"], environment["left_wall"]["centre_m"], [177, 188, 201, 255]),
        ("lounge_rug", environment["lounge_rug"]["size_m"], environment["lounge_rug"]["centre_m"], [57, 92, 111, 255]),
        ("side_table_top", environment["side_table"]["top_size_m"], environment["side_table"]["top_centre_m"], [112, 74, 47, 255]),
    ]

    top_size = environment["side_table"]["top_size_m"]
    top_centre = environment["side_table"]["top_centre_m"]
    top_bottom = float(top_centre[1]) - float(top_size[1]) / 2.0
    leg_height = top_bottom
    leg_size = [0.07, leg_height, 0.07]
    for x_sign in (-1.0, 1.0):
        for z_sign in (-1.0, 1.0):
            centre = [
                float(top_centre[0]) + x_sign * (float(top_size[0]) / 2.0 - 0.08),
                leg_height / 2.0,
                float(top_centre[2]) + z_sign * (float(top_size[2]) / 2.0 - 0.08),
            ]
            definitions.append(
                (
                    f"side_table_leg_{int((x_sign + 1) / 2)}_{int((z_sign + 1) / 2)}",
                    leg_size,
                    centre,
                    [94, 60, 40, 255],
                )
            )

    records: list[dict[str, Any]] = []
    identity = np.eye(4, dtype=np.float64)
    for name, size, centre, colour in definitions:
        mesh = coloured_box(trimesh, size, centre, colour)
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=identity)
        chunks.append(geometry_chunk(mesh, identity, name))
        records.append(
            {
                "name": name,
                "size_m": [float(value) for value in size],
                "centre_m": [float(value) for value in centre],
                "colour_rgba": colour,
            }
        )
    return records


def support_height(spec: dict[str, Any], support: str) -> float:
    environment = spec["procedural_environment"]
    if support == "floor":
        return float(spec["coordinate_system"]["ground_height"])
    if support == "lounge_rug":
        rug = environment["lounge_rug"]
        return float(rug["centre_m"][1]) + float(rug["size_m"][1]) / 2.0
    if support == "side_table":
        return float(environment["side_table"]["top_height_m"])
    raise ValueError(f"Unknown support surface: {support}")


def build_scene(
    spec: dict[str, Any],
    assets: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    import trimesh

    scene = trimesh.Scene(base_frame="world")
    chunks: list[dict[str, Any]] = []
    asset_records: list[dict[str, Any]] = []
    for asset in assets:
        source = trimesh.load(asset["resolved_path"], force="scene", process=False)
        outer = asset_outer_transform(trimesh, np, source, asset)
        placed_bounds = transformed_bounds(source.bounds, outer)
        expected_support = support_height(spec, asset["support"])
        contact_error = float(placed_bounds[0, 1] - expected_support)
        if abs(contact_error) > 1e-5:
            raise ValueError(
                f"Support contact failed for {asset['asset_id']}: {contact_error}"
            )

        geometry_records: list[dict[str, Any]] = []
        for index, node_name in enumerate(source.graph.nodes_geometry):
            node_transform, geometry_name = source.graph[node_name]
            geometry = source.geometry[geometry_name]
            world_transform = outer @ node_transform
            unique_name = f"asset/{asset['asset_id']}/{index}"
            scene.add_geometry(
                geometry.copy(),
                node_name=unique_name,
                geom_name=unique_name,
                transform=world_transform,
            )
            chunk = geometry_chunk(geometry, world_transform, unique_name)
            chunks.append(chunk)
            geometry_records.append(
                {
                    "node": unique_name,
                    "vertices": int(len(chunk["vertices"])),
                    "faces": int(len(chunk["faces"])),
                    "colour_source": chunk["colour_source"],
                }
            )

        asset_records.append(
            {
                "asset_id": asset["asset_id"],
                "case_id": asset["case_id"],
                "source_glb": asset["source_glb"],
                "source_sha256": asset["source_sha256"],
                "role": asset["role"],
                "support": asset["support"],
                "support_height_m": expected_support,
                "contact_error_m": contact_error,
                "source_bounds": source.bounds.tolist(),
                "placed_bounds": placed_bounds.tolist(),
                "outer_transform": outer.tolist(),
                "geometry": geometry_records,
            }
        )

    environment_records = add_environment(scene, chunks, spec)
    return scene, chunks, asset_records, environment_records


def concatenate_chunks(chunks: list[dict[str, Any]]) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    vertices: list[Any] = []
    normals: list[Any] = []
    colours: list[Any] = []
    faces: list[Any] = []
    offset = 0
    for chunk in chunks:
        vertices.append(chunk["vertices"])
        normals.append(chunk["normals"])
        colours.append(chunk["colours"])
        faces.append(chunk["faces"] + offset)
        offset += len(chunk["vertices"])
    return (
        np.ascontiguousarray(np.concatenate(vertices, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(normals, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(colours, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(faces, axis=0), dtype=np.int32),
    )


def render_scene(
    chunks: list[dict[str, Any]],
    camera: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np
    import nvdiffrast.torch as dr
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    width, height = [int(value) for value in camera["resolution"]]
    vertices, normals, colours, faces = concatenate_chunks(chunks)
    view = look_at_matrix(camera["position_m"], camera["target_m"])
    projection = perspective_matrix(
        float(camera["vertical_fov_degrees"]),
        width / height,
        near=0.1,
        far=100.0,
    )
    mvp = (projection @ view).astype(np.float32)

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    position = torch.from_numpy(vertices).to(device)
    normal = torch.from_numpy(normals).to(device)
    colour = torch.from_numpy(colours).to(device)
    triangles = torch.from_numpy(faces).to(device)
    homogeneous = torch.cat(
        [position, torch.ones((len(position), 1), device=device)], dim=1
    )
    matrix = torch.from_numpy(mvp).to(device)
    clip = (homogeneous @ matrix.T).unsqueeze(0).contiguous()

    context = dr.RasterizeCudaContext()
    raster, _ = dr.rasterize(
        context,
        clip,
        triangles,
        resolution=[height, width],
    )
    world_position, _ = dr.interpolate(position.unsqueeze(0), raster, triangles)
    world_normal, _ = dr.interpolate(normal.unsqueeze(0), raster, triangles)
    base_colour, _ = dr.interpolate(colour.unsqueeze(0), raster, triangles)
    world_normal = functional.normalize(world_normal, dim=-1, eps=1e-8)

    eye = torch.tensor(camera["position_m"], dtype=torch.float32, device=device)
    view_direction = functional.normalize(
        eye[None, None, None, :] - world_position,
        dim=-1,
        eps=1e-8,
    )
    facing = torch.sum(world_normal * view_direction, dim=-1, keepdim=True)
    world_normal = torch.where(facing < 0.0, -world_normal, world_normal)

    key_position = torch.tensor([4.5, 7.5, 6.0], device=device)
    fill_position = torch.tensor([-4.0, 4.0, 3.0], device=device)
    key_direction = functional.normalize(
        key_position[None, None, None, :] - world_position,
        dim=-1,
        eps=1e-8,
    )
    fill_direction = functional.normalize(
        fill_position[None, None, None, :] - world_position,
        dim=-1,
        eps=1e-8,
    )
    key = torch.clamp(torch.sum(world_normal * key_direction, dim=-1, keepdim=True), 0.0, 1.0)
    fill = torch.clamp(torch.sum(world_normal * fill_direction, dim=-1, keepdim=True), 0.0, 1.0)
    half_vector = functional.normalize(key_direction + view_direction, dim=-1, eps=1e-8)
    specular = torch.clamp(
        torch.sum(world_normal * half_vector, dim=-1, keepdim=True), 0.0, 1.0
    ) ** 40.0
    lighting = 0.38 + 0.52 * key + 0.16 * fill
    shaded = torch.clamp(base_colour * lighting + 0.10 * specular, 0.0, 1.0)

    mask = (raster[..., 3:4] > 0).to(torch.float32)
    shaded = dr.antialias(shaded, raster, clip, triangles)
    mask = dr.antialias(mask, raster, clip, triangles)
    vertical = torch.linspace(0.0, 1.0, height, device=device)[None, :, None, None]
    background_bottom = torch.tensor([0.035, 0.045, 0.065], device=device)
    background_top = torch.tensor([0.14, 0.18, 0.24], device=device)
    background = (
        background_bottom[None, None, None, :] * (1.0 - vertical)
        + background_top[None, None, None, :] * vertical
    ).expand(1, height, width, 3)
    output = shaded * mask + background * (1.0 - mask)
    output = torch.flip(output, dims=[1])
    torch.cuda.synchronize()

    pixels = (
        output[0].detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(output_path, format="PNG", optimize=True)
    return {
        "path": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "resolution": [width, height],
        "render_mode": "nvdiffrast_vertex_colour_lambert_specular",
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }


def run(
    spec_path: Path,
    project_root: Path,
    output_dir: Path,
) -> None:
    import trimesh

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "scene_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "Step 5 - Composite a scene",
        "status": "running",
        "started_at": now_iso(),
        "scene_spec": str(spec_path),
    }
    atomic_json(manifest_path, manifest)

    try:
        spec = read_spec(spec_path)
        assets = validate_spec(spec, project_root)
        scene, chunks, asset_records, environment_records = build_scene(spec, assets)
        scene_path = output_dir / "scene.glb"
        scene.export(file_obj=scene_path, file_type="glb")
        reloaded = trimesh.load(scene_path, force="scene", process=False)
        if len(reloaded.geometry) != len(scene.geometry):
            raise ValueError(
                "Scene GLB round-trip geometry mismatch: "
                f"{len(scene.geometry)} -> {len(reloaded.geometry)}"
            )

        render = render_scene(chunks, spec["camera"], output_dir / "hero_view.png")
        spec_snapshot = output_dir / "scene_spec.json"
        shutil.copy2(spec_path, spec_snapshot)
        manifest.update(
            {
                "status": "completed",
                "finished_at": now_iso(),
                "concept": spec["concept"],
                "coordinate_system": spec["coordinate_system"],
                "generated_asset_count": len(asset_records),
                "assets": asset_records,
                "procedural_environment": environment_records,
                "camera": spec["camera"],
                "scene": {
                    "path": scene_path.name,
                    "size_bytes": scene_path.stat().st_size,
                    "sha256": sha256_file(scene_path),
                    "geometry_count": len(scene.geometry),
                    "reload_geometry_count": len(reloaded.geometry),
                    "bounds": scene.bounds.tolist(),
                    "extents": scene.extents.tolist(),
                },
                "render": render,
                "software": {
                    "python": sys.version,
                    "trimesh": importlib.metadata.version("trimesh"),
                    "numpy": importlib.metadata.version("numpy"),
                    "torch": importlib.metadata.version("torch"),
                    "nvdiffrast": importlib.metadata.version("nvdiffrast"),
                    "Pillow": importlib.metadata.version("Pillow"),
                },
                "acceptance": {
                    "minimum_generated_asset_count": spec["acceptance"]["minimum_generated_asset_count"],
                    "generated_asset_count_passed": len(asset_records)
                    >= int(spec["acceptance"]["minimum_generated_asset_count"]),
                    "all_source_hashes_verified": True,
                    "all_support_contacts_verified": all(
                        abs(float(asset["contact_error_m"])) <= 1e-5
                        for asset in asset_records
                    ),
                    "single_scene_file_created": True,
                    "rendered_view_count": 1,
                },
            }
        )
        atomic_json(manifest_path, manifest)
        print("STEP 5 FORMAL SCENE COMPOSITION: PASSED")
        print(f"Scene: {scene_path}")
        print(f"Render: {output_dir / 'hero_view.png'}")
        print(f"Manifest: {manifest_path}")
    except Exception as error:
        manifest.update(
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
        atomic_json(manifest_path, manifest)
        raise


def run_self_test(spec_path: Path, project_root: Path) -> None:
    import numpy as np

    spec = read_spec(spec_path)
    assets = validate_spec(spec, project_root)
    camera = spec["camera"]
    width, height = camera["resolution"]
    view = look_at_matrix(camera["position_m"], camera["target_m"])
    projection = perspective_matrix(
        camera["vertical_fov_degrees"], width / height, 0.1, 100.0
    )
    eye = np.asarray([*camera["position_m"], 1.0])
    transformed_eye = view @ eye
    assert np.allclose(transformed_eye[:3], 0.0, atol=1e-8)
    assert np.isfinite(projection).all()
    supports = {asset["support"] for asset in assets}
    assert supports == {"floor", "lounge_rug", "side_table"}
    for asset in assets:
        assert math.isclose(
            float(asset["translation_m"][1]),
            support_height(spec, asset["support"]),
            abs_tol=1e-12,
        )
    print(f"Validated generated assets: {len(assets)}")
    print("Camera matrix: PASSED")
    print("Support heights: PASSED")
    print("STEP 5 FORMAL COMPOSER SELF-TEST: PASSED")


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
        run_self_test(args.spec, args.project_root.resolve())
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --self-test is used")
    run(args.spec.resolve(), args.project_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
