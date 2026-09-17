#!/usr/bin/env python3
"""Decode and render the two TRELLIS.2 geometry-stage trajectories.

Stage 1 uses the dense sparse-structure decoder and renders thresholded
occupancy voxels.  Stage 2 reconstructs a SparseTensor, reverses the model's
shape-SLat normalization, decodes a mesh, and renders its surface normals.

PyTorch and TRELLIS.2 imports are intentionally lazy so schedule/manifest
self-tests can run on the local Windows machine without the cluster runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from scripts.step4_capture import CapturedInvocation
except ModuleNotFoundError:
    from step4_capture import CapturedInvocation


STAGE1_MODEL = "sparse_structure_flow_model"
STAGE2_LR_MODEL = "shape_slat_flow_model_512"
STAGE2_HR_MODEL = "shape_slat_flow_model_1024"


@dataclass(frozen=True)
class RenderCamera:
    resolution: int = 512
    radius: float = 2.0
    fov_degrees: float = 36.0
    yaw_degrees: float = -16.0
    pitch_degrees: float = 20.0
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DecodedFrame:
    stage: str
    invocation_index: int
    model_label: str
    state_index: int
    timestep: float
    measurement: dict[str, int | float]
    render_mode: str
    image: Any


def decoder_pool_ratio(
    decoder_resolution: int,
    structure_resolution: int,
) -> int:
    """Return the max-pooling ratio used by the pinned upstream pipeline."""
    if decoder_resolution <= 0 or structure_resolution <= 0:
        raise ValueError("Resolutions must be positive")
    if decoder_resolution < structure_resolution:
        raise ValueError(
            "Sparse-structure decoder output cannot be smaller than the "
            "requested structure resolution"
        )
    if decoder_resolution % structure_resolution:
        raise ValueError(
            f"Decoder resolution {decoder_resolution} is not divisible by "
            f"structure resolution {structure_resolution}"
        )
    return decoder_resolution // structure_resolution


def select_high_resolution_shape_invocation(
    invocations: Sequence[CapturedInvocation],
) -> CapturedInvocation:
    """Select the final 1024-resolution shape pass from a cascade run."""
    matches = [
        invocation
        for invocation in invocations
        if invocation.model_label == STAGE2_HR_MODEL
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one high-resolution shape invocation; "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_five_states(invocation: CapturedInvocation) -> None:
    if len(invocation.selected_states) != 5:
        raise ValueError(
            f"{invocation.stage} invocation {invocation.invocation_index} "
            f"contains {len(invocation.selected_states)} selected states, not 5"
        )
    timesteps = [state.timestep for state in invocation.selected_states]
    if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
        raise ValueError(f"Timesteps are not strictly descending: {timesteps}")


def _require_payload(payload: Any, kind: str, fields: Sequence[str]) -> None:
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise TypeError(f"Expected {kind} payload, got {type(payload)!r}: {payload}")
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"Payload is missing fields: {missing}")


def _camera_offset(camera: RenderCamera) -> tuple[float, float]:
    import math

    return (
        math.radians(camera.yaw_degrees),
        math.radians(camera.pitch_degrees),
    )


def _blank_frame(camera: RenderCamera) -> Any:
    import numpy as np

    color = np.asarray(camera.background, dtype=np.float32)
    return np.broadcast_to(
        color[None, None, :] * 255.0,
        (camera.resolution, camera.resolution, 3),
    ).astype(np.uint8).copy()


def decode_stage1_invocation(
    pipeline: Any,
    invocation: CapturedInvocation,
    *,
    structure_resolution: int,
    camera: RenderCamera = RenderCamera(),
) -> list[DecodedFrame]:
    """Decode five dense flow states to occupancy and render fixed-view voxels."""
    import torch
    import torch.nn.functional as functional

    from trellis2.representations import Voxel
    from trellis2.utils import render_utils

    validate_five_states(invocation)
    if invocation.model_label != STAGE1_MODEL:
        raise ValueError(
            f"Unexpected Stage 1 model label: {invocation.model_label}"
        )

    decoder = pipeline.models["sparse_structure_decoder"]
    frames: list[DecodedFrame] = []
    if pipeline.low_vram:
        decoder.to(pipeline.device)

    try:
        with torch.no_grad():
            for selected in invocation.selected_states:
                _require_payload(
                    selected.payload,
                    "dense_tensor",
                    ("tensor",),
                )
                latent = selected.payload["tensor"].to(pipeline.device)
                occupancy = decoder(latent) > 0
                if (
                    occupancy.ndim != 5
                    or occupancy.shape[0] != 1
                    or occupancy.shape[1] != 1
                ):
                    raise RuntimeError(
                        "Expected Stage 1 occupancy with shape [1, 1, D, H, W], "
                        f"got {tuple(occupancy.shape)}"
                    )
                if not (
                    occupancy.shape[2]
                    == occupancy.shape[3]
                    == occupancy.shape[4]
                ):
                    raise RuntimeError(
                        f"Occupancy grid is not cubic: {tuple(occupancy.shape)}"
                    )

                ratio = decoder_pool_ratio(
                    int(occupancy.shape[2]),
                    structure_resolution,
                )
                if ratio != 1:
                    occupancy = (
                        functional.max_pool3d(
                            occupancy.float(), ratio, ratio, 0
                        )
                        > 0.5
                    )

                # Match upstream axis selection: omit channel, retain XYZ.
                indexed = torch.argwhere(occupancy)[:, [0, 2, 3, 4]]
                indexed = indexed[indexed[:, 0] == 0]
                coords = indexed[:, 1:].int().contiguous()
                occupied_voxels = int(coords.shape[0])
                occupancy_fraction = occupied_voxels / float(
                    structure_resolution**3
                )

                if occupied_voxels == 0:
                    image = _blank_frame(camera)
                    render_mode = "blank_empty_occupancy"
                else:
                    colors = coords.float() / float(structure_resolution)
                    voxel = Voxel(
                        origin=[-0.5, -0.5, -0.5],
                        voxel_size=1 / structure_resolution,
                        coords=coords,
                        attrs=colors,
                        layout={"color": slice(0, 3)},
                        device=pipeline.device,
                    )
                    rendered = render_utils.render_snapshot(
                        voxel,
                        resolution=camera.resolution,
                        bg_color=camera.background,
                        offset=_camera_offset(camera),
                        r=camera.radius,
                        fov=camera.fov_degrees,
                        nviews=1,
                        colors_overwrite=colors,
                    )
                    image = rendered["color"][0]
                    render_mode = "occupancy_voxel_color"

                frames.append(
                    DecodedFrame(
                        stage=invocation.stage,
                        invocation_index=invocation.invocation_index,
                        model_label=invocation.model_label,
                        state_index=selected.state_index,
                        timestep=selected.timestep,
                        measurement={
                            "occupied_voxels": occupied_voxels,
                            "occupancy_fraction": occupancy_fraction,
                            "structure_resolution": structure_resolution,
                        },
                        render_mode=render_mode,
                        image=image,
                    )
                )
                del latent, occupancy, indexed, coords
                torch.cuda.empty_cache()
    finally:
        if pipeline.low_vram:
            decoder.cpu()

    return frames


def decode_stage2_invocation(
    pipeline: Any,
    invocation: CapturedInvocation,
    *,
    shape_resolution: int,
    camera: RenderCamera = RenderCamera(),
) -> list[DecodedFrame]:
    """Denormalize five shape-SLat states, decode meshes, and render normals."""
    import torch

    from trellis2.modules.sparse import SparseTensor
    from trellis2.utils import render_utils

    validate_five_states(invocation)
    if invocation.model_label != STAGE2_HR_MODEL:
        raise ValueError(
            "Formal Stage 2 renders must come from the 1024-resolution shape "
            f"pass, not {invocation.model_label}"
        )

    decoder = pipeline.models["shape_slat_decoder"]
    decoder.set_resolution(shape_resolution)
    had_decoder_low_vram = hasattr(decoder, "low_vram")
    original_low_vram = getattr(decoder, "low_vram", None)
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True

    frames: list[DecodedFrame] = []
    try:
        with torch.no_grad():
            for selected in invocation.selected_states:
                _require_payload(
                    selected.payload,
                    "sparse_tensor",
                    ("feats", "coords"),
                )
                slat = SparseTensor(
                    feats=selected.payload["feats"].to(pipeline.device),
                    coords=selected.payload["coords"].to(pipeline.device),
                )
                std = torch.tensor(
                    pipeline.shape_slat_normalization["std"],
                )[None].to(slat.device)
                mean = torch.tensor(
                    pipeline.shape_slat_normalization["mean"],
                )[None].to(slat.device)
                slat = slat * std + mean

                try:
                    meshes, subs = decoder(slat, return_subs=True)
                except Exception as error:
                    raise RuntimeError(
                        "Stage 2 decoder failed for the real sampler state at "
                        f"t={selected.timestep:.9f}"
                    ) from error
                if len(meshes) != 1:
                    raise RuntimeError(
                        f"Expected one decoded mesh, got {len(meshes)}"
                    )
                mesh = meshes[0]
                vertex_count = int(mesh.vertices.shape[0])
                face_count = int(mesh.faces.shape[0])
                if vertex_count == 0 or face_count == 0:
                    raise RuntimeError(
                        "Stage 2 produced an empty mesh at "
                        f"t={selected.timestep:.9f}"
                    )

                rendered = render_utils.render_snapshot(
                    mesh,
                    resolution=camera.resolution,
                    bg_color=camera.background,
                    offset=_camera_offset(camera),
                    r=camera.radius,
                    fov=camera.fov_degrees,
                    nviews=1,
                )
                if "normal" not in rendered:
                    raise RuntimeError(
                        f"MeshRenderer did not return normals: {rendered.keys()}"
                    )
                frames.append(
                    DecodedFrame(
                        stage=invocation.stage,
                        invocation_index=invocation.invocation_index,
                        model_label=invocation.model_label,
                        state_index=selected.state_index,
                        timestep=selected.timestep,
                        measurement={
                            "vertices": vertex_count,
                            "faces": face_count,
                            "shape_resolution": shape_resolution,
                            "sparse_tokens": int(slat.feats.shape[0]),
                        },
                        render_mode="mesh_surface_normals",
                        image=rendered["normal"][0],
                    )
                )
                del slat, meshes, subs, mesh
                torch.cuda.empty_cache()
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            if had_decoder_low_vram:
                decoder.low_vram = original_low_vram
            else:
                delattr(decoder, "low_vram")

    return frames


def frame_manifest(frame: DecodedFrame) -> dict[str, Any]:
    """Return the JSON-safe part of a decoded frame record."""
    return {
        "stage": frame.stage,
        "invocation_index": frame.invocation_index,
        "model_label": frame.model_label,
        "state_index": frame.state_index,
        "timestep": frame.timestep,
        "measurement": frame.measurement,
        "render_mode": frame.render_mode,
    }


def run_self_test() -> None:
    from step4_capture import CapturedInvocation, CapturedState

    selected = tuple(
        CapturedState(
            state_index=index,
            timestep=timestep,
            target_timestep=timestep,
            payload={"kind": "test"},
        )
        for index, timestep in enumerate((0.9, 0.7, 0.5, 0.2, 0.0))
    )
    low = CapturedInvocation(
        stage="stage2_shape_slat",
        invocation_index=0,
        model_label=STAGE2_LR_MODEL,
        steps=12,
        rescale_t=3.0,
        returned_timesteps=(),
        selected_states=selected,
    )
    high = CapturedInvocation(
        stage="stage2_shape_slat",
        invocation_index=1,
        model_label=STAGE2_HR_MODEL,
        steps=12,
        rescale_t=3.0,
        returned_timesteps=(),
        selected_states=selected,
    )
    assert select_high_resolution_shape_invocation([low, high]) is high
    validate_five_states(high)
    assert decoder_pool_ratio(64, 32) == 2
    assert decoder_pool_ratio(32, 32) == 1

    try:
        decoder_pool_ratio(48, 32)
    except ValueError as error:
        assert "not divisible" in str(error)
    else:
        raise AssertionError("Invalid decoder/structure ratio was accepted")

    manifest = frame_manifest(
        DecodedFrame(
            stage="stage1_sparse_structure",
            invocation_index=0,
            model_label=STAGE1_MODEL,
            state_index=19,
            timestep=0.5,
            measurement={"occupied_voxels": 123},
            render_mode="occupancy_voxel_color",
            image=object(),
        )
    )
    assert "image" not in manifest
    assert manifest["measurement"]["occupied_voxels"] == 123
    print("High-resolution cascade selection: PASSED")
    print("Sparse occupancy resolution validation: PASSED")
    print("JSON-safe frame manifest: PASSED")
    print("STEP 4 DECODER/RENDER HELPER SELF-TEST: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run dependency-free decoder helper tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.self_test:
        raise SystemExit("No standalone action selected; use --self-test")
    run_self_test()


if __name__ == "__main__":
    main()
