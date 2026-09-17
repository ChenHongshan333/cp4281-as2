#!/usr/bin/env python3
"""Capture true FlowEuler running states without modifying TRELLIS.2.

The pinned sampler appends ``pred_x_prev`` to ``pred_x_t`` after every Euler
update.  Consequently, element ``i`` in ``pred_x_t`` is the running iterate at
``flow_euler_schedule(...)[i + 1]``, not at the loop's input timestep.

This module is deliberately independent of TRELLIS.2 and PyTorch at import
time.  The Step 4 GPU runner can import it, wrap the instantiated samplers, and
provide a payload snapshot function for dense tensors and SparseTensor values.
"""

from __future__ import annotations

import argparse
import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence


STAGE1_STEPS = 24
STAGE1_RESCALE_T = 5.0
STAGE1_TARGETS = (0.5, 0.375, 0.25, 0.125, 0.0)

STAGE2_STEPS = 12
STAGE2_RESCALE_T = 3.0
STAGE2_TARGETS = (1.0, 0.75, 0.5, 0.25, 0.0)


def flow_euler_schedule(steps: int, rescale_t: float) -> tuple[float, ...]:
    """Reproduce the pinned FlowEulerSampler schedule exactly.

    The upstream implementation first creates ``linspace(1, 0, steps + 1)``
    and then applies ``r * t / (1 + (r - 1) * t)``.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if not math.isfinite(rescale_t) or rescale_t <= 0:
        raise ValueError(
            f"rescale_t must be positive and finite, got {rescale_t}"
        )

    schedule = []
    for index in range(steps + 1):
        raw_t = 1.0 - index / steps
        timestep = (
            rescale_t
            * raw_t
            / (1.0 + (rescale_t - 1.0) * raw_t)
        )
        schedule.append(timestep)

    if not math.isclose(schedule[0], 1.0, abs_tol=1e-12):
        raise AssertionError(schedule[0])
    if not math.isclose(schedule[-1], 0.0, abs_tol=1e-12):
        raise AssertionError(schedule[-1])
    if any(left <= right for left, right in zip(schedule, schedule[1:])):
        raise AssertionError("FlowEuler schedule is not strictly descending")
    return tuple(schedule)


def select_nearest_state_indices(
    returned_timesteps: Sequence[float],
    targets: Sequence[float],
    *,
    minimum: float,
    maximum: float,
) -> tuple[int, ...]:
    """Choose one distinct returned state nearest to each requested target."""
    if minimum > maximum:
        raise ValueError(f"Invalid timestep window: {minimum} > {maximum}")
    if any(left < right for left, right in zip(targets, targets[1:])):
        raise ValueError("Targets must be in descending order")

    candidates = [
        index
        for index, timestep in enumerate(returned_timesteps)
        if minimum - 1e-12 <= timestep <= maximum + 1e-12
    ]
    if len(candidates) < len(targets):
        raise ValueError(
            f"Only {len(candidates)} states fall within [{minimum}, {maximum}], "
            f"but {len(targets)} are required"
        )

    selected = []
    available = set(candidates)
    for target in targets:
        index = min(
            available,
            key=lambda item: (
                abs(returned_timesteps[item] - target),
                -returned_timesteps[item],
            ),
        )
        selected.append(index)
        available.remove(index)

    selected.sort(key=lambda item: returned_timesteps[item], reverse=True)
    return tuple(selected)


def snapshot_tensor_payload(state: Any) -> dict[str, Any]:
    """Detach and copy a dense tensor or TRELLIS SparseTensor to CPU.

    SparseTensor is stored as plain feature and coordinate tensors so no
    backend-specific sparse object remains alive after sampler return.
    """
    if hasattr(state, "feats") and hasattr(state, "coords"):
        return {
            "kind": "sparse_tensor",
            "feats": state.feats.detach().cpu().clone(),
            "coords": state.coords.detach().cpu().clone(),
        }
    if hasattr(state, "detach") and hasattr(state, "cpu"):
        tensor = state.detach().cpu()
        if hasattr(tensor, "clone"):
            tensor = tensor.clone()
        return {"kind": "dense_tensor", "tensor": tensor}
    raise TypeError(f"Unsupported sampler state type: {type(state)!r}")


@dataclass(frozen=True)
class CapturedState:
    state_index: int
    timestep: float
    target_timestep: float
    payload: Any


@dataclass(frozen=True)
class CapturedInvocation:
    stage: str
    invocation_index: int
    model_label: str
    steps: int
    rescale_t: float
    returned_timesteps: tuple[float, ...]
    selected_states: tuple[CapturedState, ...]


class SamplerCapture(AbstractContextManager["SamplerCapture"]):
    """Temporarily wrap one sampler instance and retain selected states only."""

    def __init__(
        self,
        sampler: Any,
        *,
        stage: str,
        targets: Sequence[float],
        minimum_timestep: float,
        maximum_timestep: float,
        snapshot: Callable[[Any], Any] = snapshot_tensor_payload,
        model_label: Callable[[Any], str] | None = None,
    ) -> None:
        self.sampler = sampler
        self.stage = stage
        self.targets = tuple(float(value) for value in targets)
        self.minimum_timestep = float(minimum_timestep)
        self.maximum_timestep = float(maximum_timestep)
        self.snapshot = snapshot
        self.model_label = model_label or self._default_model_label
        self.invocations: list[CapturedInvocation] = []
        self._original_sample: Callable[..., Any] | None = None
        self._had_instance_sample = False
        self._previous_instance_sample: Any = None

    @staticmethod
    def _default_model_label(model: Any) -> str:
        return type(model).__name__

    def __enter__(self) -> "SamplerCapture":
        if self._original_sample is not None:
            raise RuntimeError("SamplerCapture cannot be entered twice")

        instance_dict = vars(self.sampler)
        self._had_instance_sample = "sample" in instance_dict
        self._previous_instance_sample = instance_dict.get("sample")
        self._original_sample = self.sampler.sample

        def wrapped_sample(*args: Any, **kwargs: Any) -> Any:
            if self._original_sample is None:
                raise RuntimeError("SamplerCapture is not active")
            if not args:
                raise RuntimeError("Sampler sample() did not receive a model")

            steps = int(kwargs.get("steps", 50))
            rescale_t = float(kwargs.get("rescale_t", 1.0))
            result = self._original_sample(*args, **kwargs)
            states = tuple(result.pred_x_t)
            schedule = flow_euler_schedule(steps, rescale_t)
            returned_timesteps = schedule[1:]
            if len(states) != steps or len(states) != len(returned_timesteps):
                raise RuntimeError(
                    "Sampler output length does not match its timestep schedule: "
                    f"states={len(states)}, steps={steps}"
                )

            selected_indices = select_nearest_state_indices(
                returned_timesteps,
                self.targets,
                minimum=self.minimum_timestep,
                maximum=self.maximum_timestep,
            )
            selected_states = tuple(
                CapturedState(
                    state_index=index,
                    timestep=returned_timesteps[index],
                    target_timestep=self.targets[position],
                    payload=self.snapshot(states[index]),
                )
                for position, index in enumerate(selected_indices)
            )
            self.invocations.append(
                CapturedInvocation(
                    stage=self.stage,
                    invocation_index=len(self.invocations),
                    model_label=self.model_label(args[0]),
                    steps=steps,
                    rescale_t=rescale_t,
                    returned_timesteps=returned_timesteps,
                    selected_states=selected_states,
                )
            )

            # The pipeline only consumes result.samples. Clearing the two full
            # histories after snapshotting avoids keeping every GPU state alive.
            result.pred_x_t.clear()
            result.pred_x_0.clear()
            return result

        self.sampler.sample = wrapped_sample
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._had_instance_sample:
            self.sampler.sample = self._previous_instance_sample
        else:
            delattr(self.sampler, "sample")
        self._original_sample = None
        return None


def format_timesteps(values: Iterable[float]) -> str:
    return ", ".join(f"{value:.6f}" for value in values)


def run_self_test() -> None:
    class FakeSampler:
        def sample(
            self,
            model: str,
            noise: object,
            *,
            steps: int = 50,
            rescale_t: float = 1.0,
            **kwargs: Any,
        ) -> SimpleNamespace:
            states = [f"{model}-state-{index}" for index in range(steps)]
            return SimpleNamespace(
                samples=states[-1],
                pred_x_t=states,
                pred_x_0=[object() for _ in range(steps)],
            )

    stage1_sampler = FakeSampler()
    with SamplerCapture(
        stage1_sampler,
        stage="stage1_sparse_structure",
        targets=STAGE1_TARGETS,
        minimum_timestep=0.0,
        maximum_timestep=0.5,
        snapshot=lambda state: state,
        model_label=str,
    ) as stage1_capture:
        result = stage1_sampler.sample(
            "sparse_structure_flow_model",
            object(),
            steps=STAGE1_STEPS,
            rescale_t=STAGE1_RESCALE_T,
        )
        assert result.pred_x_t == []
        assert result.pred_x_0 == []

    stage1_times = tuple(
        state.timestep
        for state in stage1_capture.invocations[0].selected_states
    )
    expected_stage1 = (0.5, 5 / 12, 5 / 16, 5 / 28, 0.0)
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(stage1_times, expected_stage1)
    ), stage1_times

    # Restoration must remove the temporary instance wrapper.
    restored = stage1_sampler.sample(
        "restored", object(), steps=2, rescale_t=1.0
    )
    assert restored.pred_x_t == ["restored-state-0", "restored-state-1"]

    stage2_sampler = FakeSampler()
    with SamplerCapture(
        stage2_sampler,
        stage="stage2_shape_slat",
        targets=STAGE2_TARGETS,
        minimum_timestep=0.0,
        maximum_timestep=1.0,
        snapshot=lambda state: state,
        model_label=str,
    ) as stage2_capture:
        for model in (
            "shape_slat_flow_model_512",
            "shape_slat_flow_model_1024",
        ):
            stage2_sampler.sample(
                model,
                object(),
                steps=STAGE2_STEPS,
                rescale_t=STAGE2_RESCALE_T,
            )

    assert [item.model_label for item in stage2_capture.invocations] == [
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
    ]
    stage2_times = tuple(
        state.timestep
        for state in stage2_capture.invocations[-1].selected_states
    )
    expected_stage2 = (33 / 34, 3 / 4, 1 / 2, 3 / 14, 0.0)
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(stage2_times, expected_stage2)
    ), stage2_times

    default_stage1_returned = flow_euler_schedule(12, 5.0)[1:]
    try:
        select_nearest_state_indices(
            default_stage1_returned,
            STAGE1_TARGETS,
            minimum=0.0,
            maximum=0.5,
        )
    except ValueError as error:
        assert "Only 3 states" in str(error)
    else:
        raise AssertionError("Default Stage 1 schedule should be insufficient")

    print("Stage 1 selected:", format_timesteps(stage1_times))
    print("Stage 2 selected:", format_timesteps(stage2_times))
    print("Cascade invocations: 2 (512 then 1024)")
    print("STEP 4 SAMPLER CAPTURE SELF-TEST: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the dependency-free capture and schedule tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.self_test:
        raise SystemExit("No standalone action selected; use --self-test")
    run_self_test()


if __name__ == "__main__":
    main()
