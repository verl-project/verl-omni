# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainer-side SDE-window schedulers for FlowGRPO and MixGRPO.

Each scheduler is a lightweight state machine queried once per ``global_step``
to produce the SDE-window overrides injected into the rollout sampling params.
See ``docs/algo/mixgrpo.md`` for algorithm details and configuration reference.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from verl_omni.workers.config.diffusion.rollout import DiffusionRolloutAlgoConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base scheduler
# ---------------------------------------------------------------------------


class SDEWindowScheduler(ABC):
    """Abstract base class for SDE-window position schedulers."""

    @abstractmethod
    def get_window(self, global_step: int) -> dict[str, Any]:
        """Return the SDE-window overrides for *global_step*."""

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover - default no-op
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # pragma: no cover - default no-op
        return


# ---------------------------------------------------------------------------
# FlowGRPO baseline
# ---------------------------------------------------------------------------


class FlowGRPOWindowScheduler(SDEWindowScheduler):
    """Baseline FlowGRPO: forwards the static window configured by the user.

    The rollout backend itself randomises the window start uniformly inside
    :attr:`sde_window_range`.  This matches the legacy verl-omni behaviour.
    """

    def __init__(
        self,
        *,
        sde_window_size: Optional[int],
        sde_window_range: Optional[list[int]],
    ) -> None:
        self._size = sde_window_size
        self._range = list(sde_window_range) if sde_window_range is not None else None

    def get_window(self, global_step: int) -> dict[str, Any]:  # noqa: ARG002
        if self._size is None:
            return {}
        return {
            "sde_window_size": self._size,
            "sde_window_range": list(self._range) if self._range is not None else None,
        }


# ---------------------------------------------------------------------------
# MixGRPO schedulers
# ---------------------------------------------------------------------------


class _MixGRPOBase(SDEWindowScheduler):
    """Shared bookkeeping for MixGRPO sliding-window schedulers."""

    def __init__(
        self,
        *,
        group_size: int,
        init_timestep: int,
        max_timestep: int,
    ) -> None:
        if group_size is None or group_size <= 0:
            raise ValueError(
                "MixGRPO requires a positive group_size (set actor_rollout_ref.rollout.algo.sde_window_size)."
            )
        if max_timestep < init_timestep:
            raise ValueError(f"max_timestep ({max_timestep}) must be >= init_timestep ({init_timestep}).")
        if init_timestep + group_size > max_timestep + 1:
            raise ValueError(
                f"Window [{init_timestep}, {init_timestep + group_size}) does not fit in "
                f"[init_timestep={init_timestep}, max_timestep+1={max_timestep + 1}). "
                "Decrease sde_window_size or increase num_inference_steps / sde_window_range."
            )
        self.group_size = group_size
        self.init_timestep = init_timestep
        self.max_timestep = max_timestep

    @property
    def _max_window_start(self) -> int:
        """Largest valid value of ``window_start`` such that the window fits."""
        return max(self.init_timestep, self.max_timestep - self.group_size + 1)

    def _to_overrides(self, window_start: int) -> dict[str, Any]:
        window_start = max(self.init_timestep, min(self._max_window_start, window_start))
        return {
            "sde_window_size": self.group_size,
            "sde_window_range": [int(window_start), int(window_start + self.group_size)],
        }


class MixGRPORandomScheduler(_MixGRPOBase):
    """MixGRPO ``random`` strategy.

    Draws a fresh window start uniformly from
    ``[init_timestep, _max_window_start]`` using a deterministic seed
    derived from ``base_seed`` and ``global_step`` so all ranks agree on
    the window for any given step.
    """

    def __init__(
        self,
        *,
        group_size: int,
        init_timestep: int,
        max_timestep: int,
        base_seed: int = 0,
    ) -> None:
        super().__init__(
            group_size=group_size,
            init_timestep=init_timestep,
            max_timestep=max_timestep,
        )
        self.base_seed = base_seed

    def get_window(self, global_step: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.base_seed + int(global_step))
        start = int(rng.integers(self.init_timestep, self._max_window_start + 1))
        return self._to_overrides(start)


class MixGRPOProgressiveScheduler(_MixGRPOBase):
    """MixGRPO ``progressive`` strategy.

    The window starts at ``init_timestep`` and slides forward by
    ``group_size`` every ``iters_per_group`` iterations.  When the window
    reaches ``_max_window_start`` it stays clipped (training continues at
    the last position).
    """

    def __init__(
        self,
        *,
        group_size: int,
        init_timestep: int,
        max_timestep: int,
        iters_per_group: int = 1,
    ) -> None:
        super().__init__(
            group_size=group_size,
            init_timestep=init_timestep,
            max_timestep=max_timestep,
        )
        if iters_per_group <= 0:
            raise ValueError("iters_per_group must be positive.")
        self.iters_per_group = iters_per_group

    def get_window(self, global_step: int) -> dict[str, Any]:
        n_advances = max(0, int(global_step)) // self.iters_per_group
        start = self.init_timestep + n_advances * self.group_size
        return self._to_overrides(start)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_envelope(
    sde_window_range: Optional[list[int]],
    num_inference_steps: int,
) -> tuple[int, int]:
    """Return ``(init_timestep, max_timestep)`` from ``sde_window_range`` or full trajectory."""
    if sde_window_range is None:
        return 0, max(0, int(num_inference_steps) - 2)
    if len(sde_window_range) != 2:
        raise ValueError(f"sde_window_range must be a list of two ints, got {sde_window_range!r}.")
    start, end = int(sde_window_range[0]), int(sde_window_range[1])
    # ``end`` follows Python half-open convention in the rollout backend
    # (``[start, end)``); convert to the inclusive ``max_timestep`` we use here.
    # Clip to the trajectory limit so the final ODE step (sigma_prev = 0)
    # is never included in an SDE window.
    trajectory_max = max(0, int(num_inference_steps) - 2)
    return start, min(max(start, end - 1), trajectory_max)


def build_sde_window_scheduler(
    algo_config: DiffusionRolloutAlgoConfig,
    *,
    num_inference_steps: int,
) -> SDEWindowScheduler:
    """Build a :class:`SDEWindowScheduler` from a typed algo config."""
    sde_window_range = list(algo_config.sde_window_range) if algo_config.sde_window_range is not None else None

    if algo_config.algo_type == "flow_grpo":
        return FlowGRPOWindowScheduler(
            sde_window_size=algo_config.sde_window_size,
            sde_window_range=sde_window_range,
        )

    if algo_config.algo_type == "mix_grpo":
        init_timestep, max_timestep = _resolve_envelope(sde_window_range, num_inference_steps)
        common = dict(
            group_size=int(algo_config.sde_window_size),
            init_timestep=init_timestep,
            max_timestep=max_timestep,
        )
        if algo_config.sample_strategy == "random":
            return MixGRPORandomScheduler(base_seed=int(algo_config.sde_window_seed), **common)
        if algo_config.sample_strategy == "progressive":
            return MixGRPOProgressiveScheduler(iters_per_group=algo_config.iters_per_group, **common)

    raise ValueError(f"Unknown algo_type={algo_config.algo_type!r}.")


__all__ = [
    "SDEWindowOverrides",
    "SDEWindowScheduler",
    "FlowGRPOWindowScheduler",
    "MixGRPORandomScheduler",
    "MixGRPOProgressiveScheduler",
    "build_sde_window_scheduler",
]
