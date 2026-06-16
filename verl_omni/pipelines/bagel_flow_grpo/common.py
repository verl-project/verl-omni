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

"""Shared utilities for BAGEL FlowGRPO adapters."""

import torch

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

BAGEL_TIMESTEP_SHIFT = 3.0

# CFG defaults from flow_grpo's train_bagel.py (cfg_text_scale=4.0, global renorm).
# vllm-omni defaults differ; we override to keep rollout ↔ training aligned.
BAGEL_FLOWGRPO_CFG_DEFAULTS = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 1.0,
    "cfg_interval": (0.0, 1.0),
    "cfg_renorm_type": "global",
    "cfg_renorm_min": 0.0,
}


def maybe_to_cpu(value):
    """Move a single value to CPU if it is a ``torch.Tensor``; else return unchanged."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def bagel_time_shift(shift: float, t):
    """SD3-style time shift: ``shift * t / (1 + (shift - 1) * t)``.

    Works with both ``torch.Tensor`` and ``numpy.ndarray``.
    """
    return (shift * t) / (1 + (shift - 1) * t)


def setup_bagel_sigmas(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_steps: int,
    shift: float = BAGEL_TIMESTEP_SHIFT,
    device: str | None = None,
) -> list[float]:
    """Compute shifted sigmas and configure the scheduler for BAGEL.

    ``num_steps`` is official BAGEL's ``num_timesteps`` (e.g. 50 → 49 denoise
    sigmas).  See https://github.com/vllm-project/vllm-omni/issues/4470.

    Returns the sigma list (dropping the terminal zero) for reference.
    """
    if num_steps <= 1:
        raise ValueError(f"num_steps must be > 1 for BAGEL denoising, got {num_steps}")

    t = torch.linspace(1, 0, num_steps, dtype=torch.float32, device=device or "cpu")
    t_shifted = bagel_time_shift(shift, t)
    sigmas = t_shifted[:-1].tolist()

    scheduler.set_shift(1.0)  # identity — sigmas already shifted
    if device is not None:
        scheduler.set_timesteps(sigmas=sigmas, timesteps=sigmas, device=device)
    else:
        scheduler.set_timesteps(sigmas=sigmas, timesteps=sigmas)
    scheduler.set_begin_index(0)
    return sigmas
