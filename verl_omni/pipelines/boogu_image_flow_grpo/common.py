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

"""Shared helpers for the Boogu-Image FlowGRPO adapters.

The rollout adapter (vllm-omni) and the training adapter (diffusers) must
handle Boogu-Image with identical conventions — otherwise the rollout
trajectory and the training-time log-probs diverge and RL silently breaks.
Those shared conventions live in this module.

Convention differences
----------------------

Boogu-Image runs flow matching "backwards" relative to diffusers:

                        Boogu-Image            diffusers (FlowMatchSDEDiscreteScheduler)
  timestep direction    t: 0 -> 1 (0 = noise)  sigma: 1 -> 0 (1 = noise)
  velocity target       v_boogu = x0 - noise   v_diffusers = noise - x0

Both adapters therefore apply the same two fix-ups:

1. Timestep mapping: t = 1 - sigma, i.e. t = 1 - timestep / num_train_timesteps
   for the scheduler-native timestep values that ride the rollout trajectory
   (see boogu_timestep_from_scheduler).
2. Velocity negation: the transformer output is negated before it enters the
   scheduler. Under this mapping Boogu's Euler update x + (t_next - t) * v_boogu
   is identical to the diffusers update x + (sigma_prev - sigma) * v_diffusers.

Time shift
----------

No scheduler port is needed for Boogu's time shift either, because both
variants map exactly onto knobs the stock scheduler already has:

- the v1 logistic shift is exactly the diffusers exp-mu shift
  exp(mu) / (exp(mu) + (1/sigma - 1));
- the v2 rational shift is exactly the diffusers multiplicative shift
  m * sigma / (1 + (m - 1) * sigma).

resolve_time_shift recovers the right mu / shift from the raw checkpoint
config, so FlowMatchSDEDiscreteScheduler reproduces the original schedule
unchanged. The released checkpoints use a static v1 shift
(do_shift=true, dynamic_time_shift=false, time_shift_version="v1", seq_len=4096).
"""

import json
import math
import os
from typing import Any, Optional

import numpy as np
import torch

# Latent-token count of a 1024x1024 image at vae_scale_factor 8 divided by 4
# (the v1 shift operates on (H_lat // 2) * (W_lat // 2) tokens).
_V1_LIN_X1 = 256
_V1_LIN_X2 = 4096


def _lin_mu(seq_len: int, base_shift: float, max_shift: float) -> float:
    """Boogu's linear token-count -> mu mapping (mirrors upstream training)."""
    m = (max_shift - base_shift) / (_V1_LIN_X2 - _V1_LIN_X1)
    b = base_shift - m * _V1_LIN_X1
    return m * seq_len + b


def load_boogu_shift_config(model_path: str) -> dict:
    """Read the raw scheduler/scheduler_config.json from the checkpoint.

    The Boogu time-shift keys (do_shift / dynamic_time_shift /
    time_shift_version / seq_len / ...) are not attributes of the
    diffusers FlowMatchSDEDiscreteScheduler, so from_pretrained
    silently drops them from scheduler.config ("were passed ... but are
    not expected and will be ignored"). Reading the JSON directly is the only
    way to recover them.
    """
    config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def resolve_time_shift(
    scheduler_config: Any,
    num_tokens: Optional[int] = None,
) -> tuple[Optional[float], float]:
    """Resolve Boogu's time-shift config into diffusers scheduler terms.

    Args:
        scheduler_config: The raw checkpoint scheduler config mapping
            (from load_boogu_shift_config), carrying Boogu's do_shift /
            dynamic_time_shift / time_shift_version / seq_len keys.
            Do not pass scheduler.config — diffusers drops the custom
            keys there.
        num_tokens: Per-sample latent token count H_lat * W_lat; only
            consulted by the dynamic variants.

    Returns:
        (mu, shift) where mu is the exp-shift exponent for
        set_timesteps(mu=...) (None when the v2 / no-shift path is
        active) and shift is the multiplicative shift for the static
        scheduler path (1.0 = disabled). Exactly one of the two is
        meaningful.
    """
    get = (
        scheduler_config.get if hasattr(scheduler_config, "get") else lambda k, d=None: getattr(scheduler_config, k, d)
    )

    # An empty mapping means the Boogu keys could not be read (e.g. someone
    # passed the diffusers-filtered scheduler.config or the JSON was
    # missing). Degrade to a no-op schedule rather than silently applying the
    # dynamic-v2 defaults, which do not match the released checkpoints.
    if isinstance(scheduler_config, dict) and not scheduler_config:
        return None, 1.0

    if not get("do_shift", True):
        return None, 1.0

    version = get("time_shift_version", "v2")
    dynamic = get("dynamic_time_shift", True)
    base_shift = get("base_shift", 0.5)
    max_shift = get("max_shift", 1.15)
    seq_len = get("seq_len", None)
    half_scaling = get("time_shift_v2_half_scaling_factor", 60.0)

    if version == "v1":
        if dynamic:
            if num_tokens is None or num_tokens <= 0:
                return None, 1.0
            tokens_reduced = max(1, int(num_tokens) // 4)
            return _lin_mu(tokens_reduced, base_shift, max_shift), 1.0
        if seq_len is None or seq_len <= 0:
            return None, 1.0
        return _lin_mu(int(seq_len), base_shift, max_shift), 1.0

    if version == "v2":
        tokens = num_tokens if dynamic else seq_len
        if tokens is None or tokens <= 0:
            return None, 1.0
        return None, float(math.sqrt(tokens)) / (half_scaling * 2)

    raise ValueError(f"Unknown Boogu time_shift_version: {version!r}")


def configure_boogu_sde_timesteps(
    scheduler,
    *,
    num_inference_steps: int,
    num_tokens: int,
    device,
    shift_config: dict,
) -> None:
    """Set SDE-scheduler timesteps reproducing the checkpoint's Boogu schedule.

    shift_config is the raw scheduler config from load_boogu_shift_config
    (the diffusers scheduler.config has already dropped Boogu's custom
    keys). num_tokens is the latent pixel count H_lat * W_lat (only
    consulted by the dynamic-shift variants).
    """
    mu, shift = resolve_time_shift(shift_config, num_tokens)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if mu is not None:
        scheduler.register_to_config(use_dynamic_shifting=True)
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
    else:
        scheduler.register_to_config(use_dynamic_shifting=False, shift=shift)
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)


def boogu_timestep_from_scheduler(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Map scheduler-native timestep values (sigma * N, descending) to Boogu t (1 - sigma)."""
    return 1.0 - timestep.float() / num_train_timesteps


def apply_boogu_text_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Boogu's sequential text CFG (standard formula, no renormalisation)."""
    return noise_pred + (guidance_scale - 1.0) * (noise_pred - negative_noise_pred)


_FREQS_CIS_CACHE: dict[tuple, Any] = {}


def get_boogu_freqs_cis(axes_dim_rope, axes_lens, theta: int = 10000):
    """Build (and cache) the rotary tables the Boogu transformer consumes.

    Prefers the canonical implementation from the installed boogu-image
    package (the training-side transformer is the canonical class, so its
    rope tables must come from the same code); falls back to the verbatim
    vllm-omni port, which the rollout-side transformer uses.
    """
    key = (tuple(axes_dim_rope), tuple(axes_lens), theta)
    if key in _FREQS_CIS_CACHE:
        return _FREQS_CIS_CACHE[key]

    try:
        from boogu.models.transformers.rope import BooguImageRotaryPosEmbed as _Rope
    except ImportError:
        from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
            BooguImageDoubleStreamRotaryPosEmbed as _Rope,
        )

    freqs_cis = _Rope.get_freqs_cis(list(axes_dim_rope), list(axes_lens), theta=theta)
    _FREQS_CIS_CACHE[key] = freqs_cis
    return freqs_cis


def resolve_text_guidance_scale(guidance_scale: Optional[float]) -> float:
    """Map a possibly-unset config guidance scale to Boogu's default (4.0)."""
    return 4.0 if guidance_scale is None else float(guidance_scale)
