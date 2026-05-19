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

"""Diffusion-specific algorithm config additions for verl_omni."""

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["DiffusionAlgoConfig", "RolloutCorrectionConfig"]


@dataclass
class RolloutCorrectionConfig(BaseConfig):
    """Configuration for Rollout Correction in diffusion training.

    Mirrors ``verl.trainer.config.algorithm.RolloutCorrectionConfig`` field for
    field to keep the two stacks aligned.  See ``docs/algo/rollout_correction.md``
    for diffusion-specific usage, presets, and caveats.

    .. note::

       In **bypass mode** (``bypass_mode=True``) with ``loss_type=ppo_clip`` the
       PPO ratio ``exp(current − rollout)`` already serves as the IS correction.
       ``rollout_is`` controls *metrics reporting only*; IS weights are not
       applied to the loss.  ``rollout_rs`` rejection sampling is always applied
       regardless of mode.

       When ``loss_type=reinforce`` (reserved for future use), IS weights are
       applied explicitly because there is no PPO clipping to serve as the IS
       correction.
    """

    # --- Core mode switches (verbatim from verl) ---
    bypass_mode: bool = False
    """Skip actor old-log-prob recompute; reuse rollout_log_probs (verl-compat)."""

    loss_type: str = "ppo_clip"
    """Loss type in bypass mode: ``"ppo_clip"`` (default, IS via PPO ratio) or
    ``"reinforce"`` (reserved—IS weights applied explicitly, no PPO clipping)."""

    # --- IS weight knobs (verbatim from verl) ---
    rollout_is: Optional[str] = None
    """IS aggregation: ``null`` (off), ``"token"``, or ``"sequence"``.
    In bypass ppo_clip mode controls metrics only (IS weights not applied to loss)."""

    rollout_is_threshold: str | float = 2.0
    """IS truncation: float→TIS upper clamp, ``"lower_upper"``→IcePop."""

    rollout_is_batch_normalize: bool = False
    """Normalize IS weights to mean=1.0 across the batch."""

    # --- RS knobs (verbatim from verl) ---
    rollout_rs: Optional[str] = None
    """RS mode(s), comma-separated (e.g. ``"seq_mean_k1"``).
    See verl helper ``SUPPORTED_ROLLOUT_RS_OPTIONS`` for full list."""

    rollout_rs_threshold: Optional[str | float] = None
    """RS threshold spec. K1 modes: ``"lower_upper"``; K2/K3: single upper."""


@dataclass
class DiffusionAlgoConfig(BaseConfig):
    """Diffusion-specific algorithm config."""

    adv_estimator: str = "flow_grpo"
    norm_adv_by_std_in_grpo: bool = True
    global_std: bool = True
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
