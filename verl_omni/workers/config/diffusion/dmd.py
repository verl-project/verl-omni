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

import math
from dataclasses import dataclass, field
from functools import partial

from verl.base_config import BaseConfig
from verl.workers.config.optimizer import FSDPOptimizerConfig

__all__ = ["DiffusionDMDConfig"]


@dataclass
class DiffusionDMDConfig(BaseConfig):
    """DMD2 distribution-only settings, independent of on-policy distillation."""

    # Generator sampling steps, independent of the score-training timestep.
    num_inference_steps: int = 4
    # Explicit per-modality flow shifts and loss weights for continuous sampling.
    modality_shifts: dict[str, float] = field(default_factory=dict)
    modality_weights: dict[str, float] = field(default_factory=dict)
    # Fake-score update attempts per student attempt.
    fake_update_ratio: int = 2
    # Student physical microbatch size per data-parallel rank.
    student_micro_batch_size_per_gpu: int = 1
    # Fake-score physical microbatch size per data-parallel rank.
    fake_score_micro_batch_size_per_gpu: int = 1
    # Fake score uses the existing optimizer type, independently of the actor.
    fake_score_optim: FSDPOptimizerConfig = field(
        default_factory=partial(FSDPOptimizerConfig, lr=2e-5, weight_decay=0.001)
    )
    # Qwen teacher CFG; student and fake score remain conditional-only.
    teacher_guidance_scale: float = 4.0
    # Teacher CFG normalization; DMD2Program layer_norm requires (B, tokens, features).
    cfg_norm: str = "layer_norm"
    # Explicit negative teacher condition; an empty string is also valid.
    negative_prompt: str = " "
    # Per-sample score-difference normalization floor.
    normalization_epsilon: float = 1e-6
    # Fixed rational shift applied once to the inference sigma grid.
    rollout_timestep_shift: float = 3.0
    # Discrete score grid size; zero selects continuous score sampling.
    score_discrete_steps: int = 1000
    # Lower continuous base-time bound, before DMD2Program's per-modality shifts.
    score_time_min: float = 0.02
    # Upper continuous base-time bound, before DMD2Program's per-modality shifts.
    score_time_max: float = 0.98
    # Lower score sigma bound.
    score_sigma_min: float = 0.02
    # Upper score sigma bound.
    score_sigma_max: float = 0.98
    # Rational shift for discrete score sampling only.
    score_timestep_shift: float = 3.0
    # EMA decay, applied only after successful student updates.
    ema_decay: float = 0.999
    # Successful student update count at which EMA starts.
    ema_start_step: int = 0
    # Default inference artifact; EMA is an explicit alternative.
    export_role: str = "student"

    def __post_init__(self):
        if (
            isinstance(self.num_inference_steps, bool)
            or not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps < 1
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        if self.modality_shifts.keys() != self.modality_weights.keys():
            raise ValueError("modality_shifts and modality_weights must have matching keys")
        for name, minimum, inclusive in (("modality_shifts", 1, True), ("modality_weights", 0, False)):
            for value in getattr(self, name).values():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    or (value < minimum if inclusive else value <= minimum)
                ):
                    raise ValueError(f"Invalid {name}: shifts must be finite and >= 1, weights finite and > 0")
        for name in ("fake_update_ratio", "student_micro_batch_size_per_gpu", "fake_score_micro_batch_size_per_gpu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("score_discrete_steps", "ema_start_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        for name in (
            "teacher_guidance_scale",
            "normalization_epsilon",
            "rollout_timestep_shift",
            "score_time_min",
            "score_time_max",
            "score_sigma_min",
            "score_sigma_max",
            "score_timestep_shift",
            "ema_decay",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.teacher_guidance_scale <= 0:
            raise ValueError("teacher_guidance_scale must be positive")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be an explicit string for teacher conditioning")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if self.rollout_timestep_shift < 1 or self.score_timestep_shift < 1:
            raise ValueError("rollout_timestep_shift and score_timestep_shift must be at least 1")
        for prefix in ("score_sigma", "score_time"):
            if not 0 < getattr(self, f"{prefix}_min") < getattr(self, f"{prefix}_max") <= 1:
                raise ValueError(f"{prefix} bounds must satisfy 0 < {prefix}_min < {prefix}_max <= 1")
        if not 0 <= self.ema_decay <= 1:
            raise ValueError("ema_decay must be between 0 and 1")
        valid_cfg_norms = {"none", "layer_norm", "scalar"}
        if self.cfg_norm not in valid_cfg_norms:
            raise ValueError(f"Invalid cfg_norm: {self.cfg_norm}. Must be one of {sorted(valid_cfg_norms)}")
        valid_export_roles = {"student", "student_ema"}
        if self.export_role not in valid_export_roles:
            raise ValueError(f"Invalid export_role: {self.export_role}. Must be one of {sorted(valid_export_roles)}")
