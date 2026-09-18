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
"""Omni distillation configs.

Subclass verl's distillation configs to teach the on-policy distillation
path about (a) the ``vllm_omni`` rollout engine and (b) the hidden-state
(nitrobrew) loss modes, which consume teacher hidden states instead of
top-k logprobs and add ``kd_temperature``.
"""

from dataclasses import dataclass
from typing import Optional

from verl.trainer.distillation.losses import DistillationLossSettings
from verl.workers.config import DistillationLossConfig, DistillationTeacherModelConfig

__all__ = ["OmniDistillationTeacherModelConfig", "OmniDistillationLossConfig"]

# Loss modes that consume per-position teacher hidden states.
HIDDEN_STATE_LOSS_MODES = ("nitrobrew", "nitrobrew_reverse_kl")


@dataclass
class OmniDistillationTeacherModelConfig(DistillationTeacherModelConfig):
    """Teacher config that also accepts ``inference.name == "vllm_omni"``."""

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        if self.inference.name != "vllm_omni":
            raise ValueError(f"the inference.name should be 'vllm_omni', got {self.inference.name}")

        engine_kwargs = self.inference.engine_kwargs
        omni_engine_kwargs = dict(engine_kwargs.get("vllm_omni", {}))
        max_logprobs = omni_engine_kwargs.get("max_logprobs")
        if max_logprobs is None:
            omni_engine_kwargs["max_logprobs"] = topk
            max_logprobs = topk
        if max_logprobs < topk:
            raise ValueError(
                f"vllm_omni max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                f"({topk}) to enable distillation loss computation."
            )
        engine_kwargs["vllm_omni"] = omni_engine_kwargs


@dataclass
class OmniDistillationLossConfig(DistillationLossConfig):
    """Loss config for the omni path.

    Adds ``kd_temperature`` and teaches the hidden-state (nitrobrew) loss
    modes: those are absent from verl's registry (which only knows top-k /
    estimator modes), so their ``loss_settings`` are synthesized here with
    ``use_hidden_states=True`` instead of being looked up.
    """

    kd_temperature: float = 1.0

    def __post_init__(self):
        if self.loss_mode in HIDDEN_STATE_LOSS_MODES:
            self._mutable_fields.add("loss_settings")
            settings = object.__new__(DistillationLossSettings)
            object.__setattr__(settings, "names", [self.loss_mode])
            object.__setattr__(settings, "use_topk", False)
            object.__setattr__(settings, "use_estimator", False)
            object.__setattr__(settings, "use_hidden_states", True)
            self.loss_settings = settings

            if self.policy_loss_mode != "vanilla":
                raise NotImplementedError(
                    f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                    f"but got {self.policy_loss_mode}."
                )
            if self.use_policy_gradient and self.loss_mode in ("nitrobrew",):
                raise ValueError(
                    "nitrobrew full-vocabulary KL is most effective as a supervised distillation loss "
                    "(use_policy_gradient=False), so the whole-vocab signal is backpropagated directly. "
                    "Set distillation.distillation_loss.use_policy_gradient=false."
                )
            return

        # Non-hidden modes delegate to verl's registry + validations.
        super().__post_init__()
