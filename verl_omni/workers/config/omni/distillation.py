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

Subclass verl's ``DistillationTeacherModelConfig`` to teach the on-policy
distillation path about the ``vllm_omni`` rollout engine:
"""

from dataclasses import dataclass
from typing import Optional

from verl.workers.config import DistillationTeacherModelConfig

__all__ = ["OmniDistillationTeacherModelConfig"]


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
