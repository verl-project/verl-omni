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

"""Qwen-Image query projection compatibility for VeOmni native LoRA."""

import torch
from veomni.lora.layers import LoraLinear


class _QwenImageLoraQuery(LoraLinear):
    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight


def configure_lora_queries(transformer: torch.nn.Module) -> None:
    for block in transformer.transformer_blocks:
        query = block.attn.to_q
        if type(query) is LoraLinear and not hasattr(query, "weight"):
            # VeOmni 0.1.14 reads to_q.weight.dtype. Keep the injected parameters and their checkpoint keys.
            query.__class__ = _QwenImageLoraQuery
