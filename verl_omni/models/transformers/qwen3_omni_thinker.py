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
"""Compatibility external_lib for Qwen3-Omni thinker-only training."""

from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import (
    Qwen3OmniThinkerAdapter,
    patch_hf_processor_for_qwen3_omni,
    patch_hf_tokenizer_for_qwen3_omni,
)

__all__ = [
    "Qwen3OmniThinkerAdapter",
    "apply_qwen3_omni_thinker_patches",
    "patch_hf_processor_for_qwen3_omni",
    "patch_hf_tokenizer_for_qwen3_omni",
]


def apply_qwen3_omni_thinker_patches() -> None:
    """Apply Qwen3-Omni thinker patches through the registered adapter."""
    Qwen3OmniThinkerAdapter.apply_model_patches()


apply_qwen3_omni_thinker_patches()
