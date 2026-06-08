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
"""
L2 GPU test: vLLM-Omni rollout log-probs vs FSDP actor recompute on the same trajectory.

Usage:
    # Tiny model, default recipe (CI / smoke)
    pytest tests/workers/rollout/test_qwen_image_train_rollout_logprob_consistency.py -v -s

    # OCR recipe: SDE window [0,5], LoRA r=64, layered_summon + weight sync
    pytest tests/workers/rollout/test_qwen_image_train_rollout_logprob_consistency.py -v -s --qwen-image-recipe=ocr

    # Full Qwen-Image
    pytest tests/workers/rollout/test_qwen_image_train_rollout_logprob_consistency.py -v -s --qwen-image-model=full

    # LoRA weight-sync regression (OCR recipe, hybrid engine)
    pytest tests/workers/rollout/test_qwen_image_train_rollout_lora_weight_sync_consistency.py -v -s
"""

from __future__ import annotations

import pytest
import torch

from .consistency_check import run_hybrid_consistency_check_sync, run_standalone_consistency_check
from .train_rollout_consistency_helpers import QwenImageConsistencyProfile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def test_qwen_image_train_rollout_logprob_consistency(
    ray_runtime,
    tmp_path,
    qwen_image_consistency_profile: QwenImageConsistencyProfile,
) -> None:
    """Rollout via vLLM-Omni, recompute via FSDP actor; default or OCR recipe."""
    profile = qwen_image_consistency_profile
    if profile.recipe == "ocr":
        run_hybrid_consistency_check_sync(profile, tmp_path)
    else:
        run_standalone_consistency_check(profile, tmp_path)
