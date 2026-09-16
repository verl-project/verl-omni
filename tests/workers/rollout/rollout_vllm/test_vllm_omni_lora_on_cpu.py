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
"""CPU checks for vLLM-Omni LoRA integration."""

from types import SimpleNamespace

import torch

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension


def test_diffusion_lora_stacks_follow_the_worker_device():
    layer = SimpleNamespace(
        lora_a_stacked=(torch.ones(1),),
        lora_b_stacked=(torch.ones(1),),
    )
    worker = SimpleNamespace(
        device=torch.device("meta"),
        lora_manager=SimpleNamespace(_lora_modules={"transformer.block": layer}),
    )

    vLLMOmniColocateWorkerExtension._move_diffusion_lora_stacks_to_device(worker)

    assert layer.lora_a_stacked[0].device.type == "meta"
    assert layer.lora_b_stacked[0].device.type == "meta"


def test_moe_weight_loader_patch_value_error_is_swallowed(monkeypatch):
    # verl raises for engines whose inner model does not resolve (MiniCPM-o
    # nests its LLM as .llm); the shared call site swallows exactly that.
    import verl_omni.workers.rollout.vllm_rollout.utils as rollout_utils

    def raising_patch(model):
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    monkeypatch.setattr("verl.utils.vllm.patch.patch_vllm_moe_model_weight_loader", raising_patch)
    rollout_utils._apply_moe_weight_loader_patch(object())  # must not raise


def test_moe_weight_loader_patch_other_errors_propagate(monkeypatch):
    import pytest

    import verl_omni.workers.rollout.vllm_rollout.utils as rollout_utils

    def raising_patch(model):
        raise ValueError("some unrelated patch failure")

    monkeypatch.setattr("verl.utils.vllm.patch.patch_vllm_moe_model_weight_loader", raising_patch)
    with pytest.raises(ValueError, match="unrelated"):
        rollout_utils._apply_moe_weight_loader_patch(object())
