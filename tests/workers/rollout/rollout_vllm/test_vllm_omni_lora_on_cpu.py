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


class _QwenStyleEngine:
    def __init__(self):
        self.model = object()


class _MiniCPMStyleEngine:
    def __init__(self):
        self.llm = object()  # MiniCPM-o nests its LLM as .llm


class _ACLGraphWrapped:
    def __init__(self):
        self.runnable = _QwenStyleEngine()


def test_moe_weight_loader_patch_applies_gate():
    from verl_omni.workers.rollout.vllm_rollout.utils import _moe_weight_loader_patch_applies

    assert _moe_weight_loader_patch_applies(_QwenStyleEngine())  # .model resolves -> patch runs
    assert _moe_weight_loader_patch_applies(_ACLGraphWrapped())  # ACLGraph unwrap, same as verl
    assert not _moe_weight_loader_patch_applies(_MiniCPMStyleEngine())  # .llm only -> skip, no ValueError
