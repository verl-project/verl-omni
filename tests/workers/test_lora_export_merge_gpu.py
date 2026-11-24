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
"""GPU test of the LoRA merge the true on-policy export shares with the actor forward."""

import copy
from types import SimpleNamespace

import pytest
import torch
from verl.utils.device import get_device_name

import verl_omni.workers.engine.lora_export as lora_export
from tests.utils.gpu_test_topology import enabled_gpu_count

pytestmark = pytest.mark.skipif(not enabled_gpu_count(), reason="requires a GPU")

IN_FEATURES, OUT_FEATURES, RANK = 2048, 4096, 32  # a thinker q_proj; smaller matmuls round alike on CPU and GPU


def test_export_of_offloaded_parameters_merges_as_the_gpu_forward(monkeypatch):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential()
    base.add_module("proj", torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.bfloat16))
    model = get_peft_model(base, LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=["proj"]))
    model.base_model.model.proj.lora_B["default"].weight.data.normal_(std=1e-2)
    monkeypatch.setattr(lora_export, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(lora_export, "convert_weight_keys", lambda params, module: params)

    params, _ = lora_export.export_lora_for_sync(
        model,  # on CPU, as FSDP2 keeps offloaded shards
        model_config=SimpleNamespace(lora={"merge": True}),
        base_sync_done=True,
        is_diffusers=False,
        merge_in_full_weight_dtype=True,
    )
    exported = dict(params)["proj.weight"]

    actor = copy.deepcopy(model).to(get_device_name(), torch.bfloat16)
    lora_export.use_merged_lora_forward(actor)
    identity = torch.eye(IN_FEATURES, dtype=torch.bfloat16, device=get_device_name())
    forward_weight = actor(identity).T  # a linear layer applied to the identity returns its weight

    assert not torch.equal(forward_weight, actor.base_model.model.proj.base_layer.weight)
    assert torch.equal(forward_weight, exported.to(forward_weight.device))
