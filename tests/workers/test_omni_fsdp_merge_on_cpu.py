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

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import verl_omni.workers.engine.fsdp.omni_impl as omni_impl
import verl_omni.workers.engine.lora_export as lora_export


class _FakeModule:
    def __init__(self):
        self.weight = torch.tensor([1.0])
        # ``peft_config`` is what routes the export onto the merged-LoRA stream.
        self.peft_config = {"default": SimpleNamespace(to_dict=lambda: {"r": 8})}

    def state_dict(self):
        # Match torch state_dict semantics: returned tensors alias live storage.
        return {"weight": self.weight}


def test_merged_weights_materialized_before_actor_restore(monkeypatch):
    module = _FakeModule()

    @contextmanager
    def merged_context(actor, backup_adapters):
        assert actor is module
        assert backup_adapters
        module.weight.fill_(2.0)
        try:
            yield
        finally:
            module.weight.fill_(1.0)

    monkeypatch.setattr(lora_export, "merged_lora_context", merged_context)
    monkeypatch.setattr(lora_export, "normalize_peft_param_name", lambda state: state)
    monkeypatch.setattr(lora_export, "convert_weight_keys", lambda state, model: state)
    monkeypatch.setattr(lora_export, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(omni_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)

    engine = object.__new__(omni_impl.OmniFSDPEngine)
    engine.module = module
    engine.model_config = SimpleNamespace(lora={"merge": True})
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = True
    engine._qat_enabled = False

    params, peft_config = engine.get_per_tensor_param(base_sync_done=True)
    merged_weights = dict(params)

    assert peft_config is None
    assert torch.equal(merged_weights["weight"], torch.tensor([2.0]))
    assert torch.equal(module.weight, torch.tensor([1.0]))
