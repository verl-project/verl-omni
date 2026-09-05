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
"""CPU regressions for role-aware LoRA switching and export."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


class PeftConfig:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name}


class PeftModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.peft_config = {
            "default": PeftConfig("default"),
            "student": PeftConfig("student"),
            "student_ema": PeftConfig("student_ema"),
        }
        self.active_adapter = "student"
        self.adapters_enabled = True

    @property
    def active_adapters(self):
        return [self.active_adapter]

    def set_adapter(self, name):
        self.active_adapter = name[0] if isinstance(name, list) else name

    def disable_adapters(self):
        self.adapters_enabled = False

    def enable_adapters(self):
        self.adapters_enabled = True


class MixinHarness(LoRAAdapterMixin):
    def __init__(self):
        self.module = PeftModule()
        self._is_offload_param = False


class TestAdapterContext:
    def test_nested_named_adapter_context_restores_exact_selection(self):
        harness = MixinHarness()
        with harness.use_adapter("student_ema"):
            assert harness.module.active_adapter == "student_ema"
            with harness.use_adapter("default"):
                assert harness.module.active_adapter == "default"
            assert harness.module.active_adapter == "student_ema"
        assert harness.module.active_adapter == "student"

    def test_exception_restores_previous_adapter(self):
        harness = MixinHarness()
        with pytest.raises(RuntimeError, match="boom"):
            with harness.use_adapter("student_ema"):
                raise RuntimeError("boom")
        assert harness.module.active_adapter == "student"

    def test_reference_context_reenables_prior_named_adapter(self):
        harness = MixinHarness()
        with harness.use_adapter("reference"):
            assert not harness.module.adapters_enabled
            assert harness.module.active_adapter == "student"
        assert harness.module.adapters_enabled
        assert harness.module.active_adapter == "student"


class TestAdapterAwareExport:
    def test_non_default_adapter_exports_its_own_peft_config(self, monkeypatch):
        harness = MixinHarness()
        harness._uses_fsdp2_cpu_offload_policy = True
        harness.model_config = SimpleNamespace(fsdp_layer_prefixes=["transformer_blocks."])
        collect_lora_params = Mock(return_value={"adapter.weight": torch.tensor([2.0])})

        monkeypatch.setattr(
            "verl_omni.workers.engine.fsdp.diffusers_impl.collect_lora_params",
            collect_lora_params,
        )
        monkeypatch.setattr(
            "verl_omni.workers.engine.fsdp.diffusers_impl.convert_weight_keys",
            lambda params, module: params,
        )
        params, peft_config = DiffusersFSDPEngine.get_per_tensor_param(
            harness,
            base_sync_done=True,
            adapter_name="student_ema",
        )
        assert dict(params) == {"transformer.adapter.weight": torch.tensor([2.0])}
        assert peft_config == {"name": "student_ema"}
        collect_lora_params.assert_called_once()
        assert collect_lora_params.call_args.kwargs["adapter_name"] == "student_ema"
        assert harness.module.active_adapter == "student"

    def test_unknown_adapter_fails_before_export(self):
        harness = MixinHarness()
        harness._uses_fsdp2_cpu_offload_policy = True
        harness.model_config = SimpleNamespace(fsdp_layer_prefixes=[])
        with pytest.raises(ValueError, match="unknown LoRA adapter"):
            DiffusersFSDPEngine.get_per_tensor_param(
                harness,
                base_sync_done=True,
                adapter_name="missing",
            )
