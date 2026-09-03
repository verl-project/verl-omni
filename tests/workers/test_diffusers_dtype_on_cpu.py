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
"""CPU tests for diffusers model dtype finalization."""

import torch

from verl_omni.pipelines.wan22_dance_grpo.diffusers_training_adapter import Wan22DanceGRPO
from verl_omni.workers.engine.fsdp.diffusers_impl import _cast_loaded_diffusers_module, _fsdp_param_dtype


class _MixedPrecisionModel(torch.nn.Module):
    _keep_in_fp32_modules = ["sensitive"]

    def __init__(self) -> None:
        super().__init__()
        self.regular = torch.nn.Linear(4, 4).to(torch.bfloat16)
        self.sensitive = torch.nn.Linear(4, 4).to(torch.float32)


class _MixedDtypeModelWithoutDeclaredFp32Islands(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(4, 4).to(torch.bfloat16)
        self.trainable = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))


def test_diffusers_declared_fp32_islands_are_not_truncated():
    model = _MixedPrecisionModel()
    sensitive_before = model.sensitive.weight.detach().clone()

    _cast_loaded_diffusers_module(model, torch.bfloat16)

    assert model.regular.weight.dtype == torch.bfloat16
    assert model.sensitive.weight.dtype == torch.float32
    torch.testing.assert_close(model.sensitive.weight, sensitive_before, rtol=0, atol=0)
    assert _fsdp_param_dtype(model, torch.bfloat16) is None


def test_adapter_can_request_uniform_dtype_for_fsdp_flattening():
    model = _MixedPrecisionModel()

    assert Wan22DanceGRPO.preserve_fp32_modules() is False
    _cast_loaded_diffusers_module(model, torch.bfloat16, preserve_fp32_modules=False)

    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert _fsdp_param_dtype(model, torch.bfloat16, preserve_fp32_modules=False) == torch.bfloat16


def test_mixed_dtype_model_without_declared_fp32_islands_uses_engine_dtype():
    model = _MixedDtypeModelWithoutDeclaredFp32Islands()

    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16, torch.float32}
    assert _fsdp_param_dtype(model, torch.bfloat16) == torch.bfloat16


def test_ordinary_diffusers_model_is_cast_to_engine_dtype():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))

    _cast_loaded_diffusers_module(model, torch.bfloat16)

    assert model[0].weight.dtype == torch.bfloat16
    assert _fsdp_param_dtype(model, torch.bfloat16) == torch.bfloat16
