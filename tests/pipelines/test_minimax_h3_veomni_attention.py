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

"""GPU checks for real FA3 Hub forward/backward on the VeOmni H3 bridge."""

import pytest
import torch

from tests.pipelines.test_minimax_h3_veomni_flow_grpo_on_cpu import (
    _FUSED_TARGETS,
    _build_models,
    _logical_inputs,
    veomni_lora,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.veomni_training_adapter import predict_veomni
from verl_omni.workers.engine.veomni.patch import _apply_attention_backend


@pytest.mark.parametrize("lora", [False, True])
def test_h3_fa3_hub_forward_backward_matches_sdpa(lora):
    torch.manual_seed(41)
    _, actual = _build_models()
    _, reference = _build_models()
    reference.load_state_dict(actual.state_dict())
    _apply_attention_backend(actual, "flash_attention_3_hub")
    _apply_attention_backend(reference, "eager")
    if lora:
        config = veomni_lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=_FUSED_TARGETS)
        actual = veomni_lora.VeOmniLoraModel(actual, config)
        reference = veomni_lora.VeOmniLoraModel(reference, config)
        reference.load_state_dict(actual.state_dict())
    actual = actual.to(device="cuda", dtype=torch.bfloat16)
    reference = reference.to(device="cuda", dtype=torch.bfloat16)
    inputs = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in _logical_inputs().items()
    }
    for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        inputs[key] = inputs[key].to(torch.bfloat16)
    expected = predict_veomni(reference, inputs, use_gradient_checkpointing=True)
    result = predict_veomni(actual, inputs, use_gradient_checkpointing=True)
    torch.testing.assert_close(result, expected, rtol=0.03, atol=0.03)
    sum(x.float().square().mean() for x in result).backward()
    sum(x.float().square().mean() for x in expected).backward()
    grads = []
    for (name, param), (ref_name, ref_param) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        if param.requires_grad:
            assert param.grad is not None and param.grad.isfinite().all(), name
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=0.05, atol=0.01, msg=name)
            grads.append(param.grad)
    assert any(grad.count_nonzero() for grad in grads)
