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
"""Check diffusion engine attention settings against VeOmni's native model configuration."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("veomni")
from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
    QwenImageAttentionKernelModule,
    QwenImageTransformer2DModel,
    QwenImageTransformer2DModelConfig,
)

from tests.workers.veomni_lora_helpers import make_veomni_engine


def _qwen_transformer(backend):
    engine = make_veomni_engine()
    engine.engine_config = SimpleNamespace(attn_implementation=backend)
    return QwenImageTransformer2DModel(
        QwenImageTransformer2DModelConfig(
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            in_channels=4,
            out_channels=4,
            axes_dims_rope=(2, 2, 4),
            ops_implementation=engine._build_ops_config(),
        )
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("eager", "eager"),
        ("sdpa", "veomni_sdpa"),
        ("veomni_sdpa", "veomni_sdpa"),
    ],
)
def test_qwen_image_processor_uses_configured_attention(backend, expected):
    model = _qwen_transformer(backend)
    assert model.transformer_blocks[0].attn.processor.config._attn_implementation == expected


@pytest.mark.parametrize("target_query", [False, True])
def test_qwen_lora_only_exposes_weight_on_the_attention_query(target_query):
    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
    from veomni.lora.layers import LoraLinear

    from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage

    transformer = _qwen_transformer("eager")
    # A same-named projection outside attention must keep its native LoRA type.
    transformer.to_q = torch.nn.Linear(8, 8)
    targets = ["to_k", "to_v", "to_out.0"]
    if target_query:
        targets.append("to_q")
    wrapped = VeOmniLoraModel(transformer, VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=targets))
    attn = transformer.transformer_blocks[0].attn
    query = attn.to_q
    params = {key: id(value) for key, value in wrapped.named_parameters()}
    state_keys = set(wrapped.state_dict())

    QwenImage.configure_veomni_lora(transformer)
    QwenImage.configure_veomni_lora(transformer)

    assert attn.to_q is query
    assert {key: id(value) for key, value in wrapped.named_parameters()} == params
    assert set(wrapped.state_dict()) == state_keys
    for projection in (attn.to_k, attn.to_v, attn.to_out[0]):
        assert type(projection) is LoraLinear
    if target_query:
        assert type(transformer.to_q) is LoraLinear
        assert query.weight is query.base_layer.weight
    else:
        assert type(query) is torch.nn.Linear
    for dtype, expected in [(torch.float32, torch.bfloat16), (torch.float16, torch.float16)]:
        query.to(dtype=dtype)
        kernel = QwenImageAttentionKernelModule("eager", attn)
        assert kernel.config._pre_quantization_dtype == expected


def test_only_qwen_adapters_opt_in_to_query_compatibility():
    from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
    from verl_omni.pipelines.qwen_image_dpo.diffusers_training_adapter import QwenImageDPO
    from verl_omni.pipelines.qwen_image_edit_flow_grpo.diffusers_training_adapter import QwenImageEditPlusFlowGRPO
    from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage
    from verl_omni.pipelines.wan22_dance_grpo.diffusers_training_adapter import Wan22DanceGRPO

    for adapter in (QwenImageDPO, QwenImageEditPlusFlowGRPO):
        assert adapter.configure_veomni_lora is QwenImage.configure_veomni_lora
    for adapter in (MiniMaxH3FlowGRPO, Wan22DanceGRPO):
        assert not hasattr(adapter, "configure_veomni_lora")


@pytest.mark.parametrize("backend", ["flash_attention_2_hub", "flash_attention_3_hub"])
def test_qwen_image_hub_backend_config(monkeypatch, backend):
    # Constructing the processor would resolve a CUDA kernel even without a forward pass.
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    engine = make_veomni_engine()
    engine.engine_config = SimpleNamespace(attn_implementation=backend)
    config = QwenImageTransformer2DModelConfig(ops_implementation=engine._build_ops_config())
    assert config.ops_config.attn_implementation == f"veomni_{backend}"


def test_qwen_image_hub_backend_is_rejected_on_npu(monkeypatch):
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", True)
    engine = make_veomni_engine()
    engine.engine_config = SimpleNamespace(attn_implementation="flash_attention_3_hub")
    with pytest.raises(ValueError, match="not supported on Ascend NPU"):
        engine._build_ops_config()
