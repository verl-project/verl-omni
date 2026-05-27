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
"""GPU integration tests for LoRAAdapterMixin on tiny Qwen-Image."""

import os
from types import SimpleNamespace

import pytest
import torch
from diffusers import QwenImageTransformer2DModel

from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

_DEFAULT_MODEL_PATH = os.path.expanduser("~/models/tiny-random/Qwen-Image")


def _require_model_path() -> str:
    if not os.path.isdir(_DEFAULT_MODEL_PATH):
        pytest.skip(
            f"Tiny Qwen-Image model not found at {_DEFAULT_MODEL_PATH!r}. "
            "Provide the model or adjust _DEFAULT_MODEL_PATH."
        )
    return _DEFAULT_MODEL_PATH


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for LoRA adapter GPU tests.")


class _QwenLoRAEngine(LoRAAdapterMixin):
    def __init__(self, model_config, module):
        self.model_config = model_config
        self.module = module


def _make_qwen_engine(model_path: str):
    model_config = SimpleNamespace(
        lora_adapter_path=None,
        policy_state_adapters=("default", "old"),
        lora_rank=8,
        lora_alpha=16,
        lora_init_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        target_parameters=None,
        exclude_modules=None,
        use_shm=False,
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    engine = _QwenLoRAEngine(model_config, transformer)
    module = engine._build_lora_module(transformer)
    return engine, module


def _trainable_lora_tensors(module, adapter_name: str) -> list[torch.Tensor]:
    module.set_adapter(adapter_name)
    return [p.detach().clone() for p in module.parameters() if p.requires_grad]


def test_copy_adapter_syncs_default_to_old():
    _require_cuda()
    model_path = _require_model_path()
    engine, module = _make_qwen_engine(model_path)

    module.set_adapter("default")
    for param in module.parameters():
        if param.requires_grad:
            param.data.fill_(1.0)

    engine.copy_adapter(source="default", target="old")
    default_params = _trainable_lora_tensors(module, "default")
    old_params = _trainable_lora_tensors(module, "old")
    assert len(default_params) == len(old_params) > 0
    for default_param, old_param in zip(default_params, old_params, strict=True):
        assert torch.allclose(default_param, old_param)


def _fill_trainable_params(module, adapter_name: str, base: float, step: float) -> int:
    module.set_adapter(adapter_name)
    count = 0
    for param in module.parameters():
        if param.requires_grad:
            param.data.fill_(base + count * step)
            count += 1
    return count


def test_ema_update_adapter_blends_decay_nine():
    """EMA blend on real Qwen LoRA: target = 0.9 * old + 0.1 * default."""
    _require_cuda()
    model_path = _require_model_path()
    engine, module = _make_qwen_engine(model_path)

    old_base, old_step = 3.5, 0.25
    default_base, default_step = 7.25, -0.1
    decay = 0.9

    assert _fill_trainable_params(module, "old", old_base, old_step) > 0
    _fill_trainable_params(module, "default", default_base, default_step)

    engine.ema_update_adapter(source="default", target="old", decay=decay)

    module.set_adapter("old")
    for idx, param in enumerate(p for p in module.parameters() if p.requires_grad):
        old_val = old_base + idx * old_step
        default_val = default_base + idx * default_step
        expected = old_val * decay + default_val * (1.0 - decay)
        assert torch.allclose(
            param.float(),
            torch.full_like(param.float(), expected),
            rtol=1e-2,
            atol=1e-2,
        )
