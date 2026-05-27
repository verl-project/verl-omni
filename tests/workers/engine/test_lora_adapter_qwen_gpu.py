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
    engine.module = module
    return engine, module


def test_build_lora_module_creates_default_and_old():
    _require_cuda()
    model_path = _require_model_path()
    _, module = _make_qwen_engine(model_path)

    assert "default" in module.peft_config
    assert "old" in module.peft_config
    trainable = [p for p in module.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_use_adapter_switches_active_adapter():
    _require_cuda()
    model_path = _require_model_path()
    engine, module = _make_qwen_engine(model_path)

    assert module.active_adapter == "default"
    with engine.use_adapter("old"):
        assert module.active_adapter == "old"
    assert module.active_adapter == "default"
