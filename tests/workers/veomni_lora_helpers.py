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
"""CPU test helpers shared by the VeOmni diffusion LoRA tests."""

from unittest.mock import MagicMock

import torch

import verl_omni.workers.engine.veomni.diffusion_impl as veomni_impl
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.veomni.diffusion_impl import VeOmniDiffusionEngine


def make_veomni_engine(
    module=None,
    *,
    lora_rank: int = 0,
    lora_alpha: int = 64,
    target_modules=None,
    lora_adapter_path=None,
    lora: dict | None = None,
) -> VeOmniDiffusionEngine:
    """Build an engine without running ``__init__`` (which needs torch.distributed)."""
    engine = object.__new__(VeOmniDiffusionEngine)
    engine.module = module
    engine._is_offload_param = False
    engine._is_lora = lora_rank > 0 or lora_adapter_path is not None
    # DiffusionModelConfig.__post_init__ does I/O; set the fields under test directly.
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "lora_rank", lora_rank)
    object.__setattr__(model_config, "lora_alpha", lora_alpha)
    object.__setattr__(model_config, "target_modules", target_modules)
    object.__setattr__(model_config, "exclude_modules", None)
    object.__setattr__(model_config, "lora_adapter_path", lora_adapter_path)
    object.__setattr__(model_config, "lora", lora if lora is not None else {})
    engine.model_config = model_config
    engine.engine_config = MagicMock(model_dtype="bf16")
    return engine


def export_veomni_params(engine, monkeypatch, **kwargs):
    """Run ``get_per_tensor_param`` on CPU and return the exported tensors and PEFT config."""
    monkeypatch.setattr(veomni_impl, "load_model_to_gpu", MagicMock())
    monkeypatch.setattr(veomni_impl, "offload_model_to_cpu", MagicMock())
    monkeypatch.setattr(veomni_impl, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(veomni_impl.PrecisionType, "to_dtype", staticmethod(lambda _: torch.float32))
    generator, peft_config = engine.get_per_tensor_param(**kwargs)
    return dict(generator), peft_config
