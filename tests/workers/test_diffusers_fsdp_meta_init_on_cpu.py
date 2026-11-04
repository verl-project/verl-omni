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
"""CPU tests for meta-rank config-only init in the diffusers FSDP engine."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from accelerate import init_empty_weights

import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


class _FakeDiffusersModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace()
        self.attn_backend = None

    def set_attention_backend(self, backend):
        self.attn_backend = backend


def _make_engine(**overrides):
    engine = object.__new__(PPODiffusersFSDPEngine)
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "config_path", None)
    object.__setattr__(model_config, "local_path", "/fake/ckpt")
    object.__setattr__(model_config, "trust_remote_code", True)
    object.__setattr__(model_config, "transformer_subfolder", "transformer")
    object.__setattr__(model_config, "attn_backend", "flash_attn")
    object.__setattr__(model_config, "enable_gradient_checkpointing", False)
    for key, value in overrides.items():
        object.__setattr__(model_config, key, value)
    engine.model_config = model_config
    engine.engine_config = MagicMock(model_dtype=None, forward_only=False)
    engine.device_mesh = None
    return engine


def _build_default_path(engine, *, init_context, loaded_module):
    adapter_cls = MagicMock()
    adapter_cls.preserve_fp32_modules.return_value = False

    with (
        patch.object(engine, "_build_module_from_registry", return_value=None),
        patch.object(diffusers_impl, "get_init_weight_context_manager", return_value=init_context),
        patch.object(diffusers_impl.DiffusionModelBase, "get_class", return_value=adapter_cls),
        patch("diffusers.AutoModel.from_config", return_value=loaded_module) as mock_from_config,
        patch("diffusers.AutoModel.from_pretrained", return_value=loaded_module) as mock_from_pretrained,
        patch("verl.utils.torch_dtypes.PrecisionType") as precision_type,
    ):
        precision_type.to_dtype.side_effect = lambda value: value
        result = engine._build_module()
    return result, mock_from_config, mock_from_pretrained


def test_build_module_on_meta_rank_builds_from_config_without_reading_the_checkpoint():
    engine = _make_engine()
    loaded = _FakeDiffusersModule()

    result, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=init_empty_weights, loaded_module=loaded
    )

    mock_from_pretrained.assert_not_called()
    mock_from_config.assert_called_once_with(
        engine.model_config.local_path,
        trust_remote_code=True,
        subfolder="transformer",
    )
    assert result is loaded
    assert loaded.attn_backend == "flash_attn"
    assert loaded.proj.weight.dtype == torch.float32
    assert callable(loaded.can_generate)
    assert callable(loaded.config.save_pretrained)


def test_build_module_on_rank0_still_loads_from_pretrained():
    engine = _make_engine()
    loaded = _FakeDiffusersModule()

    result, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=nullcontext, loaded_module=loaded
    )

    mock_from_config.assert_not_called()
    mock_from_pretrained.assert_called_once_with(
        engine.model_config.local_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        subfolder="transformer",
    )
    assert result is loaded
    assert loaded.attn_backend == "flash_attn"
    assert loaded.proj.weight.dtype == torch.float32


def test_build_module_on_meta_rank_uses_config_path_when_set():
    engine = _make_engine(config_path="/fake/config", transformer_subfolder="transformer")
    loaded = _FakeDiffusersModule()

    _, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=init_empty_weights, loaded_module=loaded
    )

    mock_from_pretrained.assert_not_called()
    mock_from_config.assert_called_once_with(
        "/fake/config",
        trust_remote_code=True,
        subfolder="",
    )
