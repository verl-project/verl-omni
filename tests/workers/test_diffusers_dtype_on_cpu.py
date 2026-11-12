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


def test_from_config_layout_matches_from_pretrained_mixed_dtypes(tmp_path):
    """A from_config-built module must land on from_pretrained's dtype layout.

    from_pretrained keeps ``_keep_in_fp32_modules`` entries in fp32 and casts the
    rest to torch_dtype. Meta ranks build via from_config (all fp32); if the cast
    skipped islands wholesale their layout stays uniform fp32 while rank 0 is
    mixed, FSDP2 shard sizes diverge across ranks, and weight-sync collectives
    deadlock (MiniMax-H3 T2VA gpu-smoke).
    """
    from diffusers import MiniMaxH3Transformer3DModel

    from verl_omni.workers.engine.fsdp.diffusers_impl import _cast_loaded_diffusers_module

    config = {
        "attention_head_dim": 32,
        "audio_in_channels": 32,
        "ffn_dim": 128,
        "freq_dim": 64,
        "hidden_size": 64,
        "in_channels": 24,
        "num_attention_heads": 2,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "patch_size": [1, 2, 2],
        "rope_freq_dim": 4,
        "rope_theta": 10000.0,
        "text_dim": 32,
        "time_embed_dim": 32,
        "time_embed_hidden_dim": 64,
    }
    MiniMaxH3Transformer3DModel.from_config(config).save_pretrained(tmp_path)

    rank0 = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
    _cast_loaded_diffusers_module(rank0, torch.bfloat16)

    from accelerate import init_empty_weights as _init_empty

    with _init_empty():
        meta = MiniMaxH3Transformer3DModel.from_config(config)
    _cast_loaded_diffusers_module(meta, torch.bfloat16)

    def layout(module):
        return {name: tensor.dtype for name, tensor in [*module.named_parameters(), *module.named_buffers()]}

    assert layout(meta) == layout(rank0)
    assert layout(rank0)["proj_in.weight"] == torch.float32
    assert layout(rank0)["transformer_blocks.0.attn.to_q.weight"] == torch.bfloat16


def test_island_matching_uses_whole_name_segments_like_from_pretrained():
    """diffusers matches _keep_in_fp32_modules on dot segments, not substrings."""

    class _IslandModel(torch.nn.Module):
        _keep_in_fp32_modules = ["modulation"]

        def __init__(self) -> None:
            super().__init__()
            self.modulation = torch.nn.Linear(4, 4)
            self.va_modulation = torch.nn.Linear(4, 4)

    model = _IslandModel()
    _cast_loaded_diffusers_module(model, torch.bfloat16)

    assert model.modulation.weight.dtype == torch.float32
    assert model.va_modulation.weight.dtype == torch.bfloat16


def test_bare_string_island_declaration_is_normalized():
    class _BareStringIslandModel(torch.nn.Module):
        _keep_in_fp32_modules = "sensitive"

        def __init__(self) -> None:
            super().__init__()
            self.sensitive = torch.nn.Linear(4, 4)
            self.regular = torch.nn.Linear(4, 4)

    model = _BareStringIslandModel()
    _cast_loaded_diffusers_module(model, torch.bfloat16)

    assert model.sensitive.weight.dtype == torch.float32
    assert model.regular.weight.dtype == torch.bfloat16
