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
"""CPU tests for the MiniCPM thinker training adapter."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter, split_minicpm_forward_kwargs
from verl_omni.pipelines.model_base import OmniModelBase


def _prepare_inputs_for_generation(self, input_ids, **kwargs):
    return {"input_ids": input_ids, **kwargs}


class _LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def forward(self, input_ids=None, **kwargs):
        del kwargs
        return input_ids

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, embeddings):
        self.embed = embeddings


class _MiniCPMOStyle(nn.Module):
    """Mirrors remote MiniCPMO.forward(self, data, **kwargs)."""

    def __init__(self):
        super().__init__()
        self.llm = _LLM()
        self.llm.prepare_inputs_for_generation = MethodType(_prepare_inputs_for_generation, self.llm)
        self.audio_decoder = nn.Linear(4, 4)
        self.code2wav = nn.Linear(4, 4)
        self.last_data = None
        self.last_llm_kwargs = None

    def forward(self, data, **kwargs):
        self.last_data = data
        self.last_llm_kwargs = kwargs
        return self.llm(input_ids=data["input_ids"], **kwargs)


def test_configure_model_packs_hf_kwargs_into_minicpmo_data():
    module = _MiniCPMOStyle()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    input_ids = torch.ones(2, 3, dtype=torch.long)
    position_ids = torch.arange(3).repeat(2, 1)
    attention_mask = torch.ones(2, 3)
    pixel_values = [[torch.zeros(3, 2, 2)], []]

    output = configured(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        use_cache=False,
    )

    assert configured.last_data["input_ids"] is input_ids
    assert configured.last_data["position_ids"] is position_ids
    assert configured.last_data["pixel_values"] is pixel_values
    assert configured.last_llm_kwargs["attention_mask"] is attention_mask
    assert configured.last_llm_kwargs["use_cache"] is False
    assert torch.equal(output, input_ids)


def test_prepare_model_inputs_returns_data_dict_for_engine_unpack():
    input_ids = torch.ones(1, 4, dtype=torch.long)
    position_ids = torch.arange(4).unsqueeze(0)
    packed = MiniCPMThinkerAdapter.prepare_model_inputs(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": torch.ones(1, 4),
            "pixel_values": [[]],
        },
        micro_batch=None,
        model_config=_model_config(),
    )

    assert set(packed) == {"data", "attention_mask"}
    assert set(packed["data"]) >= {
        "input_ids",
        "position_ids",
        "pixel_values",
        "tgt_sizes",
        "image_bound",
        "audio_bounds",
    }
    assert packed["data"]["input_ids"] is input_ids
    assert packed["data"]["image_bound"] == [[]]
    assert packed["data"]["audio_bounds"] == [[]]


def test_split_minicpm_forward_kwargs_collapses_empty_audio_placeholders():
    data, _ = split_minicpm_forward_kwargs(
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
            "audio_features": [[], []],
            "audio_feature_lens": [[], []],
        }
    )
    assert data["audio_features"] == []
    assert data["audio_feature_lens"] == []


def test_split_minicpm_forward_kwargs_rejects_packed_rmpad_batch():
    with pytest.raises(ValueError, match="use_remove_padding=false"):
        split_minicpm_forward_kwargs(
            {
                "input_ids": torch.ones(1, 8, dtype=torch.long),
                "position_ids": torch.arange(8).unsqueeze(0),
                "image_bound": [torch.tensor([[1, 3]]), torch.tensor([[0, 2]])],
            }
        )


def _model_config(**override_config):
    return SimpleNamespace(
        local_path="/fake/minicpm",
        hf_config=SimpleNamespace(),
        trust_remote_code=True,
        override_config=override_config,
    )


def test_minicpm_adapter_registered_for_minicpmo_architecture():
    assert OmniModelBase.get_class_by_name("MiniCPMO", "thinker") is MiniCPMThinkerAdapter


def test_configure_model_strips_generation_modules_and_keeps_outer_forward():
    module = _MiniCPMOStyle()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())

    assert configured is module
    assert not hasattr(configured, "audio_decoder")
    assert not hasattr(configured, "code2wav")
    assert configured.forward.__self__ is configured
    assert configured.forward.__func__ is not configured.llm.forward.__func__
    assert configured.get_input_embeddings.__self__ is configured.llm
    assert configured.set_input_embeddings.__self__ is configured.llm
    assert configured.prepare_inputs_for_generation.__self__ is configured.llm
    assert configured._no_split_modules == ["Qwen3DecoderLayer", "MiniCPMODecoderLayer", "MiniCPMWhisperEncoder"]


def test_build_module_uses_transformers_auto_model(monkeypatch):
    from transformers import AutoModel

    loaded = MagicMock(spec=nn.Module)
    calls = []
    patch_calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return loaded

    def fake_patch(*args, **kwargs):
        patch_calls.append((args, kwargs))

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        "verl_omni.models.transformers.remote_code_compat.patch_remote_auto_model_init",
        fake_patch,
    )
    config = _model_config()

    module = MiniCPMThinkerAdapter.build_module(config, torch.bfloat16)

    assert module is loaded
    assert patch_calls == [
        (
            ("/fake/minicpm",),
            {"trust_remote_code": True, "config": config.hf_config},
        )
    ]
    assert calls[0][0] == ("/fake/minicpm",)
    assert calls[0][1]["torch_dtype"] is torch.bfloat16
    assert calls[0][1]["trust_remote_code"] is True
    assert calls[0][1]["config"] is config.hf_config
    assert "init_tts" not in calls[0][1]
