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

import torch
import torch.nn as nn

from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter
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


class _MiniCPMModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = _LLM()
        self.llm.prepare_inputs_for_generation = MethodType(_prepare_inputs_for_generation, self.llm)
        self.audio_decoder = nn.Linear(4, 4)
        self.code2wav = nn.Linear(4, 4)


def _model_config(**override_config):
    return SimpleNamespace(
        local_path="/fake/minicpm",
        hf_config=SimpleNamespace(),
        trust_remote_code=True,
        override_config=override_config,
    )


def test_minicpm_adapter_registered_for_minicpmo_architecture():
    assert OmniModelBase.get_class_by_name("MiniCPMO", "thinker") is MiniCPMThinkerAdapter


def test_configure_model_strips_generation_modules_and_redirects_to_llm():
    module = _MiniCPMModule()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())

    assert configured is module
    assert not hasattr(configured, "audio_decoder")
    assert not hasattr(configured, "code2wav")
    assert configured.forward.__self__ is configured.llm
    assert configured.forward.__func__ is configured.llm.forward.__func__
    assert configured.get_input_embeddings.__self__ is configured.llm
    assert configured.set_input_embeddings.__self__ is configured.llm
    assert configured.prepare_inputs_for_generation.__self__ is configured.llm
    assert configured._no_split_modules == ["Qwen3DecoderLayer", "MiniCPMODecoderLayer"]


def test_build_module_uses_transformers_auto_model(monkeypatch):
    from transformers import AutoModel

    loaded = MagicMock(spec=nn.Module)
    calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)
    config = _model_config()

    module = MiniCPMThinkerAdapter.build_module(config, torch.bfloat16)

    assert module is loaded
    assert calls[0][0] == ("/fake/minicpm",)
    assert calls[0][1]["torch_dtype"] is torch.bfloat16
    assert calls[0][1]["trust_remote_code"] is True
    assert calls[0][1]["config"] is config.hf_config
    assert "init_tts" not in calls[0][1]
