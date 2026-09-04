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

from verl_omni.pipelines.minicpm.thinker_training_adapter import (
    MiniCPMThinkerAdapter,
    patch_minicpm_whisper_encoder_layers,
    split_minicpm_forward_kwargs,
)
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


def test_split_minicpm_forward_kwargs_drops_llm_bound_inputs_embeds():
    _, llm_kwargs = split_minicpm_forward_kwargs(
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "position_ids": torch.arange(2).unsqueeze(0),
            "inputs_embeds": torch.zeros(1, 2, 4),
            "attention_mask": torch.ones(1, 2),
        }
    )
    assert "inputs_embeds" not in llm_kwargs
    assert "input_ids" not in llm_kwargs
    assert "position_ids" not in llm_kwargs
    assert "attention_mask" in llm_kwargs


class _MiniCPMOLlmCall(_MiniCPMOStyle):
    """Mirrors MiniCPMO.forward binding inputs_embeds before ``**kwargs``."""

    def forward(self, data, **kwargs):
        self.last_data = data
        self.last_llm_kwargs = kwargs
        embeds = torch.ones(*data["input_ids"].shape, 1)
        return self.llm(
            input_ids=None,
            position_ids=data["position_ids"],
            inputs_embeds=embeds,
            **kwargs,
        )


def test_wrapped_forward_does_not_duplicate_inputs_embeds_into_llm():
    module = _MiniCPMOLlmCall()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    configured(
        input_ids=torch.ones(2, 3, dtype=torch.long),
        position_ids=torch.arange(3).repeat(2, 1),
        inputs_embeds=torch.zeros(2, 3, 4),
        attention_mask=torch.ones(2, 3),
        use_cache=False,
    )
    assert "inputs_embeds" not in configured.last_llm_kwargs
    assert configured.last_llm_kwargs["attention_mask"].shape == (2, 3)
    assert configured.last_llm_kwargs["use_cache"] is False


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


def test_fsdp_name_is_ignored_matches_peft_prefixed_apm():
    from verl_omni.utils.fsdp_utils import fsdp_name_is_ignored

    ignored = ["apm"]
    assert fsdp_name_is_ignored("apm", ignored)
    assert fsdp_name_is_ignored("apm.embed_positions.weight", ignored)
    assert fsdp_name_is_ignored("base_model.model.apm.conv1.weight", ignored)
    assert not fsdp_name_is_ignored("llm.layers.0.self_attn.q_proj.weight", ignored)


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
    assert configured._no_split_modules == ["Qwen3DecoderLayer", "MiniCPMODecoderLayer"]
    assert MiniCPMThinkerAdapter.get_fsdp_ignored_module_names(_model_config()) == ["apm"]


class _MiniCPMOWithEncoders(_MiniCPMOStyle):
    def __init__(self):
        super().__init__()
        self.vpm = nn.Linear(4, 4)
        self.apm = nn.Linear(4, 4)
        self.vision_calls = 0

    def get_vision_embedding(self, data):
        self.vision_calls += 1
        hidden = torch.ones(1, 2, 4, requires_grad=True)
        return [hidden]


def test_configure_model_freezes_vpm_and_apm():
    module = _MiniCPMOWithEncoders()
    assert all(param.requires_grad for param in module.vpm.parameters())
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    assert configured.vpm.training is False
    assert configured.apm.training is False
    assert all(not param.requires_grad for param in configured.vpm.parameters())
    assert all(not param.requires_grad for param in configured.apm.parameters())
    assert all(param.requires_grad for param in configured.llm.embed.parameters())


def test_patched_get_vision_embedding_skips_dummy_encoder_when_no_images():
    module = _MiniCPMOWithEncoders()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    states = configured.get_vision_embedding({"pixel_values": [[], []], "input_ids": torch.ones(2, 3)})
    assert states == [[], []]
    assert configured.vision_calls == 0


def test_patched_get_vision_embedding_runs_encoder_without_grad():
    module = _MiniCPMOWithEncoders()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    states = configured.get_vision_embedding({"pixel_values": [[torch.zeros(3, 2, 2)]]})
    assert configured.vision_calls == 1
    assert states[0].requires_grad is False


def test_cloned_vllm_embedding_scatter_supports_backward():
    module = _MiniCPMOWithEncoders()
    module.llm.model = nn.Module()
    module.llm.model.embed_tokens = module.llm.embed
    module.llm.config = SimpleNamespace()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    embeddings, _ = configured.get_vllm_embedding(
        {
            "input_ids": input_ids,
            "pixel_values": [[torch.zeros(3, 2, 2)]],
            "image_bound": [torch.tensor([[0, 2]])],
        }
    )
    embeddings.sum().backward()
    assert configured.llm.embed.weight.grad is not None


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


class _TwoTupleWhisperAttn(nn.Module):
    """Mirrors transformers WhisperAttention: returns (hidden_states, attn_weights)."""

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return hidden_states, None


class _MiniCPMWhisperEncoderLayerStub(nn.Module):
    """Mirrors MiniCPMWhisperEncoderLayer's 3-way unpack of self_attn."""

    def __init__(self):
        super().__init__()
        self.self_attn = _TwoTupleWhisperAttn()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        past_key_values=None,
        use_cache=False,
    ):
        del use_cache
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        del attn_weights
        return hidden_states, past_key_values


class _MiniCPMOWithAPM(_MiniCPMOStyle):
    def __init__(self):
        super().__init__()
        self.apm = nn.Module()
        self.apm.layers = nn.ModuleList([_MiniCPMWhisperEncoderLayerStub()])


def test_unpatched_whisper_layer_cannot_unpack_two_tuple_attn():
    layer = _MiniCPMWhisperEncoderLayerStub()
    hidden = torch.ones(1, 2, 4)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        layer(hidden)


def test_configure_model_pads_whisper_self_attn_to_three_tuple():
    module = _MiniCPMOWithAPM()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    hidden = torch.ones(1, 2, 4)

    out, past = configured.apm.layers[0](hidden, past_key_values="cache")

    assert torch.equal(out, hidden)
    assert past == "cache"


def test_patch_minicpm_whisper_encoder_layers_is_idempotent():
    module = _MiniCPMOWithAPM()
    patch_minicpm_whisper_encoder_layers(module)
    first_forward = module.apm.layers[0].self_attn.forward
    patch_minicpm_whisper_encoder_layers(module)
    assert module.apm.layers[0].self_attn.forward is first_forward
    hidden = torch.ones(1, 2, 4)
    out, _ = module.apm.layers[0](hidden)
    assert torch.equal(out, hidden)
