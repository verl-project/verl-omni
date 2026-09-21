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
    MiniCPMO,
    MiniCPMThinkerAdapter,
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


class _WhisperAttnStub(nn.Module):
    """WhisperAttention before the 3-tuple patch: returns (hidden, attn_weights)."""

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return hidden_states, None


class _WhisperLayerStub(nn.Module):
    """MiniCPMWhisperEncoderLayer: unpacks three values from its self-attn."""

    def __init__(self):
        super().__init__()
        self.self_attn = _WhisperAttnStub()

    def forward(self, hidden_states, **kwargs):
        hidden_states, _, _ = self.self_attn(hidden_states, **kwargs)
        return hidden_states


class _MiniCPMOStyle(nn.Module):
    """The remote MiniCPMO's shape: towers, the inner LLM, and its four embedders."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(version="4.5")
        self.llm = _LLM()
        self.llm.model = nn.Module()
        self.llm.model.embed_tokens = self.llm.embed
        self.llm.config = SimpleNamespace()
        self.llm.prepare_inputs_for_generation = MethodType(_prepare_inputs_for_generation, self.llm)
        # apm carries both the Whisper conv stack and its encoder layers.
        self.apm = nn.Module()
        self.apm.conv1 = nn.Linear(4, 4)
        self.apm.layers = nn.ModuleList([_WhisperLayerStub()])
        self.vpm = nn.Linear(4, 4)
        self.resampler = nn.Linear(4, 4)
        self.tts = nn.Linear(4, 4)
        self.last_data = None
        self.last_llm_kwargs = None

    def forward(self, data, **kwargs):
        self.last_data = data
        self.last_llm_kwargs = kwargs
        return self.llm(input_ids=data["input_ids"], **kwargs)

    def get_vision_embedding(self, data):
        del data
        return []

    def get_vllm_embedding(self, data):
        return self.llm.model.embed_tokens(data["input_ids"]), []

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True, **kwargs):
        del data, chunk_length, dummy, kwargs
        return []

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False, **kwargs):
        del data, chunk_length, stream_input, kwargs
        return input_embeddings


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
    packed_pixels = configured.last_data["pixel_values"]
    assert len(packed_pixels) == 2
    assert len(packed_pixels[0]) == 1
    torch.testing.assert_close(packed_pixels[0][0], pixel_values[0][0])
    assert packed_pixels[1] == []
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


def test_split_minicpm_forward_kwargs_keeps_object_array_batches_per_sample():
    # The flagged padded-batch shape: DataProto collates ragged media into an
    # object-dtype ndarray. Without the unwrap, both samples' slices were read as
    # one sample's and every image landed on batch row 0.
    import numpy as np

    pixel = np.empty(2, dtype=object)
    pixel[0] = [torch.zeros(3, 2, 2), torch.zeros(3, 2, 2)]
    pixel[1] = [torch.zeros(3, 2, 2)]
    sizes = np.empty(2, dtype=object)
    sizes[0] = np.array([[2, 2], [2, 2]], dtype=np.int64)
    sizes[1] = np.array([[2, 2]], dtype=np.int64)

    data, _ = split_minicpm_forward_kwargs(
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
            "pixel_values": pixel,
            "tgt_sizes": sizes,
        }
    )
    assert [len(sample) for sample in data["pixel_values"]] == [2, 1]
    assert all(tuple(slice_.shape) == (3, 2, 2) for sample in data["pixel_values"] for slice_ in sample)
    assert [sample.tolist() for sample in data["tgt_sizes"]] == [[[2, 2], [2, 2]], [[2, 2]]]


def test_split_minicpm_forward_kwargs_rejects_unprepared_packed_batch():
    with pytest.raises(ValueError, match="without going through MiniCPMThinkerAdapter.prepare_model_inputs"):
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
    assert MiniCPMThinkerAdapter.auto_model_class is MiniCPMO


def test_configure_model_strips_generation_modules_and_keeps_outer_forward():
    module = _MiniCPMOStyle()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())

    assert configured is module
    assert not hasattr(configured, "tts")
    assert configured.forward.__self__ is configured
    assert configured.forward.__func__ is not configured.llm.forward.__func__
    assert configured.get_input_embeddings.__self__ is configured.llm
    assert configured.set_input_embeddings.__self__ is configured.llm
    assert configured.prepare_inputs_for_generation.__self__ is configured.llm
    assert configured._no_split_modules == ["Qwen3DecoderLayer", "MiniCPMODecoderLayer"]
    assert MiniCPMThinkerAdapter.get_fsdp_ignored_module_names(_model_config()) == ["apm", "vpm", "resampler"]


class _MiniCPMOWithEncoders(_MiniCPMOStyle):
    """Counts tower invocations to prove the empty rows skip the encoder."""

    def __init__(self):
        super().__init__()
        self.vision_calls = 0

    def get_vision_embedding(self, data):
        del data
        self.vision_calls += 1
        hidden = torch.ones(1, 2, 4, requires_grad=True)
        return [hidden]


def test_configure_model_does_not_freeze_encoders():
    module = _MiniCPMOWithEncoders()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    assert configured.vpm.training is True
    assert configured.apm.training is True
    assert all(param.requires_grad for param in configured.vpm.parameters())
    assert all(param.requires_grad for param in configured.apm.parameters())


def test_patched_get_vision_embedding_skips_dummy_encoder_when_no_images():
    module = _MiniCPMOWithEncoders()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    states = configured.get_vision_embedding({"pixel_values": [[], []], "input_ids": torch.ones(2, 3)})
    assert states == [[], []]
    assert configured.vision_calls == 0


def test_cloned_vllm_embedding_scatter_supports_backward():
    module = _MiniCPMOWithEncoders()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    embeddings, _ = configured.get_vllm_embedding(
        {
            "input_ids": input_ids,
            "pixel_values": [[torch.zeros(3, 2, 2)]],
            "tgt_sizes": [torch.tensor([[1, 1]], dtype=torch.int32)],
            "image_bound": [torch.tensor([[0, 2]])],
        }
    )
    embeddings.sum().backward()
    assert configured.llm.embed.weight.grad is not None


def test_cloned_vllm_embedding_scatters_unequal_span_lengths():
    # Per-slice grids give spans of different token counts; torch.stack of the
    # per-span aranges raised on the first mixed-length batch.
    class _FiveTokenVision(_MiniCPMOWithEncoders):
        def get_vision_embedding(self, data):
            del data
            self.vision_calls += 1
            return [torch.arange(5 * 4, dtype=torch.float32).reshape(1, 5, 4)]

    module = _FiveTokenVision()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    vision = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    id_embedding = configured.llm.embed(torch.tensor([[7, 7, 7, 7, 7, 7]])).detach()

    embeddings, _ = configured.get_vllm_embedding(
        {
            "input_ids": torch.tensor([[7, 7, 7, 7, 7, 7]]),
            "pixel_values": [[torch.zeros(3, 2, 2)]],
            "tgt_sizes": [torch.tensor([[2, 2]], dtype=torch.int32)],
            # Spans of lengths 2 and 3, with position 2 left to the id embedding.
            "image_bound": [torch.tensor([[0, 2], [3, 6]])],
        }
    )
    torch.testing.assert_close(embeddings[0, 0], vision[0])
    torch.testing.assert_close(embeddings[0, 1], vision[1])
    torch.testing.assert_close(embeddings[0, 2], id_embedding[0, 2])  # non-span position untouched
    torch.testing.assert_close(embeddings[0, 3:6], vision[2:5])


def test_minicpmo_from_pretrained_patches_then_loads_auto_model(monkeypatch):
    from transformers import AutoModel

    loaded = MagicMock(spec=nn.Module)
    calls = []
    patch_calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return loaded

    def fake_patch(*args, **kwargs):
        patch_calls.append((args, kwargs))

    siglip_calls = []

    def fake_siglip_patch(*args, **kwargs):
        siglip_calls.append((args, kwargs))

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        "verl_omni.models.transformers.minicpm_o.patch_minicpm_auto_model_init",
        fake_patch,
    )
    monkeypatch.setattr(
        "verl_omni.models.transformers.minicpm_o.patch_minicpm_siglip_flash_attn_support",
        fake_siglip_patch,
    )
    config = _model_config()

    module = MiniCPMO.from_pretrained(
        "/fake/minicpm",
        torch_dtype=torch.bfloat16,
        config=config.hf_config,
        trust_remote_code=True,
    )

    assert module is loaded
    assert patch_calls == [(("/fake/minicpm",), {"config": config.hf_config})]
    assert calls[0][0] == ("/fake/minicpm",)
    assert calls[0][1]["torch_dtype"] is torch.bfloat16
    assert calls[0][1]["trust_remote_code"] is True
    assert calls[0][1]["config"] is config.hf_config
    assert "init_tts" not in calls[0][1]
    assert siglip_calls == [(("/fake/minicpm",), {"config": config.hf_config})]


def test_minicpmo_from_pretrained_requires_trust_remote_code():
    # The checkpoint defines its classes in remote code, and the patches resolve
    # them through it, so a False value must fail here rather than confuse the load.
    import pytest

    with pytest.raises(ValueError, match="trust_remote_code must be True"):
        MiniCPMO.from_pretrained("/fake/minicpm", config=None, trust_remote_code=False)


def test_configure_model_applies_remote_whisper_compat(monkeypatch):
    from verl_omni.models.transformers import minicpm_o

    seen = []
    monkeypatch.setattr(minicpm_o, "patch_minicpm_whisper_self_attn", lambda module: seen.append(module))
    module = _MiniCPMOStyle()
    MiniCPMThinkerAdapter.configure_model(module, _model_config())
    assert seen == [module]


def test_rollout_adapter_registers_thinker_only_text_pipeline():
    from verl_omni.pipelines.minicpm.omni_rollout_adapter import MiniCPMORolloutAdapter
    from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

    assert OmniRolloutPipelineBase.get_class("minicpmo_4_5") is MiniCPMORolloutAdapter
    stages = MiniCPMORolloutAdapter.build_stage_configs("thinker_only")
    assert len(stages) == 1
    assert stages[0].engine_output_type == "text"
    assert stages[0].final_output_type == "text"
    # AVQA feeds both encoders; the adapter must not disable audio like the
    # text-only RFC design did.
    assert stages[0].requires_multimodal_data is True
    assert MiniCPMORolloutAdapter.get_pipeline_id("thinker_only") == "minicpmo_4_5_thinker_only"
    assert MiniCPMORolloutAdapter.get_stage_engine_extras(0, "thinker_only") == {
        "model_arch": "MiniCPMO45OmniLLMForConditionalGeneration",
        # Mirrors the upstream MiniCPM-o deploy profiles: the AR async
        # scheduler's placeholder accounting underflows under KV-cache
        # preemption (assert in vllm async_scheduler.py).
        "async_scheduling": False,
    }
    assert MiniCPMORolloutAdapter.get_engine_hf_overrides("thinker_only") == {}
    with pytest.raises(ValueError, match="thinker_only only"):
        MiniCPMORolloutAdapter.build_stage_configs("full")


def test_ensure_pipeline_registered_checks_the_plugin_entry_point_first(monkeypatch):
    """A stale plugin entry point must surface before the engine cores spawn."""
    from verl_omni.pipelines.minicpm import omni_rollout_adapter

    calls: list[str] = []
    monkeypatch.setattr(omni_rollout_adapter, "assert_entry_point_installed", lambda: calls.append("assert"))
    monkeypatch.setattr(omni_rollout_adapter, "register_pipeline", lambda pipeline: calls.append("register"))

    omni_rollout_adapter.MiniCPMORolloutAdapter.ensure_pipeline_registered()
    assert calls == ["assert", "register"]


def test_configure_model_applies_omni_embedding_splice_patch():
    class _WithOmniEmbedding(_MiniCPMOStyle):
        def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
            return input_embeddings

    module = _WithOmniEmbedding()
    configured = MiniCPMThinkerAdapter.configure_model(module, _model_config())
    assert getattr(configured, "_verl_omni_get_omni_embedding_patched", False)


def test_merge_packed_media_stashes_per_example_slice_counts():
    data = {
        "pixel_values": [[torch.zeros(3, 2, 2)] * 2, [], [torch.zeros(3, 2, 2)] * 3],
        "tgt_sizes": [
            torch.tensor([[1, 1]] * 2, dtype=torch.int32),
            torch.zeros(0, 2, dtype=torch.int32),
            torch.tensor([[1, 1]] * 3, dtype=torch.int32),
        ],
        "audio_feature_lens": [[], [], []],
        "image_bound": [[[0, 2]], [], [[2, 5]]],
        "audio_bounds": [[], [], []],
    }
    from verl_omni.pipelines.minicpm.thinker_training_adapter import _merge_packed_media

    _merge_packed_media(data)

    assert data["packed_vision_slices"] == [2, 0, 3]
    assert len(data["pixel_values"]) == 1  # folded into one pseudo-row
    assert len(data["pixel_values"][0]) == 5


def test_from_pretrained_applies_training_config_invariants(monkeypatch):
    from transformers import AutoModel

    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMO

    loaded = object()
    captured = {}

    def fake_from_pretrained(path, *args, **kwargs):
        captured.update(kwargs)
        return loaded

    monkeypatch.setattr(AutoModel, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr("verl_omni.models.transformers.minicpm_o.patch_minicpm_auto_model_init", lambda *a, **k: None)
    monkeypatch.setattr(
        "verl_omni.models.transformers.minicpm_o.patch_minicpm_siglip_flash_attn_support", lambda *a, **k: None
    )

    config = SimpleNamespace(init_tts=True, use_cache=True, stream_input=True)
    result = MiniCPMO.from_pretrained("/fake/model", config=config, trust_remote_code=True)

    assert result is loaded
    assert captured["config"] is config
    assert (config.init_tts, config.use_cache, config.stream_input) == (False, False, False)
