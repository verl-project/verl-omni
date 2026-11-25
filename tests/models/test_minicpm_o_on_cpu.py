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
"""CPU tests for MiniCPM-o transformers remote-code compatibility shims."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from verl_omni.models.transformers import minicpm_o


class _FakeConfig(SimpleNamespace):
    model_type = "fake_remote"


def _remote_model_without_post_init():
    """A fresh remote-shape class per call; the patches mark the class itself."""

    class RemoteModelWithoutPostInit(nn.Module):
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.linear = nn.Linear(4, 4)
            self.post_init_calls = 0

        def get_expanded_tied_weights_keys(self, all_submodels=False):
            del all_submodels
            return dict(self._tied_weights_keys)

        def post_init(self):
            self.post_init_calls += 1
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)

        def init_weights(self):
            return

    return RemoteModelWithoutPostInit


def test_wrap_init_with_post_init_runs_once():
    # Wrapping twice must not double-call post_init (the wrap is idempotent).
    model_cls = _remote_model_without_post_init()
    minicpm_o._wrap_init_with_post_init(model_cls)
    minicpm_o._wrap_init_with_post_init(model_cls)
    model = model_cls(_FakeConfig())

    assert model.post_init_calls == 1
    assert model.all_tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}


def test_patch_minicpm_auto_model_init_wraps_dynamic_class(monkeypatch):
    config = SimpleNamespace(auto_map={"AutoModel": "modeling_fake.FakeModel"})
    model_cls = _remote_model_without_post_init()

    import transformers.models.auto.auto_factory as auto_factory

    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", lambda *args, **kwargs: model_cls)

    minicpm_o.patch_minicpm_auto_model_init("/fake/minicpm", config=config)

    model = model_cls(_FakeConfig())
    assert model.post_init_calls == 1
    assert hasattr(model, "all_tied_weights_keys")


class _TwoTupleWhisperAttn(nn.Module):
    """Mirrors current transformers WhisperAttention: returns (hidden_states, attn_weights)."""

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


class _ModuleWithAPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.apm = nn.Module()
        self.apm.layers = nn.ModuleList([_MiniCPMWhisperEncoderLayerStub()])


def test_patch_minicpm_whisper_self_attn_pads_to_three_tuple():
    module = _ModuleWithAPM()
    minicpm_o.patch_minicpm_whisper_self_attn(module)
    hidden = torch.ones(1, 2, 4)

    # The remote layer's 3-way unpack succeeds, and the singular past_key_value
    # kwarg is still honoured.
    out, past = module.apm.layers[0](hidden, past_key_values="cache")
    assert torch.equal(out, hidden)
    assert past == "cache"

    # Wrapping twice must not re-wrap the same module.
    first_forward = module.apm.layers[0].self_attn.forward
    minicpm_o.patch_minicpm_whisper_self_attn(module)
    assert module.apm.layers[0].self_attn.forward is first_forward
    out, _ = module.apm.layers[0](hidden)
    assert torch.equal(out, hidden)


def _wire_main_model_module(monkeypatch, *bound_classes):
    """Bind classes into the auto_map-resolved main modeling module's namespace.

    Mirrors the loader: get_class_from_dynamic_module resolves the config's
    auto_map entry, and the main module's namespace holds the classes the
    remote ``from .modeling_navit_siglip import ...`` binds. Classes NOT
    passed model the orphaned copies a divergent cache dir produces.
    """
    import sys
    import types as types_module

    from transformers import PreTrainedModel

    main_name = "transformers_modules.fake.modeling_minicpmo"
    main_module = types_module.ModuleType(main_name)
    for bound_cls in bound_classes:
        setattr(main_module, bound_cls.__name__, bound_cls)

    class _FakeMainModel(PreTrainedModel):
        pass

    _FakeMainModel.__module__ = main_name
    main_module.MiniCPMO = _FakeMainModel
    monkeypatch.setitem(sys.modules, main_name, main_module)

    import transformers.models.auto.auto_factory as auto_factory

    monkeypatch.setattr(
        auto_factory,
        "get_class_from_dynamic_module",
        lambda *args, **kwargs: _FakeMainModel,
    )
    return SimpleNamespace(auto_map={"AutoModel": "modeling_minicpmo.MiniCPMO"})


def test_patch_minicpm_siglip_flash_attn_support_aliases_old_flag(monkeypatch):
    from transformers import PreTrainedModel

    from verl_omni.models.transformers import minicpm_o

    class _RemoteSiglip(PreTrainedModel):
        _supports_flash_attn_2 = True

    config = _wire_main_model_module(monkeypatch, _RemoteSiglip)
    minicpm_o.patch_minicpm_siglip_flash_attn_support("/fake/minicpm", config=config)

    assert _RemoteSiglip.__dict__["_supports_flash_attn"] is True
    assert _RemoteSiglip._supports_flash_attn_2 is True  # remote declaration untouched

    # A second call must not re-alias an already-aliased class.
    marker = object()
    _RemoteSiglip._supports_flash_attn = marker
    minicpm_o.patch_minicpm_siglip_flash_attn_support("/fake/minicpm", config=config)
    assert _RemoteSiglip._supports_flash_attn is marker


def test_patch_minicpm_siglip_never_fabricates_support(monkeypatch):
    from transformers import PreTrainedModel

    from verl_omni.models.transformers import minicpm_o

    class _NoFlashAttn(PreTrainedModel):
        _supports_flash_attn_2 = False

    class _AlreadyRenamed(PreTrainedModel):
        _supports_flash_attn_2 = True
        _supports_flash_attn = False

    config = _wire_main_model_module(monkeypatch, _NoFlashAttn, _AlreadyRenamed)
    minicpm_o.patch_minicpm_siglip_flash_attn_support("/fake/minicpm", config=config)
    assert "_supports_flash_attn" not in _NoFlashAttn.__dict__  # old flag False: no alias
    assert "_verl_omni_siglip_fa2_aliased" not in _AlreadyRenamed.__dict__  # new name declared: untouched


def test_patch_minicpm_siglip_aliases_the_class_the_model_module_uses(monkeypatch):
    # Local-path cache-hash divergence: two distinct SiglipVisionTransformer
    # class objects; only the one bound into the auto_map-resolved main
    # modeling module is the class the model imports — the alias must land
    # there, not on the orphan.
    from transformers import PreTrainedModel

    from verl_omni.models.transformers import minicpm_o

    class _OrphanCopy(PreTrainedModel):
        _supports_flash_attn_2 = True

    class _UsedCopy(PreTrainedModel):
        _supports_flash_attn_2 = True

    config = _wire_main_model_module(monkeypatch, _UsedCopy)  # the orphan is never bound into main
    minicpm_o.patch_minicpm_siglip_flash_attn_support("/fake/minicpm", config=config)

    assert _UsedCopy.__dict__.get("_supports_flash_attn") is True
    assert "_supports_flash_attn" not in _OrphanCopy.__dict__  # orphan untouched


class _AudioModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.apm = torch.nn.Module()
        self.apm.conv1 = torch.nn.Linear(4, 4)
        self.llm = torch.nn.Module()
        self.llm.config = SimpleNamespace(hidden_size=8)
        self.original_calls = []

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        self.original_calls.append((data, chunk_length, dummy))
        if len(data.get("audio_features", [])) == 0:
            return []  # the remote's empty branch never touches the lens
        # The real remote's first statement inside the audio branch:
        # torch.hstack over per-sample 1-D tensors. A nested-list lens
        # contract violation raises here — the line the per-clip
        # regression slipped past.
        audio_feature_lens = torch.hstack(data["audio_feature_lens"])
        return [[torch.zeros(int(length), 8)] for length in audio_feature_lens.tolist()]


def _fresh_audio_module():
    module = _AudioModule()
    if hasattr(module, "_verl_omni_get_audio_embedding_patched"):
        delattr(module, "_verl_omni_get_audio_embedding_patched")
    return module


def test_patch_get_audio_embedding_returns_zero_token_for_empty_training_batch():
    from verl_omni.models.transformers import minicpm_o

    module = _fresh_audio_module()
    module.train()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    result = module.get_audio_embedding({"audio_features": []}, chunk_length=1.0)
    assert len(result) == 1
    # One zero "audio token": get_omni_embedding's `audio_embeddings[0].mean() * 0`
    # branch works without running the Whisper encoder at all.
    assert result[0].shape == (1, 8, 1)
    assert result[0].dtype == module.apm.conv1.weight.dtype
    assert module.original_calls == []


def test_patch_get_audio_embedding_runs_each_clip_alone():
    from verl_omni.models.transformers import minicpm_o

    module = _fresh_audio_module()
    module.train()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    result = module.get_audio_embedding(
        {"audio_features": torch.zeros(2, 80, 10), "audio_feature_lens": [torch.tensor([10]), torch.tensor([6])]},
        chunk_length=1.0,
    )
    # One exact-length, contiguous [1, 80, len] clip per original call, with
    # per-sample 1-D tensor lens (hstack-able, on the features' device).
    assert [call[0]["audio_features"].shape for call in module.original_calls] == [(1, 80, 10), (1, 80, 6)]
    assert [call[0]["audio_features"].is_contiguous() for call in module.original_calls] == [True, True]
    assert [torch.hstack(call[0]["audio_feature_lens"]).tolist() for call in module.original_calls] == [[10], [6]]
    assert all(
        isinstance(row, torch.Tensor) and row.device == torch.device("cpu")
        for call in module.original_calls
        for row in call[0]["audio_feature_lens"]
    )
    assert len(result) == 2  # regrouped per row

    module.eval()
    module.get_audio_embedding({"audio_features": []}, chunk_length=1.0)
    # 2 per-clip calls + the eval-empty delegation to the remote's own branch.
    assert len(module.original_calls) == 3


class _RemoteBuggyOmniModule(torch.nn.Module):
    """Remote get_omni_embedding with the dedented splice: only the last row is written."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(stream_input=False)
        self.original_calls = []

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        return data["audio_embeddings_stub"]

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        self.original_calls.append(dict(data))
        if len(data.get("audio_features", [])) == 0:
            return input_embeddings  # the remote's audio-free branch
        # The remote bug, verbatim in structure: the loop only rebinds, the
        # splice below uses the leaked row index.
        audio_embeddings = data["audio_embeddings_stub"]
        audio_bounds = data["audio_bounds"]
        i = None
        for i in range(len(input_embeddings)):
            audio_embs = audio_embeddings[i]
            bounds = audio_bounds[i]
        one_to_one_match = len(audio_embs) == len(bounds) and all(
            embs.shape[0] == int(bound[1] - bound[0]) for embs, bound in zip(audio_embs, bounds, strict=False)
        )
        if one_to_one_match:
            for embs, bound in zip(audio_embs, bounds, strict=False):
                input_embeddings[i, int(bound[0]) : int(bound[1])] = embs
        else:
            flat = torch.cat(audio_embs, dim=0)
            offset = 0
            for bound in bounds:
                n = int(bound[1] - bound[0])
                input_embeddings[i, int(bound[0]) : int(bound[1])] = flat[offset : offset + n]
                offset += n
        return input_embeddings


def _omni_splice_data():
    # Three rows, one audio span each, distinct marker values per row.
    embeddings = [torch.full((3, 4), float(row + 1)) for row in range(3)]
    data = {
        "audio_features": torch.zeros(3, 80, 10),  # non-empty signals the audio path
        "audio_embeddings_stub": [[emb] for emb in embeddings],
        "audio_bounds": [[[2, 5]], [[1, 4]], [[0, 3]]],
    }
    return data, embeddings


def test_patch_get_omni_embedding_splices_every_row():
    from verl_omni.models.transformers import minicpm_o

    module = _RemoteBuggyOmniModule()
    data, embeddings = _omni_splice_data()
    base = torch.zeros(3, 6, 4)

    # Unpatched: only the last row is written (the remote bug).
    buggy = module.get_omni_embedding(data, base.clone())
    assert buggy[0].abs().sum() == 0 and buggy[1].abs().sum() == 0
    assert torch.equal(buggy[2, 0:3], embeddings[2])

    minicpm_o.patch_minicpm_get_omni_embedding(module)
    patched = module.get_omni_embedding(data, base.clone())

    assert torch.equal(patched[0, 2:5], embeddings[0])
    assert torch.equal(patched[1, 1:4], embeddings[1])
    assert torch.equal(patched[2, 0:3], embeddings[2])
    # The input tensor is never mutated in place (clone-before-write).
    assert base.abs().sum() == 0


def test_patch_get_omni_embedding_flat_layout_and_mismatch_check():
    from verl_omni.models.transformers import minicpm_o

    module = _RemoteBuggyOmniModule()
    minicpm_o.patch_minicpm_get_omni_embedding(module)

    # Flat layout: one clip group covering two spans (one-to-one fails).
    clips = [torch.arange(5, dtype=torch.float).unsqueeze(1).repeat(1, 4)]
    data = {
        "audio_features": torch.zeros(1, 80, 10),
        "audio_embeddings_stub": [clips],
        "audio_bounds": [[[0, 2], [3, 6]]],
    }
    patched = module.get_omni_embedding(data, torch.zeros(1, 6, 4))
    assert torch.equal(patched[0, 0:2], clips[0][0:2])
    assert torch.equal(patched[0, 3:6], clips[0][2:5])

    data["audio_bounds"] = [[[0, 2], [4, 9]]]  # 5 embeddings vs 7 bound tokens
    with pytest.raises(ValueError, match="Audio total length mismatch"):
        module.get_omni_embedding(data, torch.zeros(1, 6, 4))


def test_patch_get_omni_embedding_delegates_streaming_and_audio_free():
    from verl_omni.models.transformers import minicpm_o

    module = _RemoteBuggyOmniModule()
    minicpm_o.patch_minicpm_get_omni_embedding(module)

    data, _ = _omni_splice_data()
    module.get_omni_embedding(data, torch.zeros(3, 6, 4), stream_input=True)
    assert len(module.original_calls) == 1  # streaming delegates

    module.config = SimpleNamespace(stream_input=True)
    module.get_omni_embedding(data, torch.zeros(3, 6, 4))
    assert len(module.original_calls) == 2  # config-level streaming delegates too

    module.config = SimpleNamespace(stream_input=False)
    module.get_omni_embedding({"audio_features": []}, torch.zeros(3, 6, 4))
    assert len(module.original_calls) == 3  # audio-free delegates (training anchor)


class _VisionTowerModule(torch.nn.Module):
    """Remote-shaped get_vision_embedding recording every tower invocation."""

    def __init__(self):
        super().__init__()
        self.tower_calls = []

    def get_vision_embedding(self, data):
        self.tower_calls.append((data["pixel_values"], data["tgt_sizes"]))
        rows = []
        for values in data["pixel_values"]:
            if not values:
                rows.append([])
                continue
            # One marker token per slice so concatenation order is checkable.
            rows.append(torch.stack([torch.full((1, 4), float(values[i][0, 0, 0].item())) for i in range(len(values))]))
        return rows


def test_patched_get_vllm_embedding_accepts_bare_bound_tensor():
    # A collated (N, 2) image_bound tensor must not be bool()'d by an `or`
    # default — bool of a multi-element tensor raises.
    from types import SimpleNamespace

    class _Embeds(nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = SimpleNamespace(
                model=SimpleNamespace(embed_tokens=nn.Embedding(8, 4)),
                config=SimpleNamespace(),
            )

        def get_vision_embedding(self, data):
            del data
            return [[]]

    module = _Embeds()
    minicpm_o.patch_minicpm_get_vllm_embedding(module)
    embeddings, _ = module.get_vllm_embedding(
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "image_bound": torch.tensor([[0, 2]], dtype=torch.long),
        }
    )
    assert embeddings.shape == (1, 4, 4)


def test_patch_get_vision_embedding_runs_each_sample_alone():
    from verl_omni.models.transformers import minicpm_o

    module = _VisionTowerModule()
    minicpm_o.patch_minicpm_get_vision_embedding(module)
    slices = [
        [torch.full((3, 2, 2), 1.0), torch.full((3, 2, 2), 1.0)],  # row 0: two slices
        [],  # row 1: empty — no tower call
        [torch.full((3, 2, 2), 3.0)],  # row 2: one slice
    ]
    tgt = [
        torch.tensor([[1, 2]], dtype=torch.int32),
        torch.zeros(0, 2, dtype=torch.int32),
        torch.tensor([[2, 1]], dtype=torch.int32),
    ]
    result = module.get_vision_embedding({"pixel_values": slices, "tgt_sizes": tgt})

    assert len(module.tower_calls) == 2  # one per non-empty sample, none for the empty row
    assert module.tower_calls[0][0] == [slices[0]]  # exactly that row's slices
    assert module.tower_calls[1][0] == [slices[2]]
    assert torch.equal(result[0][:, 0, 0], torch.tensor([1.0, 1.0]))
    assert result[1] == []
    assert torch.equal(result[2][:, 0, 0], torch.tensor([3.0]))


def test_patch_get_vision_embedding_resplits_the_packed_pseudo_row():
    from verl_omni.models.transformers import minicpm_o

    module = _VisionTowerModule()
    minicpm_o.patch_minicpm_get_vision_embedding(module)
    flat = [torch.full((3, 2, 2), float(i)) for i in range(5)]  # 5 slices, flat span order
    data = {
        "pixel_values": [flat],  # one packed pseudo-row
        "tgt_sizes": [torch.tensor([[1, 1]] * 5, dtype=torch.int32)],
        "packed_vision_slices": [2, 0, 3],  # three examples: 2 + 0 + 3 slices
    }
    result = module.get_vision_embedding(data)

    assert len(module.tower_calls) == 2  # per example, empty example skipped
    assert module.tower_calls[0][0] == [[flat[0], flat[1]]]
    assert module.tower_calls[1][0] == [[flat[2], flat[3], flat[4]]]
    assert isinstance(result, list) and len(result) == 1  # back to one pseudo-row
    assert torch.equal(result[0][:, 0, 0], torch.arange(5.0))  # flat span order preserved


def test_patch_get_audio_embedding_delegates_non_tensor_features(caplog):
    from verl_omni.models.transformers import minicpm_o

    module = _fresh_audio_module()
    module.train()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    features = ["clip-a", "clip-b"]  # not the stacked tensor form

    with caplog.at_level("WARNING", logger="verl_omni.models.transformers.minicpm_o"):
        module.get_audio_embedding(
            {"audio_features": features, "audio_feature_lens": [torch.tensor([3]), torch.tensor([5])]},
            chunk_length=1.0,
        )

    assert len(module.original_calls) == 1  # whole-batch delegation, no per-clip calls
    assert module.original_calls[0][0]["audio_features"] == features
    assert "delegating" in caplog.text


def test_patch_get_audio_embedding_rejects_clip_len_disagreement():
    from verl_omni.models.transformers import minicpm_o

    module = _fresh_audio_module()
    module.train()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    with pytest.raises(ValueError, match="disagree"):
        module.get_audio_embedding(
            {"audio_features": torch.zeros(2, 80, 10), "audio_feature_lens": [torch.tensor([10])]},
            chunk_length=1.0,
        )


def _pooled_rows(mel_frames: int) -> int:
    """Tower rows a clip pools to, by the checkpoint's ``_get_feat_extract_output_lengths``."""
    return (((mel_frames - 1) // 2 + 1) - 5) // 5 + 1


class _DuplexAudioModule(torch.nn.Module):
    """Remote-shaped audio module recording both encoders' calls.

    The streaming spy reads its KV history the way the remote method does
    (layer 0's key length) and leaves an ``EncoderDecoderCache`` behind, as the
    remote encoder does; it returns one row per pooled frame, filled with the
    history's frame count, so a streamed embedding carries both its shape and
    the history it was encoded with.
    """

    def __init__(self):
        super().__init__()
        self.apm = torch.nn.Module()
        self.apm.conv1 = torch.nn.Linear(4, 4)
        self.llm = torch.nn.Module()
        self.llm.config = SimpleNamespace(hidden_size=8)
        self.audio_past_key_values = None
        self.offline_calls = []
        self.streaming_calls = []

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        lengths = torch.hstack(data["audio_feature_lens"]).tolist()
        self.offline_calls.append((tuple(data["audio_features"].shape), lengths))
        return [[torch.full((_pooled_rows(length), 8), -1.0) for length in lengths]]

    def get_audio_embedding_streaming(
        self, data, use_extra_context=False, prefix_extra_frames=1, suffix_extra_frames=1, cnn_min_length=None
    ):
        history_frames = 0 if self.audio_past_key_values is None else self.audio_past_key_values[0][0].shape[2]
        length = int(torch.hstack(data["audio_feature_lens"]).item())
        self.streaming_calls.append(
            SimpleNamespace(
                clip=tuple(data["audio_features"].shape),
                contiguous=data["audio_features"].is_contiguous(),
                use_extra_context=use_extra_context,
                prefix_extra_frames=prefix_extra_frames,
                suffix_extra_frames=suffix_extra_frames,
                cnn_min_length=cnn_min_length,
                history_frames=history_frames,
                length=length,
            )
        )
        grown = torch.zeros(1, 1, history_frames + 50, 1)
        self_attention = DynamicCache()
        self_attention.update(grown, grown, 0)
        self.audio_past_key_values = EncoderDecoderCache(self_attention, DynamicCache())
        return [[torch.full((_pooled_rows(length), 8), float(history_frames))]]


def _duplex_audio_batch(rows: int = 2, units: int = 3) -> dict:
    """``rows`` episodes of one 130-frame reference clip and ``units`` unit clips, padded layout."""
    # Unit 0's mel carries no prefix context (102 frames); later units carry 2+2 (104).
    row_lengths = [130, 102, *([104] * (units - 1))]
    lengths = row_lengths * rows
    features = torch.zeros(len(lengths), 80, max(lengths))
    bounds, start = [], 0
    for length in lengths:
        bounds.append([start, start + _pooled_rows(length)])
        start += _pooled_rows(length) + 1
    return {
        "audio_features": features,
        "audio_feature_lens": [torch.tensor(row_lengths) for _ in range(rows)],
        "audio_bounds": [
            bounds[index : index + len(row_lengths)] for index in range(0, len(lengths), len(row_lengths))
        ],
        "audio_clip_kinds": ["reference", "unit", *(["unit"] * (units - 1))] * rows,
    }


def _streamed_call(history_frames: int, length: int, prefix_extra_frames: int) -> dict:
    """The streaming call the engine makes for one unit clip."""
    return dict(
        clip=(1, 80, length),
        contiguous=True,
        use_extra_context=True,
        prefix_extra_frames=prefix_extra_frames,
        suffix_extra_frames=2,
        cnn_min_length=None,
        history_frames=history_frames,
        length=length,
    )


def test_patch_get_audio_embedding_streams_duplex_units_with_the_engines_call_convention():
    from verl_omni.models.transformers import minicpm_o

    module = _DuplexAudioModule()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    data = _duplex_audio_batch(rows=2, units=3)

    result = module.get_audio_embedding(data)

    # Feed order per row: the reference clip stays offline, every unit streams
    # with the engine's kwargs; the history is None before each row's unit 0
    # and grows 0 -> 50 -> 100 across the row's units.
    assert module.offline_calls == [((1, 80, 130), [130])] * 2
    assert [vars(call) for call in module.streaming_calls] == (
        [_streamed_call(0, 102, 0), _streamed_call(50, 104, 2), _streamed_call(100, 104, 2)] * 2
    )

    # Regrouped per row in clip order, each embedding as long as its span.
    assert len(result) == 2
    for embeddings, bounds in zip(result, data["audio_bounds"], strict=True):
        assert [tuple(emb.shape) for emb in embeddings] == [(bound[1] - bound[0], 8) for bound in bounds]
        assert embeddings[0][0, 0].item() == -1.0  # the offline reference
        assert [emb[0, 0].item() for emb in embeddings[1:]] == [0.0, 50.0, 100.0]
    # The last row's history is per-forward scratch: the module is left as found.
    assert module.audio_past_key_values is None


def test_patch_get_audio_embedding_streams_packed_episodes_independently():
    from verl_omni.models.transformers import minicpm_o

    module = _DuplexAudioModule()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    unpacked = _duplex_audio_batch(rows=2, units=2)
    packed = dict(unpacked)
    packed["audio_feature_lens"] = [torch.cat(unpacked["audio_feature_lens"])]
    packed["audio_bounds"] = [[span for spans in unpacked["audio_bounds"] for span in spans]]
    packed["packed_audio_clips"] = [3, 3]

    result = module.get_audio_embedding(packed)

    assert [call.history_frames for call in module.streaming_calls] == [0, 50, 0, 50]  # episode 1 restarts from None
    assert [vars(call) for call in module.streaming_calls[2:]] == [
        _streamed_call(0, 102, 0),
        _streamed_call(50, 104, 2),
    ]
    assert len(result) == 1 and len(result[0]) == 6  # one pseudo-row, clips in order


def test_patch_get_audio_embedding_without_clip_kinds_stays_offline():
    from verl_omni.models.transformers import minicpm_o

    module = _DuplexAudioModule()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    data = _duplex_audio_batch(rows=1, units=2)
    del data["audio_clip_kinds"]

    result = module.get_audio_embedding(data)

    assert module.streaming_calls == []
    assert [call[1] for call in module.offline_calls] == [[130], [102], [104]]
    assert [tuple(emb.shape) for emb in result[0]] == [(13, 8), (10, 8), (10, 8)]


@pytest.mark.parametrize(
    "kinds, match",
    [
        (["reference", "unit"], "disagree"),  # two kinds over three clips
        (["reference", "unit", "clip"], "expected only 'reference' or 'unit'"),
    ],
)
def test_patch_get_audio_embedding_rejects_malformed_clip_kinds(kinds, match):
    from verl_omni.models.transformers import minicpm_o

    module = _DuplexAudioModule()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    data = _duplex_audio_batch(rows=1, units=2)
    data["audio_clip_kinds"] = kinds
    with pytest.raises(ValueError, match=match):
        module.get_audio_embedding(data)
    assert module.streaming_calls == [] and module.offline_calls == []  # rejected before any tower call


def _minicpm_snapshot() -> Path | None:
    """The local MiniCPM-o 4.5 snapshot's directory, or ``None`` without one (no download)."""
    override = os.environ.get("VERL_OMNI_DUPLEX_REF_AUDIO_MODEL_PATH")
    if override:
        return Path(override) if (Path(override) / "modeling_minicpmo.py").exists() else None
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    remote_code = next(iter(sorted(hub.glob("models--openbmb--MiniCPM-o-4_5/snapshots/*/modeling_minicpmo.py"))), None)
    return remote_code.parent if remote_code is not None else None


@pytest.mark.skipif(_minicpm_snapshot() is None, reason="needs the local MiniCPM-o 4.5 snapshot's remote code")
def test_patched_audio_embedding_streams_units_through_the_real_remote_encoder():
    # The checkpoint's remote code on a tiny config: the streaming path runs
    # under the installed transformers, and units >= 1 (encoded with the KV
    # history the offline encoder lacks) come out different from offline.
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    from verl_omni.models.transformers import minicpm_o

    snapshot = str(_minicpm_snapshot())
    config_cls = get_class_from_dynamic_module(
        "configuration_minicpmo.MiniCPMOConfig", snapshot, trust_remote_code=True
    )
    model_cls = get_class_from_dynamic_module("modeling_minicpmo.MiniCPMO", snapshot, trust_remote_code=True)
    config = config_cls(
        init_vision=False,
        init_tts=False,
        init_audio=True,
        audio_config=dict(
            encoder_layers=2,
            d_model=32,
            encoder_ffn_dim=128,
            encoder_attention_heads=4,
            num_mel_bins=80,
            _attn_implementation="eager",
        ),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
        head_dim=8,
        use_cache=False,
        stream_input=False,
        _attn_implementation="eager",
    )
    torch.manual_seed(0)
    module = model_cls(config).eval()
    minicpm_o.patch_minicpm_whisper_self_attn(module)
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    lengths = [102, 104, 104]
    features = torch.randn(3, 80, 104)
    data = {
        "audio_features": features,
        "audio_feature_lens": [torch.tensor(lengths)],
        "audio_clip_kinds": ["unit"] * 3,
    }

    with torch.no_grad():
        streamed = module.get_audio_embedding(data)[0]
        offline = module.get_audio_embedding({key: value for key, value in data.items() if key != "audio_clip_kinds"})[
            0
        ]

    assert [tuple(emb.shape) for emb in streamed] == [(10, 32)] * 3 == [tuple(emb.shape) for emb in offline]
    gaps = [(stream - encode).abs().max().item() for stream, encode in zip(streamed, offline, strict=True)]
    assert gaps[1] > 0 and gaps[2] > 0
    assert module.audio_past_key_values is None
