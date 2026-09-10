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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl_omni.models.transformers import minicpm_o


class _FakeConfig(SimpleNamespace):
    model_type = "fake_remote"


class _RemoteModelWithoutPostInit(nn.Module):
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


def test_wrap_model_init_with_post_init_adds_all_tied_weights_keys():
    model_cls = _RemoteModelWithoutPostInit
    if hasattr(model_cls, minicpm_o._PATCHED_ATTR):
        delattr(model_cls, minicpm_o._PATCHED_ATTR)

    minicpm_o.wrap_model_init_with_post_init(model_cls)
    model = model_cls(_FakeConfig())

    assert model.post_init_calls == 1
    assert model.all_tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}


def test_wrap_model_init_with_post_init_is_idempotent():
    model_cls = _RemoteModelWithoutPostInit
    if hasattr(model_cls, minicpm_o._PATCHED_ATTR):
        delattr(model_cls, minicpm_o._PATCHED_ATTR)

    minicpm_o.wrap_model_init_with_post_init(model_cls)
    minicpm_o.wrap_model_init_with_post_init(model_cls)

    model = model_cls(_FakeConfig())
    assert model.post_init_calls == 1


def test_patch_remote_auto_model_init_wraps_dynamic_class(monkeypatch):
    config = SimpleNamespace(auto_map={"AutoModel": "modeling_fake.FakeModel"})
    model_cls = _RemoteModelWithoutPostInit
    if hasattr(model_cls, minicpm_o._PATCHED_ATTR):
        delattr(model_cls, minicpm_o._PATCHED_ATTR)

    monkeypatch.setattr(minicpm_o, "_needs_transformers5_compat", lambda: True)

    import transformers.models.auto.auto_factory as auto_factory

    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", lambda *args, **kwargs: model_cls)

    minicpm_o.patch_remote_auto_model_init(
        "/fake/minicpm",
        trust_remote_code=True,
        config=config,
    )

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


def test_unpatched_whisper_layer_cannot_unpack_two_tuple_attn():
    layer = _MiniCPMWhisperEncoderLayerStub()
    hidden = torch.ones(1, 2, 4)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        layer(hidden)


def test_patch_remote_whisper_self_attn_pads_to_three_tuple():
    module = _ModuleWithAPM()
    minicpm_o.patch_remote_whisper_self_attn(module)
    hidden = torch.ones(1, 2, 4)

    out, past = module.apm.layers[0](hidden, past_key_values="cache")

    assert torch.equal(out, hidden)
    assert past == "cache"


def test_patch_remote_whisper_self_attn_is_idempotent():
    module = _ModuleWithAPM()
    minicpm_o.patch_remote_whisper_self_attn(module)
    first_forward = module.apm.layers[0].self_attn.forward
    minicpm_o.patch_remote_whisper_self_attn(module)
    assert module.apm.layers[0].self_attn.forward is first_forward
    hidden = torch.ones(1, 2, 4)
    out, _ = module.apm.layers[0](hidden)
    assert torch.equal(out, hidden)


def test_patch_remote_whisper_self_attn_noop_without_apm():
    minicpm_o.patch_remote_whisper_self_attn(nn.Linear(4, 4))


def _fake_siglip_module(monkeypatch, *classes):
    """Point the dynamic-module resolver at a synthetic siglip module."""
    import sys
    import types as types_module

    module = types_module.ModuleType("transformers_modules.fake.modeling_navit_siglip")
    for fake_cls in classes:
        module.__dict__[fake_cls.__name__] = fake_cls
        fake_cls.__module__ = module.__name__  # the patch scans sys.modules[cls.__module__]
    sys.modules[module.__name__] = module

    import transformers.models.auto.auto_factory as auto_factory

    monkeypatch.setattr(
        auto_factory,
        "get_class_from_dynamic_module",
        lambda *args, **kwargs: classes[0],
    )
    return module


def test_patch_remote_siglip_flash_attn_support_aliases_old_flag(monkeypatch):
    from transformers import PreTrainedModel

    from verl_omni.models.transformers import minicpm_o

    class _RemoteSiglip(PreTrainedModel):
        _supports_flash_attn_2 = True

    fake_cls = _RemoteSiglip
    _fake_siglip_module(monkeypatch, fake_cls)
    minicpm_o.patch_remote_siglip_flash_attn_support("/fake/minicpm", trust_remote_code=True)

    assert fake_cls.__dict__["_supports_flash_attn"] is True
    assert fake_cls._supports_flash_attn_2 is True  # remote declaration untouched


def test_patch_remote_siglip_flash_attn_support_is_idempotent(monkeypatch):
    from transformers import PreTrainedModel

    from verl_omni.models.transformers import minicpm_o

    class _RemoteSiglip(PreTrainedModel):
        _supports_flash_attn_2 = True

    _fake_siglip_module(monkeypatch, _RemoteSiglip)
    minicpm_o.patch_remote_siglip_flash_attn_support("/fake/minicpm", trust_remote_code=True)
    marker = object()
    _RemoteSiglip._supports_flash_attn = marker
    minicpm_o.patch_remote_siglip_flash_attn_support("/fake/minicpm", trust_remote_code=True)
    assert _RemoteSiglip._supports_flash_attn is marker  # second call is a no-op


def test_patch_remote_siglip_never_fabricates_support(monkeypatch):
    from transformers import PreTrainedModel

    from verl_omni.models.transformers import minicpm_o

    class _NoFlashAttn(PreTrainedModel):
        _supports_flash_attn_2 = False

    class _AlreadyRenamed(PreTrainedModel):
        _supports_flash_attn_2 = True
        _supports_flash_attn = False

    _fake_siglip_module(monkeypatch, _NoFlashAttn, _AlreadyRenamed)
    minicpm_o.patch_remote_siglip_flash_attn_support("/fake/minicpm", trust_remote_code=True)
    assert "_supports_flash_attn" not in _NoFlashAttn.__dict__  # old flag False: no alias
    assert "_verl_omni_siglip_fa2_aliased" not in _AlreadyRenamed.__dict__  # new name declared: untouched


def test_patch_remote_siglip_skips_without_trust_or_transformers4(monkeypatch):
    import transformers.models.auto.auto_factory as auto_factory

    from verl_omni.models.transformers import minicpm_o

    calls = []
    monkeypatch.setattr(
        auto_factory,
        "get_class_from_dynamic_module",
        lambda *args, **kwargs: calls.append(args),
    )
    minicpm_o.patch_remote_siglip_flash_attn_support("/fake/minicpm", trust_remote_code=False)
    assert calls == []  # never resolves remote code without trust_remote_code
    monkeypatch.setattr(minicpm_o, "_needs_transformers5_compat", lambda: False)
    minicpm_o.patch_remote_siglip_flash_attn_support("/fake/minicpm", trust_remote_code=True)
    assert calls == []  # no-op on transformers < 5


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
        return [torch.zeros(1, 8, 1)]


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


def test_patch_get_audio_embedding_delegates_for_real_audio_or_eval():
    from verl_omni.models.transformers import minicpm_o

    module = _fresh_audio_module()
    module.train()
    minicpm_o.patch_minicpm_get_audio_embedding(module)
    module.get_audio_embedding({"audio_features": torch.zeros(1, 80, 10)}, chunk_length=1.0)
    assert len(module.original_calls) == 1  # real audio: original Whisper path

    module.eval()
    module.get_audio_embedding({"audio_features": []}, chunk_length=1.0)
    assert len(module.original_calls) == 2  # eval: remote's own no-dummy branch


def test_patch_get_audio_embedding_noop_without_apm_or_llm():
    from verl_omni.models.transformers import minicpm_o

    bare = torch.nn.Linear(4, 4)
    minicpm_o.patch_minicpm_get_audio_embedding(bare)
    assert not hasattr(bare, "_verl_omni_get_audio_embedding_patched")
