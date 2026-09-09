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
