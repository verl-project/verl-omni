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
"""CPU tests for transformers remote-code compatibility shims."""

from types import SimpleNamespace

import torch.nn as nn

from verl_omni.models.transformers import remote_code_compat


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
    if hasattr(model_cls, remote_code_compat._PATCHED_ATTR):
        delattr(model_cls, remote_code_compat._PATCHED_ATTR)

    remote_code_compat.wrap_model_init_with_post_init(model_cls)
    model = model_cls(_FakeConfig())

    assert model.post_init_calls == 1
    assert model.all_tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}


def test_wrap_model_init_with_post_init_is_idempotent():
    model_cls = _RemoteModelWithoutPostInit
    if hasattr(model_cls, remote_code_compat._PATCHED_ATTR):
        delattr(model_cls, remote_code_compat._PATCHED_ATTR)

    remote_code_compat.wrap_model_init_with_post_init(model_cls)
    remote_code_compat.wrap_model_init_with_post_init(model_cls)

    model = model_cls(_FakeConfig())
    assert model.post_init_calls == 1


def test_patch_remote_auto_model_init_wraps_dynamic_class(monkeypatch):
    config = SimpleNamespace(auto_map={"AutoModel": "modeling_fake.FakeModel"})
    model_cls = _RemoteModelWithoutPostInit
    if hasattr(model_cls, remote_code_compat._PATCHED_ATTR):
        delattr(model_cls, remote_code_compat._PATCHED_ATTR)

    monkeypatch.setattr(remote_code_compat, "_needs_transformers5_compat", lambda: True)

    import transformers.models.auto.auto_factory as auto_factory

    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", lambda *args, **kwargs: model_cls)

    remote_code_compat.patch_remote_auto_model_init(
        "/fake/minicpm",
        trust_remote_code=True,
        config=config,
    )

    model = model_cls(_FakeConfig())
    assert model.post_init_calls == 1
    assert hasattr(model, "all_tied_weights_keys")
