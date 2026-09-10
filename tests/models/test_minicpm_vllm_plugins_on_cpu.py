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
"""CPU tests for the vLLM general plugin aliasing embed_multimodal."""

from __future__ import annotations

import sys

from verl_omni.vllm_plugins import register

_MODULE = "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm"


def _engine_cls():
    module = __import__(_MODULE, fromlist=["MiniCPMO45OmniLLMForConditionalGeneration"])
    return module.MiniCPMO45OmniLLMForConditionalGeneration


def test_register_aliases_embed_multimodal_on_engine_class():
    register()
    model_cls = _engine_cls()
    # Either natively implemented or aliased onto get_multimodal_embeddings —
    # never left as the SupportsMultiModal stub returning None.
    assert "embed_multimodal" in model_cls.__dict__


def test_register_is_idempotent():
    register()
    model_cls = _engine_cls()
    current = model_cls.embed_multimodal
    register()  # plugins may be loaded multiple times per process
    assert model_cls.embed_multimodal is current


def test_register_skips_quietly_without_vllm_omni_module(monkeypatch):
    monkeypatch.setitem(sys.modules, _MODULE, None)  # forces ImportError
    register()  # must not raise: the plugin runs in every vLLM process


def test_register_does_not_overwrite_native_implementation():
    register()
    model_cls = _engine_cls()
    native = model_cls.__dict__.get("embed_multimodal")
    register()
    assert model_cls.__dict__.get("embed_multimodal") is native


def test_alias_points_at_the_old_style_method_when_needed():
    model_cls = _engine_cls()
    native = model_cls.__dict__.get("embed_multimodal")
    if native is not None:
        delattr(model_cls, "embed_multimodal")  # simulate the unpatched class
    try:
        register()
        assert model_cls.embed_multimodal is model_cls.get_multimodal_embeddings
    finally:
        if native is not None:  # restore a native implementation if one existed
            model_cls.embed_multimodal = native
