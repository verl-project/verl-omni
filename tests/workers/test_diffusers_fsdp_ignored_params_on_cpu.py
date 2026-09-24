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
"""CPU tests for adapter-declared FSDP2 ignored subtrees in the diffusers engine."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


class _FrozenTowerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = torch.nn.Linear(4, 4)
        # Ignored subtrees must be frozen: FSDP2 never syncs their gradients.
        self.tower.requires_grad_(False)
        self.llm = torch.nn.Linear(4, 4)
        self._no_split_modules = ["DecoderLayer"]


def _engine(strategy="fsdp2"):
    """A bare engine; the adapter class is injected by patching ``DiffusionModelBase.get_class``."""
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.device_mesh = None
    engine.model_config = SimpleNamespace(lora_rank=0)
    engine.engine_config = SimpleNamespace(
        strategy=strategy,
        mixed_precision=None,
        offload_policy=False,
        forward_only=False,
        reshard_after_forward=True,
        forward_prefetch=False,
        use_orig_params=True,
        wrap_policy={},
        get=lambda key, default=None: {},
    )
    return engine


def _adapter_cls(ignored_names):
    adapter_cls = MagicMock()
    adapter_cls.preserve_fp32_modules.return_value = True
    adapter_cls.get_fsdp_ignored_module_names.return_value = ignored_names or []
    return adapter_cls


def test_build_fsdp_module_threads_adapter_ignored_names_into_fsdp2(monkeypatch):
    module = _FrozenTowerModule()
    captured = {}

    def fake_apply_fsdp2(model, fsdp_kwargs, config, ignored_names=()):
        captured["ignored_names"] = list(ignored_names)

    monkeypatch.setattr(
        diffusers_impl.DiffusionModelBase, "get_class", staticmethod(lambda cfg: _adapter_cls(["tower"]))
    )
    monkeypatch.setattr(diffusers_impl, "apply_fsdp2", fake_apply_fsdp2)
    monkeypatch.setattr(diffusers_impl, "get_fsdp_wrap_policy", lambda **kwargs: None)
    monkeypatch.setattr(diffusers_impl, "get_sharding_strategy", lambda mesh: None)
    monkeypatch.setattr(diffusers_impl, "fsdp2_load_full_state_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    engine = _engine()
    assert engine._build_fsdp_module(module) is module
    assert captured["ignored_names"] == ["tower"]

    # An adapter that does not opt in keeps the empty default.
    monkeypatch.setattr(diffusers_impl.DiffusionModelBase, "get_class", staticmethod(lambda cfg: _adapter_cls(None)))
    engine = _engine()
    engine._build_fsdp_module(module)
    assert captured["ignored_names"] == []


def test_build_fsdp_module_fsdp1_rejects_ignored_names(monkeypatch):
    monkeypatch.setattr(
        diffusers_impl.DiffusionModelBase, "get_class", staticmethod(lambda cfg: _adapter_cls(["tower"]))
    )
    engine = _engine(strategy="fsdp")
    with pytest.raises(NotImplementedError, match="strategy=fsdp2"):
        engine._build_fsdp_module(_FrozenTowerModule())
