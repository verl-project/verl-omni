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
"""Shared fixtures for Qwen3-Omni pipeline CPU tests.

The isolated module loaders here avoid triggering the ``verl_omni`` package
init (which pulls in GPU deps); see the individual fixture docstrings.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import parse as parse_version

# Qwen3-Omni ``<|audio_pad|>`` token id — single source for the stubbed
# tokenizer below and the parity/dedup tests that assert on it.
AUDIO_PAD_ID = 103

_REPO_ROOT = Path(__file__).parents[2]
_OMNI_RL_DATASETS_PATH = _REPO_ROOT / "verl_omni" / "utils" / "dataset" / "omni_rl_datasets.py"
_ADAPTER_PATH = _REPO_ROOT / "verl_omni" / "pipelines" / "qwen3_omni" / "thinker_training_adapter.py"


def _require_version(pkg_name: str, min_version: str):
    ver = importlib.metadata.version(pkg_name)
    assert parse_version(ver) >= parse_version(min_version), f"{pkg_name} >= {min_version} required, got {ver}"


@pytest.fixture(scope="module")
def require_version():
    """Fixture returning the ``_require_version(pkg_name, min_version)`` callable."""
    return _require_version


def _load_omni_rl_datasets(monkeypatch):
    """Load omni_rl_datasets.py without triggering the verl_omni package init."""
    for pkg in ("verl_omni", "verl_omni.utils", "verl_omni.utils.dataset"):
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, mod)
    module_name = "verl_omni.utils.dataset.omni_rl_datasets"
    spec = importlib.util.spec_from_file_location(module_name, _OMNI_RL_DATASETS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def omni_rl_datasets(monkeypatch):
    """Load omni_rl_datasets.py without triggering the verl_omni package init."""
    return _load_omni_rl_datasets(monkeypatch)


def _build_processor_with_dedup(monkeypatch):
    """Run the real adapter's configure_processor on stubs; returns a processor with dedup_pad_tokens bound."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

    class _FakeOmniModelBase:
        @classmethod
        def register(cls, *args, **kwargs):
            return lambda subclass: subclass

    model_base = types.ModuleType("verl_omni.pipelines.model_base")
    model_base.OmniModelBase = _FakeOmniModelBase
    for package_name in ("verl_omni", "verl_omni.pipelines", "verl_omni.pipelines.qwen3_omni"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "verl_omni.pipelines.model_base", model_base)

    module_name = "verl_omni.pipelines.qwen3_omni.thinker_training_adapter"
    spec = importlib.util.spec_from_file_location(module_name, _ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    adapter_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, adapter_module)
    spec.loader.exec_module(adapter_module)

    class AgentLoopWorker:
        def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs=None):
            del input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs

    for package_name in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    agent_loop_tq_module = types.ModuleType("verl.trainer.ppo.v1.agent_loop_tq")
    agent_loop_tq_module.AgentLoopWorkerTQ = SimpleNamespace(
        __ray_metadata__=SimpleNamespace(modified_class=AgentLoopWorker)
    )
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.agent_loop_tq", agent_loop_tq_module)

    tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {"<|audio_pad|>": AUDIO_PAD_ID}.get(token, 0),
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        audio_token="<|audio_pad|>",
    )
    config = SimpleNamespace(
        thinker_config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
        talker_config=SimpleNamespace(vision_start_token_id=104),
    )
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    # configure_processor binds get_rope_index/get_llm_pos_ids_for_vision via
    # MethodType on real model class methods; ensure they exist.
    assert hasattr(Qwen3OmniMoeThinkerForConditionalGeneration, "get_rope_index")

    configured = adapter_module.Qwen3OmniThinkerAdapter.configure_processor(
        "/fake/qwen3-omni",
        SimpleNamespace(trust_remote_code=False),
    )
    assert hasattr(configured, "dedup_pad_tokens")
    return configured


@pytest.fixture
def processor_with_dedup(monkeypatch):
    """Run the real adapter's configure_processor on stubs; returns a processor with dedup_pad_tokens bound."""
    return _build_processor_with_dedup(monkeypatch)


@pytest.fixture
def audio_pad_id():
    """Qwen3-Omni ``<|audio_pad|>`` token id (single source defined in conftest)."""
    return AUDIO_PAD_ID
