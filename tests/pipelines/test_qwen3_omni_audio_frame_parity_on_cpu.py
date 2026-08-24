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
"""CPU parity test for Qwen3-Omni audio mel frames (actor vs rollout).

The HF feature extractor yields floor(L/hop) frames while vllm-omni pads the
waveform to a hop multiple first (ceil(L/hop)). ``pad_audio_to_hop_multiple``
in omni_rl_datasets restores parity; these tests pin both facts.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging.version import parse as parse_version

HOP = 160
SR = 16000

_REPO_ROOT = Path(__file__).parents[2]
_OMNI_RL_DATASETS_PATH = _REPO_ROOT / "verl_omni" / "utils" / "dataset" / "omni_rl_datasets.py"
_ADAPTER_PATH = _REPO_ROOT / "verl_omni" / "pipelines" / "qwen3_omni" / "thinker_training_adapter.py"


def _require_version(pkg_name: str, min_version: str):
    ver = importlib.metadata.version(pkg_name)
    assert parse_version(ver) >= parse_version(min_version), f"{pkg_name} >= {min_version} required, got {ver}"


def _audio_lengths() -> list[int]:
    """Synthetic 16kHz lengths covering L % hop in {0, 1, 79, 159}."""
    base = int(0.5 * SR)  # 0.5 s @ 16 kHz == 8000 samples (L % 160 == 0)
    return [base, base + 1, base + 79, base + 159]


@pytest.fixture(scope="module")
def feature_extractor():
    """Installed WhisperFeatureExtractor (padding=True, 16kHz, hop=160)."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    return WhisperFeatureExtractor(sampling_rate=SR, hop_length=HOP)


# ---------------------------------------------------------------------------
# Semantics helpers
# ---------------------------------------------------------------------------


def _actor_frames(fe, waveform: np.ndarray) -> int:
    """Actor-side frame count: WhisperFeatureExtractor attention_mask.sum()."""
    out = fe(waveform, sampling_rate=SR, padding=True, return_attention_mask=True)
    return int(out["attention_mask"].sum())


def _rollout_frames_after_pad(waveform: np.ndarray, hop: int = HOP) -> int:
    """Rollout-side frame count, replicating vllm-omni's pad_to_hop_length + num_frame."""
    x = np.asarray(waveform)
    length = x.shape[-1]
    if length % hop != 0:
        pad_length = hop - (length % hop)
        x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
    # After padding, length % hop == 0, so num_frame == length // hop.
    return x.shape[-1] // hop


def _audio_token_count(frames: int) -> int:
    """Audio token count via the shared _get_feat_extract_output_lengths formula."""
    pytest.importorskip("transformers")
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths

    return int(_get_feat_extract_output_lengths(torch.tensor([frames])).item())


# ---------------------------------------------------------------------------
# Isolated module loaders (avoid verl_omni/__init__.py GPU deps)
# ---------------------------------------------------------------------------


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
        convert_tokens_to_ids=lambda token: {"<|audio_pad|>": 103}.get(token, 0),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_divergence_without_hop_pad_documents_bug(feature_extractor):
    """(a) WITHOUT hop-padding, frames diverge for L%hop!=0, match for L%hop==0."""
    for L in _audio_lengths():
        waveform = np.zeros(L, dtype=np.float32)
        actor = _actor_frames(feature_extractor, waveform)
        rollout = _rollout_frames_after_pad(waveform)
        if L % HOP == 0:
            assert actor == rollout, f"L={L}: frames should match when L%hop==0"
        else:
            assert actor != rollout, f"L={L}: frames should diverge when L%hop!=0"
            assert abs(actor - rollout) == 1, f"L={L}: divergence should be exactly 1 (floor vs ceil)"


def test_frame_and_token_parity_with_hop_pad(feature_extractor, monkeypatch):
    """(b) WITH hop-padding, actor and rollout frames and token counts match everywhere."""
    omni = _load_omni_rl_datasets(monkeypatch)
    pad_audio = omni.pad_audio_to_hop_multiple

    for L in _audio_lengths():
        waveform = np.zeros(L, dtype=np.float32)
        padded = pad_audio(waveform, HOP)

        actor_frames = _actor_frames(feature_extractor, padded)
        rollout_frames = _rollout_frames_after_pad(padded)

        # Frames agree and both equal ceil(original L / hop).
        assert actor_frames == rollout_frames, f"L={L}: frames should match after hop-pad"
        assert actor_frames == (L + HOP - 1) // HOP, f"L={L}: frames should be ceil(L/hop)"

        # Token counts (audio-pad expansions) agree too.
        assert _audio_token_count(actor_frames) == _audio_token_count(rollout_frames)


def test_dedup_roundtrip_is_count_preserving(monkeypatch):
    """(c) dedup -> vllm-omni re-expand is count-preserving on hop-padded expanded ids."""
    processor = _build_processor_with_dedup(monkeypatch)
    audio_pad_id = int(processor.tokenizer.convert_tokens_to_ids(processor.audio_token))
    assert audio_pad_id == 103

    for frames in (7, 25, 50, 51, 100, 200):
        T = _audio_token_count(frames)
        assert T > 0

        expanded = [1, 2] + [audio_pad_id] * T + [3, 4]
        collapsed = processor.dedup_pad_tokens(expanded)

        collapsed_audio = sum(1 for t in collapsed if t == audio_pad_id)
        assert collapsed_audio == 1, f"frames={frames}: dedup should collapse the run to 1"
        assert [t for t in collapsed if t != audio_pad_id] == [1, 2, 3, 4]
