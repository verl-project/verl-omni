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
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging.version import parse as parse_version

from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter
from verl_omni.utils.dataset.omni_rl_datasets import DEFAULT_AUDIO_HOP_LENGTH, pad_audio_to_hop_multiple

HOP = 160
SR = 16000

# Qwen3-Omni ``<|audio_pad|>`` token id — used by the stubbed tokenizer below
# and asserted on in the dedup round-trip test.
AUDIO_PAD_ID = 103


def _require_version(pkg_name: str, min_version: str):
    ver = importlib.metadata.version(pkg_name)
    assert parse_version(ver) >= parse_version(min_version), f"{pkg_name} >= {min_version} required, got {ver}"


@pytest.fixture(scope="module")
def feature_extractor():
    """Installed WhisperFeatureExtractor (padding=True, 16kHz, hop=160)."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    return WhisperFeatureExtractor(sampling_rate=SR, hop_length=HOP)


@pytest.fixture
def processor_with_dedup(monkeypatch):
    """Run the real adapter's configure_processor on stubs; returns a processor with dedup_pad_tokens bound."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

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

    configured = Qwen3OmniThinkerAdapter.configure_processor(
        "/fake/qwen3-omni",
        SimpleNamespace(trust_remote_code=False),
    )
    assert hasattr(configured, "dedup_pad_tokens")
    return configured


def _audio_lengths() -> list[int]:
    """Synthetic 16kHz lengths covering L % hop in {0, 1, 79, 159}."""
    base = int(0.5 * SR)  # 0.5 s @ 16 kHz == 8000 samples (L % 160 == 0)
    return [base, base + 1, base + 79, base + 159]


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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", _audio_lengths())
def test_unpadded_audio_frame_counts_diverge_from_rollout(feature_extractor, L):
    """(a) WITHOUT hop-padding, frames diverge for L%hop!=0, match for L%hop==0."""
    waveform = np.zeros(L, dtype=np.float32)
    actor = _actor_frames(feature_extractor, waveform)
    rollout = _rollout_frames_after_pad(waveform)
    if L % HOP == 0:
        assert actor == rollout, f"L={L}: frames should match when L%hop==0"
    else:
        assert actor != rollout, f"L={L}: frames should diverge when L%hop!=0"
        assert abs(actor - rollout) == 1, f"L={L}: divergence should be exactly 1 (floor vs ceil)"


@pytest.mark.parametrize("L", _audio_lengths())
def test_hop_pad_restores_actor_rollout_frame_parity(feature_extractor, L):
    """(b) WITH hop-padding, actor and rollout frame counts match everywhere.

    ``DEFAULT_AUDIO_HOP_LENGTH`` is pinned to the local ``HOP`` here so the
    keep-in-sync note in omni_rl_datasets.py stays enforced by this test.
    """
    assert DEFAULT_AUDIO_HOP_LENGTH == HOP

    waveform = np.zeros(L, dtype=np.float32)
    padded = pad_audio_to_hop_multiple(waveform, HOP)

    actor_frames = _actor_frames(feature_extractor, padded)
    rollout_frames = _rollout_frames_after_pad(padded)

    # Frames agree and both equal ceil(original L / hop).
    assert actor_frames == rollout_frames, f"L={L}: frames should match after hop-pad"
    assert actor_frames == (L + HOP - 1) // HOP, f"L={L}: frames should be ceil(L/hop)"


@pytest.mark.parametrize("frames", [7, 25, 50, 51, 100, 200])
def test_dedup_roundtrip_is_count_preserving(processor_with_dedup, frames):
    """(c) dedup -> vllm-omni re-expand is count-preserving on hop-padded expanded ids."""
    processor = processor_with_dedup
    audio_pad = int(processor.tokenizer.convert_tokens_to_ids(processor.audio_token))
    assert audio_pad == AUDIO_PAD_ID

    T = _audio_token_count(frames)
    assert T > 0

    expanded = [1, 2] + [audio_pad] * T + [3, 4]
    collapsed = processor.dedup_pad_tokens(expanded)

    collapsed_audio = sum(1 for t in collapsed if t == audio_pad)
    assert collapsed_audio == 1, f"frames={frames}: dedup should collapse the run to 1"
    assert [t for t in collapsed if t != audio_pad] == [1, 2, 3, 4]
