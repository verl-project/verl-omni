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
"""CPU behavior tests for the SQUIM no-reference audio quality scorer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# A deterministic non-silent waveform: pure silence short-circuits to the
# floor score before the (stubbed) model runs.
_WAVE = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)

pytest.importorskip("torchaudio")


def _load_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/squim.py"
    spec = importlib.util.spec_from_file_location("squim_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


squim = _load_module()


class _StubModel:
    """Returns fixed (STOI, PESQ, SI-SDR) predictions for any input."""

    def __init__(self, stoi: float, pesq: float, sisdr: float):
        self.stoi = stoi
        self.pesq = pesq
        self.sisdr = sisdr
        self.calls: list[tuple[torch.Tensor, ...]] = []

    def __call__(self, waveform: torch.Tensor):
        self.calls.append(tuple(waveform.shape))
        return (
            torch.tensor([self.stoi]),
            torch.tensor([self.pesq]),
            torch.tensor([self.sisdr]),
        )


@pytest.fixture()
def stub_model(monkeypatch):
    """Patch the model cache so tests never touch the network or a GPU."""

    def install(stoi: float, pesq: float, sisdr: float) -> _StubModel:
        model = _StubModel(stoi, pesq, sisdr)
        monkeypatch.setattr(squim, "_cached_model", lambda device: (model, None))
        return model

    return install


def test_extracts_mono_waveform_and_rate():
    waveform = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    tensor, rate = squim._extract_waveform((waveform, 16_000))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (64,)
    assert rate == 16_000


def test_extracts_channel_singleton_dimensions():
    tensor, rate = squim._extract_waveform((np.zeros((1, 1, 32), dtype=np.float32), 24_000))
    assert tensor.shape == (32,)
    assert rate == 24_000


def test_rejects_stereo_waveform():
    with pytest.raises(ValueError, match="mono waveform"):
        squim._extract_waveform((np.zeros((2, 64), dtype=np.float32), 16_000))


def test_rejects_malformed_solution_audio():
    with pytest.raises(TypeError):
        squim._extract_waveform("not-a-tuple")
    with pytest.raises(TypeError):
        squim._extract_waveform((np.zeros(32, dtype=np.float32),))


def test_rejects_short_nonfinite_or_bad_rate():
    with pytest.raises(ValueError):
        squim._extract_waveform((np.zeros(8, dtype=np.float32), 16_000))
    bad = np.zeros(64, dtype=np.float32)
    bad[7] = float("nan")
    with pytest.raises(ValueError):
        squim._extract_waveform((bad, 16_000))
    with pytest.raises(TypeError):
        squim._extract_waveform((np.zeros(64, dtype=np.float32), "16k"))
    with pytest.raises(ValueError):
        squim._extract_waveform((np.zeros(64, dtype=np.float32), 0))


def test_accepts_numpy_scalar_sample_rate():
    tensor, rate = squim._extract_waveform((np.zeros(64, dtype=np.float32), np.int64(16_000)))
    assert tensor.shape == (64,)
    assert rate == 16_000


def test_rejects_non_integer_numpy_sample_rate():
    with pytest.raises(TypeError):
        squim._extract_waveform((np.zeros(64, dtype=np.float32), np.float64(16_000.5)))


def test_normalizes_each_metric_over_its_range():
    pesq_low, pesq_high = squim._PESQ_RANGE  # wideband P.862.2 bounds from the bundle
    assert pesq_high > pesq_low > 0.0
    assert pesq_low == pytest.approx(1.0)
    assert pesq_high > 4.5
    assert squim._normalize_metric("stoi", 0.0) == 0.0
    assert squim._normalize_metric("stoi", 1.0) == 1.0
    assert squim._normalize_metric("pesq", pesq_low) == pytest.approx(0.0)
    assert squim._normalize_metric("pesq", pesq_high) == pytest.approx(1.0)
    assert squim._normalize_metric("sisdr", -20.0) == pytest.approx(0.0)
    assert squim._normalize_metric("sisdr", 30.0) == pytest.approx(1.0)
    assert squim._normalize_metric("stoi", 2.0) == 1.0  # clamped
    assert squim._normalize_metric("pesq", -10.0) == 0.0  # clamped


def test_compute_score_uniform_weights(stub_model):
    pesq_top = squim._PESQ_RANGE[1]
    stub_model(stoi=1.0, pesq=pesq_top, sisdr=30.0)
    result = squim.compute_score((_WAVE, 16_000))
    assert result["score"] == pytest.approx(1.0)
    assert result["squim_stoi"] == pytest.approx(1.0)
    assert result["squim_pesq"] == pytest.approx(pesq_top)
    assert result["squim_sisdr"] == pytest.approx(30.0)


def test_compute_score_averages_normalized_metrics(stub_model):
    stub_model(stoi=0.5, pesq=2.0, sisdr=5.0)
    result = squim.compute_score((_WAVE, 16_000))
    pesq_part = (2.0 - squim._PESQ_RANGE[0]) / (squim._PESQ_RANGE[1] - squim._PESQ_RANGE[0])
    expected = (0.5 + pesq_part + (5.0 + 20.0) / 50) / 3
    assert result["score"] == pytest.approx(expected)


def test_compute_score_resamples_to_16k(stub_model):
    model = stub_model(stoi=1.0, pesq=4.5, sisdr=30.0)
    squim.compute_score((np.linspace(-0.5, 0.5, 2400, dtype=np.float32), 24_000))
    assert model.calls, "the scorer must run the model"
    batch_shape = model.calls[0]
    assert batch_shape[0] == 1  # one mono waveform
    assert batch_shape[1] == pytest.approx(1600, rel=0.05)  # 2400 @ 24 kHz -> ~1600


def test_compute_score_custom_weights(stub_model):
    stub_model(stoi=0.0, pesq=4.5, sisdr=30.0)
    result = squim.compute_score(
        (_WAVE, 16_000),
        weights={"stoi": 1.0, "pesq": 0.0, "sisdr": 0.0},
    )
    assert result["score"] == pytest.approx(0.0)


def test_compute_score_rejects_bad_weights(stub_model):
    stub_model(stoi=1.0, pesq=1.0, sisdr=1.0)
    for weights in (
        {"stoi": 1.0},
        {"stoi": 1.0, "pesq": 1.0, "sisdr": 1.0, "extra": 1.0},
        {"stoi": -1.0, "pesq": 1.0, "sisdr": 1.0},
        {"stoi": 0.0, "pesq": 0.0, "sisdr": 0.0},
    ):
        with pytest.raises(ValueError):
            squim.compute_score((_WAVE, 16_000), weights=weights)


def test_compute_score_absorbs_forwarded_kwargs(stub_model):
    stub_model(stoi=1.0, pesq=4.5, sisdr=30.0)
    result = squim.compute_score(
        (_WAVE, 16_000),
        data_source="tts",
        ground_truth="unused",
        extra_info={"num_turns": 2},
        device="cpu",
        some_unrelated_forwarded_key="ignored",
    )
    assert 0.0 <= result["score"] <= 1.0


def test_silent_audio_returns_floor_score_without_model():
    result = squim.compute_score((np.zeros(1600, dtype=np.float32), 16_000))
    assert result["score"] == 0.0
    assert result["squim_stoi"] == squim._STOI_RANGE[0]
    assert result["squim_pesq"] == squim._PESQ_RANGE[0]
    assert result["squim_sisdr"] == squim._SISDR_RANGE_DB[0]


def test_too_short_after_resampling_rejected(stub_model):
    stub_model(stoi=1.0, pesq=1.0, sisdr=1.0)
    # 64 samples at 48 kHz pass the 16-sample pre-check but resample to ~21
    # samples, below the model's 64-sample window.
    with pytest.raises(ValueError, match="too short"):
        squim.compute_score((np.linspace(-1, 1, 64, dtype=np.float32), 48_000))


def test_weight_sum_overflow_rejected(stub_model):
    stub_model(stoi=1.0, pesq=1.0, sisdr=1.0)
    huge = 1e308
    with pytest.raises(ValueError, match="finite positive"):
        squim.compute_score(
            (_WAVE, 16_000),
            weights={"stoi": huge, "pesq": huge, "sisdr": huge},
        )
