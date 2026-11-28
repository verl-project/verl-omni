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
"""SQUIM no-reference audio quality reward.

Scores a generated waveform with torchaudio's SQUIM objective predictor
("Rethinking Audio Quality Assessment for Text-to-Speech", Dey et al.,
IEEE Signal Processing Letters 2024): a wav2vec2-based model that predicts
three no-reference quality impairments — STOI, PESQ and SI-SDR — from the
audio alone (the subjective variant needs a clean reference, which reward
time does not have).

Each metric is mapped into [0, 1] over its standard range (the PESQ range
follows the bundle's wideband P.862.2 bounds) and the score is the weighted
mean (uniform by default). Silent audio never reaches the model: it returns
the floors (score 0.0) because the model's RMS normalization is undefined
on silence:

    STOI   [0, 1]                  -> STOI
    PESQ   wideband [1.0, ~4.64]   -> (PESQ - 1.0) / (upper - 1.0)
    SI-SDR [-20, 30] dB           -> (SI-SDR + 20) / 50

The raw predictions ride along as ``squim_stoi`` / ``squim_pesq`` /
``squim_sisdr`` so the reward stays interpretable. The model is a torchaudio
bundle (28 MB) downloaded on first use into the torch cache; set ``TORCH_HOME``
on air-gapped clusters to pin the cache directory.

Configure from a recipe through the scorer keyword contract:

    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.squim
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.device=cpu
"""

from __future__ import annotations

import math
import numbers
import threading

import torch
import torchaudio
from torchaudio.models.squim.objective import PESQRange

__all__ = ["compute_score"]

_TARGET_SAMPLE_RATE = 16_000
_MIN_WAVEFORM_SAMPLES = 16  # rejects garbage before resampling
_MODEL_MIN_SAMPLES = 64  # the bundle's convolution window

_STOI_RANGE = (0.0, 1.0)
# SQUIM predicts wideband PESQ (ITU-T P.862.2), whose range differs from the
# narrowband [-0.5, 4.5]; read it from the bundle instead of hardcoding it.
_PESQ_RANGE = (float(PESQRange[0]), float(PESQRange[1]))
_SISDR_RANGE_DB = (-20.0, 30.0)

_METRIC_RANGES = {
    "stoi": _STOI_RANGE,
    "pesq": _PESQ_RANGE,
    "sisdr": _SISDR_RANGE_DB,
}

_MODEL_CACHE: dict[str, tuple] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model(device: str):
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    model = SQUIM_OBJECTIVE.get_model()
    model = model.to(device=device).eval()
    return model, SQUIM_OBJECTIVE


def _cached_model(device: str):
    key = str(device)
    with _MODEL_CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = _load_model(device)
        return _MODEL_CACHE[key]


def _normalize_metric(name: str, value: float) -> float:
    low, high = _METRIC_RANGES[name]
    return min(max((value - low) / (high - low), 0.0), 1.0)


def _extract_waveform(solution_audio) -> tuple[torch.Tensor, int]:
    if not isinstance(solution_audio, tuple | list) or len(solution_audio) != 2:
        raise TypeError("SQUIM scorer expects solution_audio=(waveform, sample_rate).")
    waveform, sample_rate = solution_audio
    tensor = torch.as_tensor(waveform).detach().float().cpu()
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(
            f"SQUIM scorer expects a mono waveform with shape (T,) or leading singleton dimensions, "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.numel() < _MIN_WAVEFORM_SAMPLES:
        raise ValueError(f"SQUIM scorer received too short a waveform ({tensor.numel()} samples).")
    if not torch.isfinite(tensor).all():
        raise ValueError("SQUIM scorer received a waveform containing NaN or infinity.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, numbers.Real):
        raise TypeError(f"SQUIM scorer sample rate must be an integer, got {sample_rate!r}.")
    if int(sample_rate) != sample_rate:
        raise TypeError(f"SQUIM scorer sample rate must be an integer, got {sample_rate!r}.")
    if int(sample_rate) <= 0:
        raise ValueError(f"SQUIM scorer sample rate must be positive, got {sample_rate!r}.")
    return tensor, int(sample_rate)


def compute_score(
    solution_audio: tuple | list,
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    data_source: str | None = None,
    *,
    device: str = "cpu",
    weights: dict[str, float] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Score one generated waveform with the SQUIM objective predictor.

    Args:
        solution_audio: ``(waveform, sample_rate)`` from the audio reward
            manager — a mono float waveform and its integer sample rate.
        ground_truth: Unused, part of the scorer contract.
        extra_info: Manager-injected metadata (unused here).
        data_source: The dataset source (unused, part of the contract).
        device: Inference device for the SQUIM model ("cpu", "cuda:0", ...).
        weights: Per-metric weights over {"stoi", "pesq", "sisdr"}; the
            normalized metrics are averaged with these weights (uniform when
            unset). Unknown keys are rejected.
        **kwargs: Absorbs forwarded config keys the scorer does not use.

    Returns:
        ``{"score": float}`` in [0, 1] plus the raw predictions as
        ``squim_stoi`` / ``squim_pesq`` / ``squim_sisdr``.
    """
    del ground_truth, extra_info, data_source, kwargs
    if weights is None:
        weights = {name: 1.0 for name in _METRIC_RANGES}
    if set(weights) != set(_METRIC_RANGES):
        raise ValueError(f"SQUIM weights keys must be exactly {sorted(_METRIC_RANGES)}, got {sorted(weights)}.")
    for name, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, int | float) or not math.isfinite(float(weight)):
            raise ValueError(f"SQUIM weight for {name!r} must be a finite number, got {weight!r}.")
        if float(weight) < 0.0:
            raise ValueError(f"SQUIM weight for {name!r} must be non-negative, got {weight!r}.")
    total = sum(float(weight) for weight in weights.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"SQUIM weights must sum to a finite positive value, got {total}.")

    waveform, sample_rate = _extract_waveform(solution_audio)
    if sample_rate != _TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, _TARGET_SAMPLE_RATE)
    # The model's convolutions use a 64-sample window; shorter inputs (which
    # resampling can also produce) fail deep inside the bundle.
    if waveform.numel() < _MODEL_MIN_SAMPLES:
        raise ValueError(f"SQUIM scorer received too short a waveform ({waveform.numel()} samples at 16 kHz).")
    # The model's RMS normalization divides by the waveform energy, so pure
    # silence is undefined; score it as the range floors instead of crashing.
    if not waveform.any():
        return {
            "score": 0.0,
            "squim_stoi": _STOI_RANGE[0],
            "squim_pesq": _PESQ_RANGE[0],
            "squim_sisdr": _SISDR_RANGE_DB[0],
        }

    model, _ = _cached_model(device)
    with torch.inference_mode(), _MODEL_CACHE_LOCK:
        stoi, pesq, sisdr = model(waveform.unsqueeze(0).to(device))
    values = {
        "stoi": float(stoi.reshape(-1)[0]),
        "pesq": float(pesq.reshape(-1)[0]),
        "sisdr": float(sisdr.reshape(-1)[0]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"SQUIM produced non-finite predictions: {values}.")
    score = sum(_normalize_metric(name, value) * float(weights[name]) for name, value in values.items()) / total
    return {
        "score": score,
        "squim_stoi": values["stoi"],
        "squim_pesq": values["pesq"],
        "squim_sisdr": values["sisdr"],
    }
