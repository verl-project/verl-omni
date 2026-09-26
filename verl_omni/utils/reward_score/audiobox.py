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

"""AudioBox Aesthetics reward adapted from zghhui/OmniNFT."""

import threading
from typing import Any

import torch
import torch.nn.functional as F

from .clap import _get_audio

_AUDIOBOX_SAMPLE_RATE = 16_000
_AUDIOBOX_WINDOW_SAMPLES = 10 * _AUDIOBOX_SAMPLE_RATE
_AUDIOBOX_HOP_SAMPLES = 10 * _AUDIOBOX_SAMPLE_RATE
_AXES = ("CE", "CU", "PC", "PQ")


def _load_model(model_path: str) -> Any:
    from audiobox_aesthetics.model.aes import AesMultiOutput

    return AesMultiOutput.from_pretrained(model_path).eval()


def _resample_audio(waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
    if source_rate == _AUDIOBOX_SAMPLE_RATE:
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(
        waveform.unsqueeze(0),
        orig_freq=source_rate,
        new_freq=_AUDIOBOX_SAMPLE_RATE,
    ).squeeze(0)


def _make_windows(waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int], list[float]]:
    """Expand waveforms into nonoverlapping 10 s windows, padding the last one.

    Return CPU ``[W, 1, 160000]`` audio and bool validity masks, local sample
    indices, and valid-duration fractions for per-sample weighted averaging.
    """
    windows = []
    masks = []
    sample_indices = []
    weights = []
    for sample_index, waveform in enumerate(waveforms):
        for start in range(0, waveform.numel(), _AUDIOBOX_HOP_SAMPLES):
            window = waveform[start : start + _AUDIOBOX_WINDOW_SAMPLES]
            valid_length = window.numel()
            if valid_length < _AUDIOBOX_WINDOW_SAMPLES:
                window = F.pad(window, (0, _AUDIOBOX_WINDOW_SAMPLES - valid_length))
            mask = torch.zeros(_AUDIOBOX_WINDOW_SAMPLES, dtype=torch.bool)
            mask[:valid_length] = True
            windows.append(window.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            sample_indices.append(sample_index)
            weights.append(valid_length / _AUDIOBOX_WINDOW_SAMPLES)
    return torch.stack(windows), torch.stack(masks), sample_indices, weights


def _validate_predictions(predictions: Any, window_count: int) -> dict[str, torch.Tensor]:
    if not isinstance(predictions, dict):
        raise ValueError("AudioBox model output must be a dict of axis tensors.")
    validated = {}
    for axis in _AXES:
        values = predictions.get(axis)
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != (window_count,)
            or not values.dtype.is_floating_point
        ):
            raise ValueError(f"AudioBox {axis} output must be a floating-point tensor with shape ({window_count},).")
        if not torch.isfinite(values).all():
            raise ValueError(f"AudioBox {axis} output must contain only finite values.")
        validated[axis] = values
    return validated


def _score_windows(output: dict[str, Any], window_count: int, axis_weights, score_scale: float) -> torch.Tensor:
    """Undo each axis's target transform and return CPU FP32 window rewards."""
    predictions = _validate_predictions(output["predictions"], window_count)
    target_transform = output["target_transform"]
    restored = {}
    for axis in _AXES:
        mean, std = target_transform[axis]
        values = predictions[axis].float().mul(std).add(mean).detach().cpu()
        restored[axis] = values
    return sum(restored[axis] * weight for axis, weight in axis_weights.items()) * score_scale


async def compute_score(
    data_source=None,
    solution_image=None,
    ground_truth=None,
    extra_info=None,
    *,
    reward_model,
    batch=None,
    axis_weights=None,
    score_scale: float = 0.025,
    **kwargs,
) -> dict[str, float]:
    """Score audio aesthetics using duration-weighted windows and configurable axes.

    Read audio/rate from extra_info or a single-sample batch. Restore the model's
    target transforms before combining CE/CU/PC/PQ; inference belongs to the executor.
    """
    del data_source, solution_image, ground_truth, kwargs
    extra_info = dict(extra_info or {})
    if batch is not None:
        if len(batch) != 1:
            raise ValueError("AudioBox scoring requires exactly one sample.")
        item = batch[0]
        for key in ("audio", "audio_sample_rate"):
            if key in item.batch:
                extra_info[key] = item.batch[key]
            elif key in item.non_tensor_batch:
                extra_info[key] = item.non_tensor_batch[key]
    waveform, source_rate = _get_audio(extra_info)
    windows, masks, _, weights = _make_windows([_resample_audio(waveform, source_rate)])
    output = await reward_model.infer(windows, masks)
    if axis_weights is None:
        axis_weights = {"CE": 1.0, "CU": 1.0, "PC": -1.0, "PQ": 1.0}
    local_scores = _score_windows(output, windows.shape[0], axis_weights, score_scale)
    window_weights = torch.tensor(weights, dtype=torch.float32)
    score = (local_scores * window_weights).sum() / window_weights.sum()
    if not torch.isfinite(score):
        raise ValueError("AudioBox score must be finite.")
    return {"score": float(score)}


class AudioBoxModel:
    """Raw AudioBox inference adapter owned by a native reward executor."""

    def __init__(self, model_path: str, device) -> None:
        self.model = _load_model(model_path)
        self.target_transform = {
            axis: (float(self.model.target_transform[axis]["mean"]), float(self.model.target_transform[axis]["std"]))
            for axis in _AXES
        }
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self._infer_lock = threading.Lock()

    def close(self) -> None:
        """Drop model/transform references; further inference requires a new adapter."""
        self.model = None
        self.target_transform = {}
        self.device = None

    @torch.inference_mode()
    def infer(self, windows: torch.Tensor, masks: torch.Tensor) -> dict[str, Any]:
        """Infer prepared ``[W, 1, 160000]`` audio and bool masks without gradients.

        Move inputs to the active device without changing dtype. Return detached
        CPU ``[W]`` predictions for CE/CU/PC/PQ in model-output dtype, plus the
        retained target-transform mapping; calibration belongs to the scorer.
        """
        with self._infer_lock:
            inputs = {"wav": windows.to(self.device), "mask": masks.to(self.device)}
            predictions = self.model(inputs)
            return {
                "predictions": {name: predictions[name].detach().cpu() for name in _AXES},
                "target_transform": self.target_transform,
            }
