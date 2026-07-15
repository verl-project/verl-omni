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

"""Audio-video semantic alignment reward for LTX-2.3 using ImageBind."""

import threading
import warnings

import torch
import torch.nn.functional as F

_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_NUM_MEL_BINS = 128
_AUDIO_TARGET_LENGTH = 204
_AUDIO_CLIP_SAMPLES = 2 * _AUDIO_SAMPLE_RATE
_AUDIO_CLIPS = 3
_AUDIO_MEAN = -4.268
_AUDIO_STD = 9.138
_VISION_SIZE = 224
_VISION_MEAN = (0.48145466, 0.4578275, 0.40821073)
_VISION_STD = (0.26862954, 0.26130258, 0.27577711)
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()


def _load_imagebind(device: str):
    if device not in _MODEL_CACHE:
        try:
            from imagebind.models import imagebind_model
        except ImportError as exc:
            raise ImportError(
                "ImageBind reward requires `pip install git+https://github.com/facebookresearch/ImageBind.git` "
                "and is licensed CC-BY-NC-SA 4.0 for non-commercial use."
            ) from exc
        warnings.warn(
            "ImageBind is licensed CC-BY-NC-SA 4.0 (NonCommercial).",
            stacklevel=2,
        )
        _MODEL_CACHE[device] = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
    return _MODEL_CACHE[device]


def _normalize_audio(audio, source_rate: int) -> torch.Tensor:
    import torchaudio.functional as audio_functional

    waveform = torch.as_tensor(audio).detach().float().cpu()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.ndim != 2:
        raise ValueError(f"Expected audio shape (T,) or (C,T), got {tuple(waveform.shape)}.")
    if source_rate != _AUDIO_SAMPLE_RATE:
        waveform = audio_functional.resample(waveform, source_rate, _AUDIO_SAMPLE_RATE)
    return waveform


def _waveform_to_melspec(waveform: torch.Tensor) -> torch.Tensor:
    import torchaudio.compliance.kaldi as kaldi

    waveform = waveform.float() - waveform.float().mean()
    fbank = kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=_AUDIO_SAMPLE_RATE,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=_AUDIO_NUM_MEL_BINS,
        dither=0.0,
        frame_length=25,
        frame_shift=10,
    ).transpose(0, 1)
    if fbank.shape[1] < _AUDIO_TARGET_LENGTH:
        fbank = F.pad(fbank, (0, _AUDIO_TARGET_LENGTH - fbank.shape[1]))
    else:
        fbank = fbank[:, :_AUDIO_TARGET_LENGTH]
    return fbank.unsqueeze(0)


def _preprocess_audio(audio, source_rate: int, device: str) -> torch.Tensor:
    waveform = _normalize_audio(audio, source_rate)
    duration = waveform.shape[1] / _AUDIO_SAMPLE_RATE
    clip_duration = _AUDIO_CLIP_SAMPLES / _AUDIO_SAMPLE_RATE
    spacing = max(duration - clip_duration, 0.0) / max(_AUDIO_CLIPS - 1, 1)
    clips = []
    for index in range(_AUDIO_CLIPS):
        start = int(index * spacing * _AUDIO_SAMPLE_RATE)
        clip = waveform[:, start : start + _AUDIO_CLIP_SAMPLES]
        if clip.shape[1] < _AUDIO_CLIP_SAMPLES:
            clip = F.pad(clip, (0, _AUDIO_CLIP_SAMPLES - clip.shape[1]))
        mel = (_waveform_to_melspec(clip) - _AUDIO_MEAN) / _AUDIO_STD
        clips.append(mel)
    return torch.stack(clips).unsqueeze(0).to(device)


def _to_tchw(video) -> torch.Tensor:
    video = torch.as_tensor(video).detach().float().cpu()
    while video.ndim > 4 and video.shape[0] == 1:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected a four-dimensional video, got {tuple(video.shape)}.")
    if video.shape[1] in (1, 3):
        pass
    elif video.shape[-1] in (1, 3):
        video = video.permute(0, 3, 1, 2)
    elif video.shape[0] in (1, 3):
        video = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Could not infer video channel dimension from {tuple(video.shape)}.")
    if video.max() > 1.0:
        video = video / 255.0
    return video


def _preprocess_video(video, device: str) -> torch.Tensor:
    video = _to_tchw(video)
    frame_count, channels, height, width = video.shape
    clips = []
    for index in range(5):
        center = int((index + 0.5) * frame_count / 5)
        indices = torch.linspace(max(0, center - 1), min(frame_count - 1, center), 2).long()
        clip = video[indices].permute(1, 0, 2, 3)
        if width <= height:
            resized_width, resized_height = _VISION_SIZE, int(height / width * _VISION_SIZE)
        else:
            resized_width, resized_height = int(width / height * _VISION_SIZE), _VISION_SIZE
        clip = F.interpolate(
            clip.reshape(channels * 2, 1, height, width),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(channels, 2, resized_height, resized_width)
        mean = torch.tensor(_VISION_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(_VISION_STD).view(3, 1, 1, 1)
        clip = (clip - mean) / std
        if resized_height > resized_width:
            offsets = [0, (resized_height - _VISION_SIZE) // 2, resized_height - _VISION_SIZE]
            clips.extend(clip[:, :, offset : offset + _VISION_SIZE, :] for offset in offsets)
        else:
            offsets = [0, (resized_width - _VISION_SIZE) // 2, resized_width - _VISION_SIZE]
            clips.extend(clip[:, :, :, offset : offset + _VISION_SIZE] for offset in offsets)
    return torch.stack(clips).unsqueeze(0).to(device)


def compute_score_imagebind_audio_video(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str = "cuda",
    **kwargs,
) -> dict:
    """Compute ImageBind cosine similarity between generated audio and video."""
    del data_source, ground_truth, kwargs
    try:
        from imagebind.models.imagebind_model import ModalityType
    except ImportError as exc:
        raise ImportError("ImageBind reward requires the non-commercial ImageBind package.") from exc

    audio = extra_info.get("audio")
    if audio is None:
        raise KeyError("ImageBind reward requires decoded audio in extra_info['audio'].")
    sample_rate = extra_info.get("audio_sample_rate")
    if isinstance(sample_rate, torch.Tensor):
        sample_rate = sample_rate.item()
    if sample_rate is None:
        raise KeyError("ImageBind reward requires extra_info['audio_sample_rate'].")

    with _MODEL_LOCK, torch.no_grad():
        model = _load_imagebind(device)
        embeddings = model(
            {
                ModalityType.AUDIO: _preprocess_audio(audio, int(sample_rate), device),
                ModalityType.VISION: _preprocess_video(solution_image, device),
            }
        )
        audio_embedding = F.normalize(embeddings[ModalityType.AUDIO], dim=-1)
        video_embedding = F.normalize(embeddings[ModalityType.VISION], dim=-1)
        score = (audio_embedding * video_embedding).sum(dim=-1)[0].float().item()
    return {"score": score}
