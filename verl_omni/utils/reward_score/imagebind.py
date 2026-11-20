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

"""Cross-modal semantic alignment rewards using Meta ImageBind.

Supported modes match FlowFactory's ImageBind reward:
``audio_video``, ``text_audio``, ``text_video``, and their weighted ``all``
combination.
"""

import os
import threading
import warnings
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from verl.utils.device import get_device_name

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
_DEFAULT_MODEL_PATH = ".checkpoints/imagebind_huge.pth"
_DEFAULT_MODE = "audio_video"
_PAIR_MODES = ("audio_video", "text_audio", "text_video")
_SUPPORTED_MODES = (*_PAIR_MODES, "all")
_DEFAULT_WEIGHTS = {
    "audio_video": 0.5,
    "text_audio": 0.25,
    "text_video": 0.25,
}


def _create_imagebind(device, model_path: str):
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

    model = imagebind_model.imagebind_huge(pretrained=False)
    if not os.path.exists(model_path):
        print(f"Downloading ImageBind weights to {model_path} ...")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
            model_path,
            progress=True,
        )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(device).eval()


def _load_imagebind(device: str, model_path: str):
    key = (model_path, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _create_imagebind(device, model_path)
    return _MODEL_CACHE[key]


def _normalize_audio(audio, source_rate: int) -> torch.Tensor:
    import torchaudio.functional as audio_functional

    waveform = torch.as_tensor(audio).detach().float().cpu()
    if waveform.ndim != 2:
        raise ValueError(f"Expected canonical CT audio, got {tuple(waveform.shape)}.")
    waveform = waveform.mean(dim=0, keepdim=True)
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
    from verl_omni.pipelines.rollout_media import MediaArtifact

    if isinstance(video, MediaArtifact):
        if (video.spec.modality, video.spec.representation) != ("video", "decoded"):
            raise ValueError(f"ImageBind requires decoded video, got {video.spec}")
        video.validate(context="ImageBind", name="video")
        if video.spec.layout != "TCHW":
            raise ValueError(f"ImageBind requires canonical TCHW, got {video.spec.layout}")
        video = video.data
    from verl_omni.utils.reward_score.reward_utils import normalize_video_tensor

    return normalize_video_tensor(video).detach().float().cpu() / 255.0


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


def _preprocess_text(text: str, device: str) -> torch.Tensor:
    try:
        from imagebind.data import load_and_transform_text
    except ImportError as exc:
        raise ImportError("ImageBind text rewards require imagebind.data.load_and_transform_text.") from exc
    return load_and_transform_text([text], device)


def _cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    return (first * second).sum(dim=-1)[0].float().item()


def _compute_similarities(embeddings: dict) -> dict[str, float]:
    from imagebind.models.imagebind_model import ModalityType

    modality_pairs = {
        "audio_video": (ModalityType.AUDIO, ModalityType.VISION),
        "text_audio": (ModalityType.TEXT, ModalityType.AUDIO),
        "text_video": (ModalityType.TEXT, ModalityType.VISION),
    }
    return {
        name: _cosine_similarity(embeddings[first], embeddings[second])
        for name, (first, second) in modality_pairs.items()
        if first in embeddings and second in embeddings
    }


def _aggregate_similarities(similarities: dict[str, float], weights: Mapping[str, float] | None) -> float:
    selected_weights = _DEFAULT_WEIGHTS if weights is None else weights
    missing = set(_PAIR_MODES) - set(selected_weights)
    if missing:
        raise ValueError(f"ImageBind weights are missing modes: {sorted(missing)}.")
    return sum(float(selected_weights[name]) * similarities[name] for name in _PAIR_MODES)


def _select_media(ground_truth: str, extra_info: dict, mode: str):
    from verl_omni.pipelines.rollout_media import select_artifact

    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unknown ImageBind mode {mode!r}; expected one of: {', '.join(_SUPPORTED_MODES)}.")

    artifacts = extra_info.get("media_artifacts", {})
    video = audio = text = None
    if mode in {"audio_video", "text_video", "all"}:
        video = select_artifact(artifacts, name="video_preview", modality="video", representation="decoded")
    if mode in {"audio_video", "text_audio", "all"}:
        audio = select_artifact(artifacts, name="audio", modality="audio", representation="decoded")
        if audio.spec.layout != "CT":
            raise ValueError(f"ImageBind requires canonical CT audio, got {audio.spec.layout}")
    if mode in {"text_audio", "text_video", "all"}:
        text = ground_truth or ""
    return text, audio, video


def _prepare_inputs(text, audio, video, device):
    try:
        from imagebind.models.imagebind_model import ModalityType
    except ImportError as exc:
        raise ImportError("ImageBind reward requires the non-commercial ImageBind package.") from exc

    inputs = {}
    if text is not None:
        inputs[ModalityType.TEXT] = _preprocess_text(text, device)
    if audio is not None:
        inputs[ModalityType.AUDIO] = _preprocess_audio(audio.data, audio.spec.sample_rate, device)
    if video is not None:
        inputs[ModalityType.VISION] = _preprocess_video(video, device)
    return inputs


def _score_result(similarities, mode, weights):
    if mode == "all":
        score = _aggregate_similarities(similarities, weights)
        return {
            "score": score,
            **{f"{name}_similarity": value for name, value in similarities.items()},
        }
    return {"score": similarities[mode]}


def compute_score(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str | None = None,
    model_name_or_path: str = _DEFAULT_MODEL_PATH,
    mode: str = _DEFAULT_MODE,
    weights: Mapping[str, float] | None = None,
    **kwargs,
) -> dict:
    """Compute a configured ImageBind cross-modal cosine similarity."""
    device = device or get_device_name()
    inputs = _prepare_inputs(*_select_media(ground_truth, extra_info, mode), device)
    with _MODEL_LOCK, torch.no_grad():
        model = _load_imagebind(device, model_name_or_path)
        similarities = _compute_similarities(model(inputs))
    return _score_result(similarities, mode, weights)


class ImageBindModel:
    """ImageBind instance owned by one native reward executor."""

    def __init__(self, model_path: str, device):
        self.device = device
        self.model = _create_imagebind(device, model_path)
        self._infer_lock = threading.Lock()

    @torch.inference_mode()
    def infer(self, text, audio, video) -> dict[str, torch.Tensor]:
        """Return embeddings in model dtype on the worker's assigned device."""
        inputs = _prepare_inputs(text, audio, video, self.device)
        with self._infer_lock:
            return self.model(inputs)

    def close(self) -> None:
        """Release the worker's model after outstanding inference has drained."""
        self.model = None


async def compute_score_native(
    ground_truth: str,
    extra_info: dict,
    *,
    reward_model,
    mode: str = _DEFAULT_MODE,
    weights: Mapping[str, float] | None = None,
) -> dict:
    """Score one sample through its configured ImageBind deployment."""
    media = _select_media(ground_truth, extra_info, mode)
    embeddings = await reward_model.infer(*media)
    return _score_result(_compute_similarities(embeddings), mode, weights)
