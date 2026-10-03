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

"""Reference-fidelity reward for Ref2VA using CLIP (image/video) and CLAP (audio).

Ref2VA conditions generation on one or more reference images, videos and/or
audio clips (see ``examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py``).
Existing cross-modal rewards (:mod:`clap`, :mod:`imagebind`) only measure
audio/video alignment against the *text prompt* -- they say nothing about
whether the generated output actually preserves the subject/appearance/sound
given in the reference. This scorer closes that gap:

- Image and video references (``extra_info["source_images"]`` /
  ``extra_info["source_videos"]``) are embedded with CLIP alongside sampled
  generated-video frames; the mean cosine similarity is the visual score.
- When a reference audio clip is also present (``extra_info["source_audios"]``)
  and the rollout produced audio (``extra_info["audio"]``), CLAP audio-audio
  cosine similarity is blended in with weight ``audio_weight``.

Every Ref2VA row carries at least one image or video reference, so the visual
score is always computed; the audio term is opportunistic.
"""

import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from verl.utils.device import get_device_name

_DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
_DEFAULT_CLAP_MODEL = "laion/larger_clap_general"
_DEFAULT_NUM_FRAMES = 4
_DEFAULT_FRAMES_PER_REFERENCE_VIDEO = 2
_DEFAULT_AUDIO_WEIGHT = 0.3
_CLAP_SAMPLE_RATE = 48_000
_CLIP_MODEL_CACHE = {}
_CLIP_MODEL_LOCK = threading.Lock()
_CLAP_MODEL_CACHE = {}
_CLAP_MODEL_LOCK = threading.Lock()


def _load_clip(model_name_or_path: str, device: str):
    key = (model_name_or_path, device)
    if key not in _CLIP_MODEL_CACHE:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name_or_path).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name_or_path)
        _CLIP_MODEL_CACHE[key] = (model, processor)
    return _CLIP_MODEL_CACHE[key]


def _load_clap(model_name_or_path: str, device: str):
    key = (model_name_or_path, device)
    if key not in _CLAP_MODEL_CACHE:
        from transformers import ClapModel, ClapProcessor

        model = ClapModel.from_pretrained(model_name_or_path).to(device).eval()
        processor = ClapProcessor.from_pretrained(model_name_or_path)
        _CLAP_MODEL_CACHE[key] = (model, processor)
    return _CLAP_MODEL_CACHE[key]


def _as_path_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _decode_video_frames(path: str, num_frames: int) -> list[Image.Image]:
    import imageio.v3 as iio

    frames = [Image.fromarray(frame) for frame in iio.imiter(path, plugin="pyav")]
    if not frames:
        raise ValueError(f"Reference video has no frames: {path}")
    sample_count = max(1, min(num_frames, len(frames)))
    indices = torch.linspace(0, len(frames) - 1, sample_count).round().long().tolist()
    return [frames[index] for index in indices]


def _reference_visual_frames(extra_info: dict, frames_per_video: int) -> list[Image.Image]:
    frames = [Image.open(path).convert("RGB") for path in _as_path_list(extra_info.get("source_images"))]
    for video_path in _as_path_list(extra_info.get("source_videos")):
        frames.extend(_decode_video_frames(video_path, frames_per_video))
    if not frames:
        raise KeyError(
            "Ref fidelity reward requires reference media in extra_info['source_images'] "
            "or extra_info['source_videos']."
        )
    return frames


def _sample_generated_frames(solution_image, num_frames: int) -> list[Image.Image]:
    from verl_omni.utils.reward_score.reward_utils import normalize_video_tensor, video_tensor_to_pil_frames

    if solution_image is None:
        raise ValueError("Ref fidelity reward requires generated video/image in solution_image.")
    solution_image = torch.as_tensor(solution_image)
    video = solution_image.unsqueeze(0) if solution_image.ndim == 3 else solution_image
    video = normalize_video_tensor(video)

    frame_count = video.shape[0]
    sample_count = max(1, min(num_frames, frame_count))
    indices = torch.linspace(0, frame_count - 1, sample_count).round().long()
    return video_tensor_to_pil_frames(video[indices])


def _pooled_features(output):
    """Unwrap the projected embedding tensor from a feature-extraction call.

    ``CLIPModel.get_image_features`` / ``ClapModel.get_audio_features`` return a plain
    tensor on some transformers versions and a ``BaseModelOutputWithPooling`` (embedding
    in ``.pooler_output``) or a tuple on others -- normalize both to the tensor.
    """
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output[0]


def _clip_similarity(model, processor, device: str, reference_frames, generated_frames) -> float:
    reference_inputs = processor(images=reference_frames, return_tensors="pt").to(device)
    generated_inputs = processor(images=generated_frames, return_tensors="pt").to(device)
    reference_embeds = F.normalize(_pooled_features(model.get_image_features(**reference_inputs)), p=2, dim=-1)
    generated_embeds = F.normalize(_pooled_features(model.get_image_features(**generated_inputs)), p=2, dim=-1)
    reference_embed = F.normalize(reference_embeds.mean(dim=0), p=2, dim=-1)
    generated_embed = F.normalize(generated_embeds.mean(dim=0), p=2, dim=-1)
    return (reference_embed * generated_embed).sum().float().item()


def _load_reference_waveform(path: str, target_sample_rate: int) -> np.ndarray:
    import torchaudio
    import torchaudio.functional as audio_functional

    waveform, source_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != target_sample_rate:
        waveform = audio_functional.resample(waveform, source_rate, target_sample_rate)
    return waveform.squeeze(0).numpy().astype(np.float32)


def _generated_waveform(extra_info: dict, target_sample_rate: int) -> np.ndarray:
    import torchaudio.functional as audio_functional

    audio = extra_info.get("audio")
    sample_rate = extra_info.get("audio_sample_rate", target_sample_rate)
    if isinstance(sample_rate, torch.Tensor):
        sample_rate = sample_rate.item()

    waveform = torch.as_tensor(audio).detach().float().cpu()
    while waveform.ndim > 1 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if int(sample_rate) != target_sample_rate:
        waveform = audio_functional.resample(waveform.unsqueeze(0), int(sample_rate), target_sample_rate).squeeze(0)
    return waveform.numpy().astype(np.float32)


def _clap_audio_similarity(model, processor, device: str, reference_waveforms, generated_waveform) -> float:
    inputs = processor(
        audio=[*reference_waveforms, generated_waveform],
        sampling_rate=_CLAP_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    embeds = F.normalize(_pooled_features(model.get_audio_features(**inputs)), p=2, dim=-1)
    reference_embed = F.normalize(embeds[:-1].mean(dim=0), p=2, dim=-1)
    generated_embed = embeds[-1]
    return (reference_embed * generated_embed).sum().float().item()


def compute_score_ref_fidelity(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str | None = None,
    clip_model_name_or_path: str = _DEFAULT_CLIP_MODEL,
    clap_model_name_or_path: str = _DEFAULT_CLAP_MODEL,
    num_frames: int = _DEFAULT_NUM_FRAMES,
    frames_per_reference_video: int = _DEFAULT_FRAMES_PER_REFERENCE_VIDEO,
    audio_weight: float = _DEFAULT_AUDIO_WEIGHT,
    **kwargs,
) -> dict:
    """Blend CLIP visual reference fidelity with CLAP audio reference fidelity, when available."""
    del data_source, ground_truth, kwargs
    device = device or get_device_name()

    reference_frames = _reference_visual_frames(extra_info, frames_per_reference_video)
    generated_frames = _sample_generated_frames(solution_image, num_frames)
    with _CLIP_MODEL_LOCK, torch.no_grad():
        clip_model, clip_processor = _load_clip(clip_model_name_or_path, device)
        visual_similarity = _clip_similarity(clip_model, clip_processor, device, reference_frames, generated_frames)

    result = {
        "score": visual_similarity,
        "ref_fidelity_visual_similarity": visual_similarity,
        "ref_fidelity_num_references": len(reference_frames),
        "ref_fidelity_num_frames": len(generated_frames),
    }

    reference_audio_paths = _as_path_list(extra_info.get("source_audios"))
    if reference_audio_paths and extra_info.get("audio") is not None:
        reference_waveforms = [_load_reference_waveform(path, _CLAP_SAMPLE_RATE) for path in reference_audio_paths]
        generated_waveform = _generated_waveform(extra_info, _CLAP_SAMPLE_RATE)
        with _CLAP_MODEL_LOCK, torch.no_grad():
            clap_model, clap_processor = _load_clap(clap_model_name_or_path, device)
            audio_similarity = _clap_audio_similarity(
                clap_model, clap_processor, device, reference_waveforms, generated_waveform
            )
        result["ref_fidelity_audio_similarity"] = audio_similarity
        result["score"] = (1.0 - audio_weight) * visual_similarity + audio_weight * audio_similarity

    return result
