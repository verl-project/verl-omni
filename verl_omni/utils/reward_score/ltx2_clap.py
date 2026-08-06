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

"""Audio-text alignment reward for LTX-2.3 using LAION CLAP."""

import threading

import numpy as np
import torch
import torch.nn.functional as F

_CLAP_SAMPLE_RATE = 48_000
_DEFAULT_MODEL = "laion/larger_clap_general"
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()


def _get_audio(extra_info: dict) -> tuple[torch.Tensor, int]:
    audio = extra_info.get("audio")
    if audio is None:
        raise KeyError("CLAP reward requires decoded audio in extra_info['audio'].")
    audio = torch.as_tensor(audio).detach().float().cpu()
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    elif audio.ndim != 1:
        raise ValueError(f"Expected audio shape (T,) or (C,T), got {tuple(audio.shape)}.")

    sample_rate = extra_info.get("audio_sample_rate", _CLAP_SAMPLE_RATE)
    if isinstance(sample_rate, torch.Tensor):
        sample_rate = sample_rate.item()
    if sample_rate is None:
        raise KeyError("CLAP reward requires extra_info['audio_sample_rate'].")
    return audio, int(sample_rate)


def _load_clap(model_name_or_path: str, device: str):
    key = (model_name_or_path, device)
    if key not in _MODEL_CACHE:
        from transformers import ClapModel, ClapProcessor

        model = ClapModel.from_pretrained(model_name_or_path).to(device).eval()
        processor = ClapProcessor.from_pretrained(model_name_or_path)
        _MODEL_CACHE[key] = (model, processor)
    return _MODEL_CACHE[key]


def compute_score_clap(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str = "cuda",
    model_name_or_path: str = _DEFAULT_MODEL,
    **kwargs,
) -> dict:
    """Compute cosine similarity between generated audio and its text prompt."""
    del data_source, solution_image, kwargs
    import torchaudio.functional as audio_functional

    waveform, source_rate = _get_audio(extra_info)
    if source_rate != _CLAP_SAMPLE_RATE:
        waveform = audio_functional.resample(
            waveform.unsqueeze(0),
            orig_freq=source_rate,
            new_freq=_CLAP_SAMPLE_RATE,
        ).squeeze(0)

    with _MODEL_LOCK, torch.no_grad():
        model, processor = _load_clap(model_name_or_path, device)
        inputs = processor(
            text=[ground_truth or ""],
            audio=[waveform.numpy().astype(np.float32)],
            sampling_rate=_CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        audio_embedding = F.normalize(outputs.audio_embeds, p=2, dim=-1)
        text_embedding = F.normalize(outputs.text_embeds, p=2, dim=-1)
        score = (audio_embedding * text_embedding).sum(dim=-1)[0].float().item()
    return {"score": score, "source_sample_rate": source_rate}
