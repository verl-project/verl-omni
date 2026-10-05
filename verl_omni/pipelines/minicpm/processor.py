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
"""Native MiniCPM-o media processing shared by collection and actor replay."""

import copy
import os

import numpy as np
import torch

IMAGE_PATTERN = "(<image>./</image>)"
AUDIO_PATTERN = "(<audio>./</audio>)"


def render_minicpmo_messages(processor, messages, **kwargs):
    """Render structured messages with the native serving placeholders."""
    patterns = {"image": IMAGE_PATTERN, "audio": AUDIO_PATTERN}
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item["text"])
                elif item.get("type") in patterns:
                    parts.append(patterns[item["type"]])
                else:
                    raise ValueError(f"Unsupported MiniCPM-o simplex content type: {item.get('type')!r}.")
            content = "".join(parts)
        normalized.append({**message, "content": content})
    return processor.tokenizer.apply_chat_template(normalized, tokenize=kwargs.pop("tokenize", False), **kwargs)


def _load_audio(audio, target_rate=16000):
    if isinstance(audio, tuple) and len(audio) == 2:
        waveform, source_rate = audio
    elif isinstance(audio, np.ndarray | torch.Tensor):
        waveform, source_rate = audio, target_rate
    elif isinstance(audio, str | os.PathLike):
        import soundfile as sf

        waveform, source_rate = sf.read(os.fspath(audio), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
    else:
        raise TypeError(f"Unsupported MiniCPM-o audio value: {type(audio).__name__}.")
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().float().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not waveform.size:
        raise ValueError("MiniCPM-o audio arrays must be non-empty mono waveforms.")
    if int(source_rate) != target_rate:
        from scipy.signal import resample_poly

        gcd = np.gcd(int(source_rate), target_rate)
        waveform = resample_poly(waveform, target_rate // gcd, int(source_rate) // gcd).astype(np.float32)
    return waveform


def prepare_minicpmo_inputs(processor, rendered_prompt, *, images=None, audios=None, mm_processor_kwargs=None):
    """Produce the exact expanded prompt and native per-sample media tensors."""
    images, audios = list(images or []), list(audios or [])
    if len(audios) > 1:
        raise NotImplementedError("MiniCPM-o simplex replay currently accepts one audio clip per prompt.")
    if rendered_prompt.count(IMAGE_PATTERN) != len(images) or rendered_prompt.count(AUDIO_PATTERN) != len(audios):
        raise ValueError("MiniCPM-o prompt placeholders must consume every image/audio input exactly once.")
    kwargs = dict(mm_processor_kwargs or {})
    kwargs.pop("sampling_rate", None)
    kwargs.setdefault("return_tensors", "pt")
    prompt = rendered_prompt.replace(IMAGE_PATTERN, "<image>./</image>").replace(AUDIO_PATTERN, "<audio>./</audio>")
    return processor(text=[prompt], images=[images] if images else None, audios=[audios] if audios else None, **kwargs)


def split_minicpmo_actor_inputs(model_inputs):
    """Unbatch native processor outputs without converting its tensors a second time."""
    actor_inputs = dict(model_inputs)
    prompt_ids = torch.as_tensor(actor_inputs.pop("input_ids")).reshape(-1).long().tolist()
    actor_inputs.pop("attention_mask", None)
    for key in (
        "pixel_values",
        "image_sizes",
        "image_bound",
        "tgt_sizes",
        "audio_bounds",
        "spk_bounds",
        "audio_feature_lens",
    ):
        value = actor_inputs.get(key)
        if isinstance(value, list | tuple) and len(value) == 1:
            actor_inputs[key] = value[0]
    features = actor_inputs.get("audio_features")
    if isinstance(features, torch.Tensor) and features.ndim == 3 and features.shape[0] == 1:
        actor_inputs["audio_features"] = features[0]
    return prompt_ids, actor_inputs


def clone_minicpmo_actor_inputs(value):
    """Keep mutable reward/transport processing separate from the replay snapshot."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_minicpmo_actor_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_minicpmo_actor_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_minicpmo_actor_inputs(item) for item in value)
    return copy.deepcopy(value)
