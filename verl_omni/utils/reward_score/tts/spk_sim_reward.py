# Copyright 2026 Gulp AI Inc and/or its affiliates
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

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_lock = threading.Lock()
_spk = None
_ref_cache: dict = {}


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _get_spk(spk_model, device):
    global _spk

    with _lock:
        if _spk is None:
            import torch
            from modelscope.pipelines import pipeline

            if device is None:
                device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            _spk = pipeline(task="speaker-verification", model=spk_model, device=device)
    return _spk


def _spk_embed(wav, sr, spk_model, device):
    try:
        out = _get_spk(spk_model, device)([_resample(wav, sr, 16000)], output_emb=True)
        return np.asarray(out["embs"][0], dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        logger.warning("speaker embedding unavailable (%s)", e)
        return None


def _load_ref_wav(path):
    if not path:
        return None
    if path not in _ref_cache:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        _ref_cache[path] = (np.asarray(wav, dtype=np.float32), int(sr))
    return _ref_cache[path]


def compute_score(
    solution_audio,
    extra_info: dict = None,
    spk_model: str = "iic/speech_eres2net_sv_en_voxceleb_16k",
    device: str = None,
    **kwargs,
) -> dict:
    """Speaker similarity reward (1 + cos) / 2, GLM-TTS form, against a reference clip.

    Args:
        solution_audio: (wav, sr) tuple with a float32 waveform, or None when synthesis failed.
        extra_info (dict, optional): Carries target_audio or ref_audio, the reference clip path
            (the per-utterance target recording is preferred over the fixed clone reference).

    Returns:
        dict: {"score": (1 + cos) / 2, "cos": raw cosine}. Unavailable inputs score 0.0.
    """
    extra_info = extra_info or {}
    ref = _load_ref_wav(extra_info.get("target_audio") or extra_info.get("ref_audio"))
    if solution_audio is None or ref is None:
        return {"score": 0.0, "cos": -1.0}
    wav, sr = solution_audio
    a = _spk_embed(wav, sr, spk_model, device)
    b = _spk_embed(ref[0], ref[1], spk_model, device)
    if a is None or b is None:
        return {"score": 0.0, "cos": -1.0}
    cos = _cosine(a, b)
    return {"score": (1.0 + cos) / 2.0, "cos": cos}
