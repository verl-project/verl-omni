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

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_lock = threading.Lock()
_emo = None
_emo_failed = False

# emotion2vec_plus_large class head order: angry, disgusted, fearful, happy, neutral, other,
# sad, surprised, unknown. GLM-TTS uses index 4 (neutral) for the untagged reward.
_EMO_NEUTRAL_IDX = 4


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def _get_emo(emo_model, device):
    global _emo, _emo_failed

    with _lock:
        if _emo is None and not _emo_failed:
            try:
                import torch
                from funasr import AutoModel

                if device is None:
                    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
                _emo = AutoModel(model=emo_model, device=device, disable_update=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("emotion2vec unavailable (%s), emotion reward disabled for this run", e)
                _emo_failed = True
    return _emo


def compute_score(
    solution_audio,
    extra_info: dict = None,
    emo_model: str = "iic/emotion2vec_plus_large",
    emo_max_sec: float = 60.0,
    device: str = None,
    **kwargs,
) -> dict:
    """Emotion reward, GLM-TTS form: P(tagged emotion), or 1 - P(neutral) for untagged prompts.

    Args:
        solution_audio: (wav, sr) tuple with a float32 waveform, or None when synthesis failed.
        extra_info (dict, optional): May carry emotion, the emotion2vec class index of the
            prompt's tag; untagged prompts (absent) use 1 - P(neutral) as expressiveness.

    Returns:
        dict: {"score": class probability, "p_neutral": P(neutral)}. Unavailable inputs score 0.0.
    """
    extra_info = extra_info or {}
    emotion = extra_info.get("emotion")
    emotion = int(emotion) if emotion is not None else -1

    if solution_audio is None:
        return {"score": 0.0, "p_neutral": -1.0}
    m = _get_emo(emo_model, device)
    if m is None:
        return {"score": 0.0, "p_neutral": -1.0}
    wav, sr = solution_audio
    try:
        # Cap the scored window: emotion2vec attention is quadratic in length and a runaway clip
        # OOMs; degenerate rollouts are already penalized by the asr stability dimension.
        max_n = int(emo_max_sec * sr)
        if max_n > 0 and wav.shape[0] > max_n:
            wav = wav[:max_n]
        audio = _resample(wav, sr, 16000)
        res = m.generate(audio, fs=16000, granularity="utterance", extract_embedding=False, disable_pbar=True)
        scores = res[0]["scores"]
        p_neutral = float(scores[_EMO_NEUTRAL_IDX])
        p_emo = float(scores[emotion]) if emotion >= 0 else 1.0 - p_neutral
        return {"score": p_emo, "p_neutral": p_neutral}
    except Exception as e:  # noqa: BLE001
        logger.warning("emotion2vec predict failed (%s)", e)
        return {"score": 0.0, "p_neutral": -1.0}
