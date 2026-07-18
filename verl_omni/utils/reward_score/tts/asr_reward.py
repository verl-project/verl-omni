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
import re
import threading

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_lock = threading.Lock()
_asr = None

# ASR checkpoint whose tokenizer ships the British->American spelling map (normalizer.json).
_SPELLING_MAP_REPO = "distil-whisper/distil-large-v3"

_WHISPER_NORM = None

_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_WORD_RE.sub(" ", s.lower()).split())


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _whisper_normalizer(spelling_map_repo: str = _SPELLING_MAP_REPO):
    global _WHISPER_NORM
    if _WHISPER_NORM is None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        spelling = {}
        try:  # British->American map (cancelled->canceled etc.) ships with the ASR tokenizer
            import json

            from transformers.utils import cached_file

            spelling = json.load(open(cached_file(spelling_map_repo, "normalizer.json")))
        except Exception:  # noqa: BLE001, fold everything else even without the spelling map
            pass
        _WHISPER_NORM = EnglishTextNormalizer(spelling)
    return _WHISPER_NORM


def normalize_for_cer(text: str) -> str:
    """Whisper-standard fold: lowercase, punctuation, contractions, digit<->word, hesitation
    fillers stripped, bracketed content fully removed. Apply to BOTH reference and hypothesis.

    The matching data-side written-to-spoken transform lives with the example
    (examples/qwen3_tts_gspo_trainer/data_process/tts_verbalize.py).
    """
    return re.sub(r"\s+", " ", _whisper_normalizer()(text)).strip()


def _cer_normalize(s: str) -> str:
    # Falls back to the simple normalizer where the Whisper normalizer cannot load
    # (CPU-only unit tests).
    try:
        return normalize_for_cer(s)
    except Exception:  # noqa: BLE001
        return normalize_text(s)


def cer(reference: str, hypothesis: str) -> float | None:
    """Character error rate over normalized text. None if the reference is empty."""
    ref = _cer_normalize(reference).replace(" ", "")
    hyp = _cer_normalize(hypothesis).replace(" ", "")
    if not ref:
        return None
    return _levenshtein(ref, hyp) / len(ref)


def has_repetition(text: str | None, n: int = 3, max_span: int = 30) -> bool:
    """True if any span of n..max_span words repeats immediately (a TTS looping artifact)."""
    if not text:
        return False
    words = text.split()
    upper = min(max_span, len(words) // 2)
    for span in range(n, upper + 1):
        for i in range(len(words) - 2 * span + 1):
            if words[i : i + span] == words[i + span : i + 2 * span]:
                return True
    return False


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def _get_asr(whisper_model, whisper_device, whisper_backend, whisper_compute_type):
    # Two backends: "faster" (faster-whisper/ctranslate2, CPU) and "transformers" (torch-native
    # whisper, default on cuda; ctranslate2 is a CUDA-12 binary and clashes with a CUDA-13 stack).
    global _asr

    with _lock:
        if _asr is not None:
            return _asr
        import torch

        if whisper_device is None:
            whisper_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        use_cuda = whisper_device.startswith("cuda")
        if whisper_model is None:
            whisper_model = "distil-whisper/distil-large-v3" if use_cuda else "large-v3"
        backend = whisper_backend or ("transformers" if use_cuda else "faster")
        if backend == "transformers":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            dtype = torch.float16 if use_cuda else torch.float32
            logger.info("loading transformers whisper %r on %s", whisper_model, whisper_device)
            model = (
                AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model, dtype=dtype, low_cpu_mem_usage=False)
                .to(whisper_device)
                .eval()
            )
            proc = AutoProcessor.from_pretrained(whisper_model)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=proc.tokenizer,
                feature_extractor=proc.feature_extractor,
                torch_dtype=dtype,
                device=whisper_device,
                chunk_length_s=30,
            )
            _asr = ("transformers", pipe)
        else:
            from faster_whisper import WhisperModel

            if use_cuda:
                idx = int(whisper_device.split(":", 1)[1]) if ":" in whisper_device else 0
                dev, kw = "cuda", {"device_index": idx}
            else:
                dev, kw = "cpu", {}
            ct = whisper_compute_type if dev == "cuda" else "int8"
            logger.info("loading faster-whisper %r (%s) on %s", whisper_model, ct, dev)
            _asr = ("faster", WhisperModel(whisper_model, device=dev, compute_type=ct, **kw))
    return _asr


def transcribe(
    wav: np.ndarray,
    sr: int,
    whisper_model: str = None,
    whisper_device: str = None,
    whisper_backend: str = None,
    whisper_compute_type: str = "int8",
    language: str = "en",
) -> str:
    backend, model = _get_asr(whisper_model, whisper_device, whisper_backend, whisper_compute_type)
    audio = np.asarray(_resample(wav, sr, 16000), dtype=np.float32)
    if backend == "transformers":
        import torch

        # The worker's torch default device can be left as meta by prior model loads, which
        # breaks the HF pipeline; pin it to cpu for the call.
        prev_dev = torch.get_default_device()
        torch.set_default_device("cpu")
        try:
            out = model({"array": audio, "sampling_rate": 16000})
        finally:
            torch.set_default_device(prev_dev)
        return (out.get("text") or "").strip()
    seg_iter, _ = model.transcribe(audio, language=language, beam_size=1, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in seg_iter).strip()


def compute_score(
    solution_audio,
    ground_truth: str,
    extra_info: dict = None,
    whisper_model: str = None,
    whisper_device: str = None,
    whisper_backend: str = None,
    whisper_compute_type: str = "int8",
    language: str = "en",
    cer_outlier_threshold: float = 0.30,
    truncation_ratio: float = 0.5,
    speaking_rate_wps: float = 2.5,
    repetition_ngram: int = 3,
    p_trunc: float = 1.0,
    p_rep: float = 1.0,
    p_outlier: float = 1.0,
    p_fail: float = 1.0,
    **kwargs,
) -> dict:
    """Intelligibility reward exp(-2.5 * CER), GLM-TTS form, plus transcript-derived stability.

    Both the score and the stability penalty consume one whisper transcription, so they live in
    one function; the stab extra is a separate reward dimension for per-dimension estimators.

    Args:
        solution_audio: (wav, sr) tuple with a float32 waveform, or None when synthesis failed.
        ground_truth (str): The text the audio should speak.
        extra_info (dict, optional): May carry text (overrides ground_truth as the CER target).

    Returns:
        dict: {"score": exp(-2.5*CER), "cer", "stab", "synth_ok", "truncated", "repeated"}.
    """
    extra_info = extra_info or {}
    text = extra_info.get("text") or ground_truth or ""

    if solution_audio is None:
        return {"score": 0.0, "cer": -1.0, "stab": -p_fail, "synth_ok": 0.0, "truncated": 0.0, "repeated": 0.0}

    wav, sr = solution_audio
    import math

    transcript = transcribe(wav, sr, whisper_model, whisper_device, whisper_backend, whisper_compute_type, language)
    c = cer(text, transcript)
    score = math.exp(-2.5 * c) if c is not None else 0.0

    duration_s = len(wav) / sr if sr else None
    words = len(text.split())
    expected_s = words / speaking_rate_wps if words and speaking_rate_wps > 0 else None
    truncated = duration_s is not None and expected_s is not None and duration_s < truncation_ratio * expected_s
    repeated = has_repetition(transcript, repetition_ngram)
    outlier = c is not None and c > cer_outlier_threshold
    stab = -(
        p_trunc * float(truncated) + p_rep * float(repeated) + p_outlier * float(outlier) + p_fail * float(c is None)
    )

    return {
        "score": score,
        "cer": c if c is not None else -1.0,
        "stab": stab,
        "synth_ok": 1.0,
        "truncated": float(truncated),
        "repeated": float(repeated),
    }
