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

"""Text-audio alignment reward using LAION CLAP."""

import asyncio
import logging
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from verl.utils.device import get_device_name

_CLAP_SAMPLE_RATE = 48_000
_DEFAULT_MODEL = "laion/larger_clap_general"
_MAX_BATCH_SIZE = 16
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()
_BATCHING_STATE = threading.local()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class _BatchingState:
    def __init__(self, loop):
        self.loop = loop
        self.queue = asyncio.Queue(maxsize=_MAX_BATCH_SIZE)
        self.consumer_task = None
        self.consumer_lock = asyncio.Lock()


def _get_batching_state() -> _BatchingState:
    loop = asyncio.get_running_loop()
    state = getattr(_BATCHING_STATE, "value", None)
    if state is None or state.loop is not loop:
        state = _BatchingState(loop)
        _BATCHING_STATE.value = state
    return state


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


def _score_batch(requests) -> list[tuple[float, int] | Exception]:
    """Prepare and score ready requests in model- and device-specific batches."""
    results = [None] * len(requests)
    grouped_requests = {}
    for index, (prompt, extra_info, model_name_or_path, device, _) in enumerate(requests):
        try:
            waveform, source_rate = _get_audio(extra_info)
            waveform = _resample_audio(waveform, source_rate)
            key = (model_name_or_path, device)
            grouped_requests.setdefault(key, []).append(
                (index, prompt, waveform.numpy().astype(np.float32), source_rate)
            )
        except Exception as e:
            results[index] = e

    for (model_name_or_path, device), group in grouped_requests.items():
        try:
            # Loop-local consumers may run in different threads, so cached model access must remain serialized.
            with _MODEL_LOCK:
                model, processor = _load_clap(model_name_or_path, device)
                inputs = processor(
                    text=[prompt for _, prompt, _, _ in group],
                    audio=[waveform for _, _, waveform, _ in group],
                    sampling_rate=_CLAP_SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    audio_embedding = F.normalize(outputs.audio_embeds, p=2, dim=-1)
                    text_embedding = F.normalize(outputs.text_embeds, p=2, dim=-1)
                    scores = (audio_embedding * text_embedding).sum(dim=-1).float().tolist()
            for (index, _, _, source_rate), score in zip(group, scores, strict=True):
                results[index] = (score, source_rate)
        except Exception as e:
            for index, _, _, _ in group:
                results[index] = e

    return results


def _fail_requests(requests, error: Exception) -> None:
    for *_, future in requests:
        if not future.done():
            future.set_exception(error)


def _drain_failed_requests(state: _BatchingState, error: Exception) -> None:
    requests = []
    while True:
        try:
            request = state.queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if request[0] is not None:
            requests.append(request)
    _fail_requests(requests, error)


async def _consumer_loop(state: _BatchingState):
    loop = asyncio.get_running_loop()
    requests = []
    stop_error = RuntimeError("CLAP batch consumer stopped before completing inference.")
    try:
        while True:
            request = await state.queue.get()
            if request[0] is None:
                _drain_failed_requests(state, stop_error)
                break

            requests = [request]
            should_stop = False
            await asyncio.sleep(0)
            while len(requests) < _MAX_BATCH_SIZE:
                try:
                    request = state.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if request[0] is None:
                    should_stop = True
                    break
                requests.append(request)

            results = await loop.run_in_executor(None, _score_batch, requests)
            for (*_, future), result in zip(requests, results, strict=True):
                if future.done():
                    continue
                if isinstance(result, Exception):
                    logger.error("CLAP inference failed", exc_info=(type(result), result, result.__traceback__))
                    future.set_exception(result)
                else:
                    future.set_result(result)
            requests = []

            if should_stop:
                _drain_failed_requests(state, stop_error)
                break
    except asyncio.CancelledError:
        error = RuntimeError("CLAP batch consumer was cancelled before completing inference.")
        _fail_requests(requests, error)
        _drain_failed_requests(state, error)
        raise
    except BaseException as error:
        if not isinstance(error, Exception):
            error = RuntimeError(f"CLAP batch consumer stopped unexpectedly: {type(error).__name__}")
        _fail_requests(requests, error)
        _drain_failed_requests(state, error)
        raise


async def _ensure_consumer(state: _BatchingState):
    if state.consumer_task is not None and not state.consumer_task.done():
        return
    async with state.consumer_lock:
        if state.consumer_task is None or state.consumer_task.done():
            state.consumer_task = asyncio.create_task(_consumer_loop(state))


async def compute_score(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str | None = None,
    model_name_or_path: str = _DEFAULT_MODEL,
    *,
    reward_model=None,
    batch=None,
    prompt_key: str | None = None,
    score_scale: float = 1.0,
    score_offset: float = 0.0,
    score_min: float | None = None,
    score_max: float | None = None,
    **kwargs,
) -> dict:
    """Score text/audio alignment through the local cache or a managed model.

    Defaults preserve the existing cached, batched cosine scorer. A supplied
    reward_model owns inference and lifecycle. With a single-sample batch, that
    path reads decoded audio and its sample rate directly; otherwise it uses
    extra_info. prompt_key selects reward_inputs.text[prompt_key] from batch;
    without it, ground_truth supplies the text. Apply scale/offset to cosine,
    then optional lower/upper bounds. No recipe or checkpoint is hardcoded.
    """
    del data_source, solution_image, kwargs
    prompt = ground_truth or ""
    if batch is not None and (reward_model is not None or prompt_key is not None):
        if len(batch) != 1:
            raise ValueError("CLAP scoring requires exactly one sample.")
        item = batch[0]
        extra_info = dict(extra_info or {})
        for key in ("audio", "audio_sample_rate"):
            if key in item.batch:
                extra_info[key] = item.batch[key]
            elif key in item.non_tensor_batch:
                extra_info[key] = item.non_tensor_batch[key]
    if prompt_key is not None:
        if batch is None:
            raise ValueError("CLAP prompt_key requires a single-sample batch.")
        prompt = item.non_tensor_batch["reward_inputs"]["text"][prompt_key]

    if reward_model is None:
        device = device or get_device_name()
        loop = asyncio.get_running_loop()
        state = _get_batching_state()
        future = loop.create_future()
        await _ensure_consumer(state)
        await state.queue.put((prompt, extra_info, model_name_or_path, device, future))
        await _ensure_consumer(state)
        score, source_rate = await future
    else:
        waveform, source_rate = _get_audio(extra_info)
        waveform = _resample_audio(waveform, source_rate)
        output = await reward_model.infer([waveform.numpy().astype(np.float32, copy=False)], [prompt])
        audio_embeddings = F.normalize(output["audio_embeddings"].float(), p=2, dim=-1)
        text_embeddings = F.normalize(output["text_embeddings"].float(), p=2, dim=-1)
        score = (audio_embeddings * text_embeddings).sum(dim=-1).item()

    score = score * score_scale + score_offset
    if score_min is not None:
        score = max(score, score_min)
    if score_max is not None:
        score = min(score, score_max)
    return {"score": score, "source_sample_rate": source_rate}


def _resample_audio(waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
    if source_rate == _CLAP_SAMPLE_RATE:
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(
        waveform.unsqueeze(0),
        orig_freq=source_rate,
        new_freq=_CLAP_SAMPLE_RATE,
    ).squeeze(0)


class CLAPModel:
    """Raw CLAP inference adapter owned by a native reward executor."""

    def __init__(self, model_path: str, device, model_kwargs=None, processor_kwargs=None) -> None:
        from transformers import AutoProcessor, ClapModel

        self.device = torch.device(device)
        self.model = ClapModel.from_pretrained(model_path, **(model_kwargs or {})).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, **(processor_kwargs or {}))
        self._infer_lock = threading.Lock()

    def close(self) -> None:
        """Drop model/processor references; reuse requires constructing a new adapter."""
        self.model = None
        self.processor = None
        self.device = None

    @torch.inference_mode()
    def infer(self, waveforms: list[np.ndarray], prompts: list[str]) -> dict[str, torch.Tensor]:
        """Encode aligned mono 48 kHz arrays and text without gradients.

        The processor pads/truncates the batch; its tensor outputs move to the
        active device. Return detached CPU ``audio_embeddings`` and
        ``text_embeddings``, each ``[B, D]`` in model-output dtype. Cosine
        normalization and score scaling are performed by the scorer.
        """
        with self._infer_lock:
            inputs = self.processor(
                text=prompts,
                audio=waveforms,
                sampling_rate=_CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            outputs = self.model(**inputs)
            return {
                "audio_embeddings": outputs.audio_embeds.detach().cpu(),
                "text_embeddings": outputs.text_embeds.detach().cpu(),
            }
