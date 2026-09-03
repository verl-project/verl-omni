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
    try:
        import torchaudio.functional as audio_functional
    except Exception as e:
        return [e] * len(requests)

    grouped_requests = {}
    for index, (prompt, extra_info, model_name_or_path, device, _) in enumerate(requests):
        try:
            waveform, source_rate = _get_audio(extra_info)
            if source_rate != _CLAP_SAMPLE_RATE:
                waveform = audio_functional.resample(
                    waveform.unsqueeze(0),
                    orig_freq=source_rate,
                    new_freq=_CLAP_SAMPLE_RATE,
                ).squeeze(0)
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
    **kwargs,
) -> dict:
    """Compute cosine similarity between generated audio and its text prompt."""
    del data_source, solution_image, kwargs
    device = device or get_device_name()

    loop = asyncio.get_running_loop()
    state = _get_batching_state()
    future = loop.create_future()
    await _ensure_consumer(state)
    await state.queue.put((ground_truth or "", extra_info, model_name_or_path, device, future))
    await _ensure_consumer(state)
    score, source_rate = await future
    return {"score": score, "source_sample_rate": source_rate}
