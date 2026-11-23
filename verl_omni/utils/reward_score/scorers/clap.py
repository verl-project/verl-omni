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
"""Class-based CLAP scorer backed by the frozen legacy CLAP adapter."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from verl.utils.device import get_device_name, get_torch_device

from verl_omni.utils.reward_score import clap as _frozen
from verl_omni.utils.reward_score.reward_utils import audio_info_from_batch
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import register_scorer

__all__ = ["CLAPScorer", "compute_score_clap_batched"]


@register_scorer("clap")
class CLAPScorer(RewardScorer):
    """LAION CLAP text-audio alignment as a batch-native scorer.

    Args:
        model_path: LAION CLAP checkpoint for weights and the processor
            (default ``laion/larger_clap_general``); the named model's top-level
            ``model_path`` forwards here unless ``executor.kwargs`` sets one.
        model_kwargs: Extra keyword arguments forwarded to
            ``ClapModel.from_pretrained`` when the executor loads the model.
        processor_kwargs: Extra keyword arguments forwarded to
            ``AutoProcessor.from_pretrained`` when the executor loads the processor.
    """

    max_batch_size = _frozen._MAX_BATCH_SIZE
    supports_cpu_offload = True

    def __init__(
        self,
        model_path: str = _frozen._DEFAULT_MODEL,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_kwargs = dict(model_kwargs or {})
        self.processor_kwargs = dict(processor_kwargs or {})
        self._adapter: _frozen.CLAPModel | None = None

    def load(self, device: Any) -> None:
        target = torch.device(device)
        self._adapter = _frozen.CLAPModel(
            model_path=self.model_path,
            device=target,
            model_kwargs=self.model_kwargs,
            processor_kwargs=self.processor_kwargs,
        )
        self.model = self._adapter.model
        self.device = target

    def score_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any] | Exception]:
        results: list[dict[str, Any] | Exception] = [None] * len(batch)
        waveforms = []
        prompts: list[str] = []
        source_rates: list[int] = []
        calibrations: list[tuple[Any, Any, Any, Any]] = []
        valid_indices: list[int] = []

        for index, request in enumerate(batch):
            try:
                single_batch = request.get("batch")
                extra_info = audio_info_from_batch(request.get("extra_info"), single_batch, scorer="CLAP")
                audio_layout = request.get("audio_layout")
                if audio_layout is not None:
                    if extra_info.get("audio_layout") not in (None, audio_layout):
                        raise ValueError("Conflicting audio_layout in scorer config and extra_info.")
                    extra_info["audio_layout"] = audio_layout
                prompt = request.get("ground_truth") or ""
                prompt_key = request.get("prompt_key")
                if prompt_key is not None:
                    if single_batch is None:
                        raise ValueError("CLAP prompt_key requires a single-sample batch.")
                    prompt = single_batch[0].non_tensor_batch["reward_inputs"]["text"][prompt_key]
                waveform, source_rate = _frozen._get_audio(extra_info)
                waveform = _frozen.resample_audio(waveform, source_rate, _frozen._CLAP_SAMPLE_RATE)
                waveform = waveform.numpy().astype(np.float32, copy=False)
                calibration = (
                    request.get("score_scale", 1.0),
                    request.get("score_offset", 0.0),
                    request.get("score_min"),
                    request.get("score_max"),
                )
                waveforms.append(waveform)
                prompts.append(prompt)
                source_rates.append(source_rate)
                calibrations.append(calibration)
                valid_indices.append(index)
            except Exception as error:
                results[index] = error

        if not valid_indices:
            return results

        device_name = get_device_name()
        if self.device.type == device_name and device_name != "cpu":
            device_context = get_torch_device().device(self.device)
        else:
            device_context = nullcontext()
        with device_context:
            output = self._require_adapter().infer(waveforms, prompts)
        audio_embeddings = output["audio_embeddings"]
        text_embeddings = output["text_embeddings"]
        audio_embeddings = F.normalize(audio_embeddings.float(), p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings.float(), p=2, dim=-1)
        scores = (audio_embeddings * text_embeddings).sum(dim=-1)

        for position, index in enumerate(valid_indices):
            try:
                scale, offset, score_min, score_max = calibrations[position]
                score = scores[position].item() * scale + offset
                if score_min is not None:
                    score = max(score, score_min)
                if score_max is not None:
                    score = min(score, score_max)
                results[index] = {"score": score, "source_sample_rate": source_rates[position]}
            except Exception as error:
                results[index] = error
        return results

    # TODO(zhtmike): drop both overrides once the migrated adapter derives
    # placement from model parameters — tracked in #723 (post-window cleanup).
    def wake_up(self) -> None:
        super().wake_up()
        self._require_adapter().device = self.device

    def sleep(self) -> None:
        super().sleep()
        self._require_adapter().device = torch.device("cpu")

    def close(self) -> None:
        super().close()
        if self._adapter is not None:
            self._adapter.close()
        self._adapter = None

    def _require_adapter(self) -> _frozen.CLAPModel:
        if self._adapter is None:
            raise RuntimeError("CLAPScorer has no weights; load(device) must run first")
        return self._adapter


async def compute_score_clap_batched(
    data_source: str,
    ground_truth: str,
    extra_info: dict,
    reward_model,
    *,
    solution_image=None,
    batch=None,
    **kwargs,
) -> dict:
    """Submit one sample to the batched CLAP scorer hosted by ``reward_model``.

    ``reward_model`` is the native executor of the named model whose
    ``executor.model`` resolves to :class:`CLAPScorer`; concurrent
    submissions are packed into CLAP batches. Configured via
    ``reward_functions.<key>.path`` and ``.name`` plus ``model=<name>``.
    """
    request = {
        "data_source": data_source,
        "solution_image": solution_image,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        "batch": batch,
    }
    request.update(kwargs)
    return await reward_model.score(request)
