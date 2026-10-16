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
"""Class-based PickScore scorer — the pilot port of the layered scorer interface.

All scoring math lives in the frozen legacy module ``pickscore_reward.py``: this
port imports it and adds only the protocol glue, so one formula serves every
entry-point family. Batching, admission, cancellation, lifecycle choreography,
gc, and accelerator-cache clearing are framework-owned — unlike the legacy
module, this file has no queue, no consumer task, no threading state, and no
cleanup code.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from verl.utils.device import get_device_id, get_device_name, get_torch_device

from verl_omni.utils.reward_score import pickscore_reward as _frozen
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import register_scorer

__all__ = ["PickScoreScorer", "compute_score_pickscore_batched"]


@register_scorer("pickscore", modality="image")
class PickScoreScorer(RewardScorer):
    """PickScore preference model as a batch-native scorer."""

    max_batch_size = _frozen._MAX_BATCH_SIZE
    supports_cpu_offload = True

    def __init__(
        self,
        model_path: str = _frozen._MODEL_PATH,
        processor_path: str = _frozen._PROCESSOR_PATH,
        dtype: torch.dtype = torch.float32,
    ):
        self.model_path = model_path
        self.processor_path = processor_path
        self.dtype = dtype
        self._inferencer: _frozen._PickScoreInferencer | None = None

    def load(self, device: Any) -> None:
        target = torch.device(device) if device is not None else torch.device(get_device_name(), get_device_id())
        self._inferencer = _frozen._PickScoreInferencer(
            device=target,
            dtype=self.dtype,
            model_path=self.model_path,
            processor_path=self.processor_path,
        )
        self.model = self._inferencer.model
        self.device = target

    def score_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any] | BaseException]:
        results: list[dict[str, Any] | BaseException] = [None] * len(batch)
        prompts = []
        divisors = []
        images = []
        valid_indices = []
        for index, request in enumerate(batch):
            try:
                image = _frozen._decoded_pickscore_image(request.get("solution_image"), request.get("extra_info"))
                score_divisor = request.get("score_divisor", 26.0)
                if score_divisor == 0:
                    raise ValueError("PickScore score_divisor must be non-zero")
                prompts.append(request.get("ground_truth") or "")
                divisors.append(score_divisor)
                images.append(image)
                valid_indices.append(index)
            except Exception as error:
                results[index] = error
        if not valid_indices:
            return results

        inferencer = self._require_inferencer()
        # The device context stays next to the forward, inside the scorer.
        device_name = get_device_name()
        if self.device.type == device_name and device_name != "cpu":
            device_context = get_torch_device().device(self.device)
        else:
            device_context = nullcontext()
        with device_context:
            output = inferencer.infer(prompts, images)
        for position, index in enumerate(valid_indices):
            raw_score = _frozen._pairwise_pickscore(
                output["text_embeddings"][position].unsqueeze(0),
                output["image_embeddings"][position].unsqueeze(0),
                output["logit_scale"],
                divisors[position],
            ).item()
            results[index] = {"score": raw_score, "pickscore_raw": raw_score}
        return results

    # TODO(zhtmike): drop both overrides once the migrated inferencer derives
    # placement from model parameters — tracked in #723 (post-window cleanup).
    def wake_up(self) -> None:
        super().wake_up()
        self._inferencer.device = self.device

    def sleep(self) -> None:
        super().sleep()
        self._inferencer.device = torch.device("cpu")

    def close(self) -> None:
        super().close()
        self._inferencer = None

    def _require_inferencer(self) -> _frozen._PickScoreInferencer:
        if self._inferencer is None:
            raise RuntimeError("PickScoreScorer has no weights; load(device) must run first")
        return self._inferencer


async def compute_score_pickscore_batched(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    reward_model,
    score_divisor: float = 26.0,
) -> dict:
    """Submit one sample to the batched PickScore scorer hosted by ``reward_model``.

    ``reward_model`` is the worker's native executor for a named model whose
    ``executor.model`` resolves to :class:`PickScoreScorer`; it groups requests
    into CLIP batches. Kept as a free-function entry point so existing config
    surfaces (``reward_functions.<key>.path/name`` + ``model=<name>``) select it
    unchanged.
    """
    return await reward_model.score(
        {
            "data_source": data_source,
            "solution_image": solution_image,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "score_divisor": score_divisor,
        }
    )
