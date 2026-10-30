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
"""PickScore preference model as a class-based reward scorer.

All scoring math lives in ``pickscore_reward.py``; this module adds only the
:class:`RewardScorer` protocol glue — no queue, consumer task, or threading.
The two exports pair up in config: :class:`PickScoreScorer` is hosted by a
native named model (``reward.models.<name>.executor.model``), while
:func:`compute_score_pickscore_batched` is the reward function
(``reward_functions.<key>.path``/``.name`` + ``model=<name>``) submitting one
sample at a time to that host.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from verl.utils.device import get_device_name, get_torch_device

from verl_omni.utils.reward_score import pickscore_reward as _frozen
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import register_scorer

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["PickScoreScorer", "compute_score_pickscore_batched"]


@register_scorer("pickscore")
class PickScoreScorer(RewardScorer):
    """PickScore preference model as a batch-native scorer.

    Args:
        model_path: PickScore checkpoint to load (default
            ``yuvalkirstain/PickScore_v1``); the named model's top-level
            ``model_path`` forwards here unless ``executor.kwargs`` sets one.
        processor_path: CLIP processor for prompt/image preprocessing
            (default ``laion/CLIP-ViT-H-14-laion2B-s32B-b79K``).
        dtype: weights dtype at load (default ``torch.float32``).
    """

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
        self._inferencer = _frozen._PickScoreInferencer(
            device=device,
            dtype=self.dtype,
            model_path=self.model_path,
            processor_path=self.processor_path,
        )
        self.model = self._inferencer.model
        self.device = self._inferencer.device

    def score_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any] | Exception]:
        results: list[dict[str, Any] | Exception] = [None] * len(batch)
        packed: list[tuple[str, float, Image.Image]] = []
        valid_indices: list[int] = []
        for index, request in enumerate(batch):
            try:
                packed.append(self._prepare_request(request))
            except Exception as error:
                results[index] = error
            else:
                valid_indices.append(index)
        if not valid_indices:
            return results

        prompts, divisors, images = map(list, zip(*packed, strict=True))
        output = self._forward(prompts, images)
        for position, index in enumerate(valid_indices):
            raw_score = _frozen._pairwise_pickscore(
                output["text_embeddings"][position].unsqueeze(0),
                output["image_embeddings"][position].unsqueeze(0),
                output["logit_scale"],
                divisors[position],
            ).item()
            results[index] = {"score": raw_score, "pickscore_raw": raw_score}
        return results

    @staticmethod
    def _prepare_request(request: dict[str, Any]) -> tuple[str, float, Image.Image]:
        """Decode one request into a ``(prompt, score_divisor, image)`` triple."""
        image = _frozen.decoded_reward_image(
            request.get("solution_image"), request.get("extra_info"), scorer="PickScore"
        )
        score_divisor = request.get("score_divisor", 26.0)
        if isinstance(score_divisor, bool) or not isinstance(score_divisor, int | float):
            raise ValueError(f"PickScore score_divisor must be a number, got {score_divisor!r}")
        if score_divisor == 0:
            raise ValueError("PickScore score_divisor must be non-zero")
        return request.get("ground_truth") or "", score_divisor, image

    def _forward(self, prompts: list[str], images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """Run one batched CLIP forward under the accelerator device context."""
        inferencer = self._require_inferencer()
        device_name = get_device_name()
        if self.device.type == device_name and device_name != "cpu":
            device_context = get_torch_device().device(self.device)
        else:
            device_context = nullcontext()
        with device_context:
            return inferencer.infer(prompts, images)

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

    ``reward_model`` is the native executor of the named model whose
    ``executor.model`` resolves to :class:`PickScoreScorer`; concurrent
    submissions are packed into CLIP batches. Configured via
    ``reward_functions.<key>.path`` and ``.name`` plus ``model=<name>``.
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
