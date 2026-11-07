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

"""CPU tests for the composite image preference reward."""

import asyncio
import threading

import pytest
import torch
from PIL import Image

from verl_omni.utils.reward_score.image_preference_utils import RequestBatcher
from verl_omni.utils.reward_score.preference_mix_reward import PreferenceMixNativeModel, compute_score_preference_mix


class _FakeRewardModel:
    async def infer(self, prompts, images):
        assert prompts == ["a red cube"]
        assert len(images) == 1 and isinstance(images[0], Image.Image)
        return [
            {
                "hpsv2_text_embedding": torch.tensor([1.0, 0.0]),
                "hpsv2_image_embedding": torch.tensor([0.6, 0.8]),
                "pickscore_text_embedding": torch.tensor([1.0, 0.0]),
                "pickscore_image_embedding": torch.tensor([0.0, 1.0]),
                "pickscore_logit_scale": torch.tensor(26.0),
                "clipscore_text_embedding": torch.tensor([1.0, 0.0]),
                "clipscore_image_embedding": torch.tensor([0.8, 0.6]),
            }
        ]


_IMAGE = torch.zeros((3, 2, 2), dtype=torch.uint8)
_IMAGE_INFO = {"media_kind": "image"}


@pytest.mark.asyncio
async def test_preference_mix_applies_component_formulas_and_weights() -> None:
    result = await compute_score_preference_mix(
        data_source="pickapic_v1",
        solution_image=_IMAGE,
        ground_truth="a red cube",
        extra_info=_IMAGE_INFO,
        reward_model=_FakeRewardModel(),
        hpsv2_weight=1.0,
        pickscore_weight=2.0,
        clipscore_weight=3.0,
    )

    assert result["hpsv2_raw"] == pytest.approx(0.6)
    assert result["pickscore_raw"] == pytest.approx(0.0)
    assert result["clipscore_raw"] == pytest.approx(2.0)
    assert result["score"] == pytest.approx(6.6)


@pytest.mark.asyncio
async def test_preference_mix_rejects_zero_pickscore_divisor() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        await compute_score_preference_mix(
            data_source="pickapic_v1",
            solution_image=_IMAGE,
            ground_truth="a red cube",
            extra_info=_IMAGE_INFO,
            reward_model=_FakeRewardModel(),
            pickscore_divisor=0,
        )


@pytest.mark.asyncio
async def test_preference_mix_rejects_undeclared_media() -> None:
    from verl_omni.pipelines.rollout_media import ArtifactContractError

    with pytest.raises(ArtifactContractError):
        await compute_score_preference_mix(
            data_source="pickapic_v1",
            solution_image=_IMAGE,
            ground_truth="a red cube",
            extra_info={},
            reward_model=_FakeRewardModel(),
        )


class _HalfPrecisionRewardModel:
    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        self.text = torch.randn(1024, generator=generator).half()
        self.image = (self.text.float() + torch.randn(1024, generator=generator)).half()

    async def infer(self, prompts, images):
        return [
            {
                "hpsv2_text_embedding": self.text,
                "hpsv2_image_embedding": self.image,
                "pickscore_text_embedding": torch.tensor([1.0, 0.0]),
                "pickscore_image_embedding": torch.tensor([0.0, 1.0]),
                "pickscore_logit_scale": torch.tensor(26.0),
                "clipscore_text_embedding": torch.tensor([1.0, 0.0]),
                "clipscore_image_embedding": torch.tensor([0.8, 0.6]),
            }
        ]


@pytest.mark.asyncio
async def test_preference_mix_hpsv2_cosine_is_float32_for_half_embeddings() -> None:
    model = _HalfPrecisionRewardModel()

    result = await compute_score_preference_mix(
        data_source="test",
        solution_image=_IMAGE,
        ground_truth="a red cube",
        extra_info=_IMAGE_INFO,
        reward_model=model,
    )

    expected = torch.nn.functional.cosine_similarity(model.text.float(), model.image.float(), dim=0).item()
    assert result["hpsv2_raw"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.asyncio
async def test_preference_mix_cancellation_cancels_requests_and_close_waits_for_inference() -> None:
    from concurrent.futures import ThreadPoolExecutor

    model = PreferenceMixNativeModel.__new__(PreferenceMixNativeModel)
    model.device = torch.device("cpu")
    model._executor = ThreadPoolExecutor(max_workers=1)
    model._closed = False
    model.hpsv2 = model.pickscore = model.clipscore = object()
    started, release = threading.Event(), threading.Event()
    encoders_seen_at_end = []

    def blocking_infer(requests):
        started.set()
        release.wait(timeout=5)
        encoders_seen_at_end.append(hasattr(model, "clipscore"))
        return [{} for _ in requests]

    model._batcher = RequestBatcher(blocking_infer, max_batch_size=16, executor=model._executor)
    request = asyncio.create_task(model.infer(["a red cube"], [Image.new("RGB", (2, 2))]))
    while not started.is_set():
        await asyncio.sleep(0.01)

    consumer = model._batcher.task
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert consumer.cancelled()

    closing = asyncio.create_task(model.close())
    await asyncio.sleep(0.05)
    assert not closing.done()
    release.set()
    await closing
    assert encoders_seen_at_end == [True]
    assert not hasattr(model, "clipscore")
