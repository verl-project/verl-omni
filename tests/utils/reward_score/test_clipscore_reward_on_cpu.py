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

"""CPU tests for CLIPScore formula and burst batching."""

import asyncio

import pytest
import torch

from verl_omni.utils.reward_score import clipscore_reward


def test_pairwise_clipscore_clamps_negative_cosine() -> None:
    text = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    image = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    score = clipscore_reward.pairwise_clipscore(text, image)

    torch.testing.assert_close(score, torch.tensor([2.5, 0.0]))


@pytest.mark.parametrize("pairwise", ["clipscore", "pickscore"])
def test_pairwise_scores_are_float32_for_bf16_embeddings(pairwise) -> None:
    from verl_omni.utils.reward_score import pickscore_reward

    generator = torch.Generator().manual_seed(0)
    text = torch.randn(64, 768, generator=generator).bfloat16()
    image = (text.float() + torch.randn(64, 768, generator=generator)).bfloat16()

    if pairwise == "clipscore":
        score = clipscore_reward.pairwise_clipscore(text, image)
    else:
        score = pickscore_reward._pairwise_pickscore(text, image, torch.tensor(100.0, dtype=torch.bfloat16))

    assert score.dtype == torch.float32
    assert score.unique().numel() == 64


class _FakeInferencer:
    def __init__(self):
        self.batches = []

    def score(self, prompts, images, scale_factor=2.5):
        self.batches.append((list(prompts), list(images), scale_factor))
        return torch.tensor([float(image.getpixel((0, 0))[0]) * scale_factor for image in images])


@pytest.fixture
def fake_inferencer(monkeypatch):
    inferencer = _FakeInferencer()
    monkeypatch.setattr(clipscore_reward, "_inferencer", inferencer)
    monkeypatch.setattr(clipscore_reward, "_inferencer_key", ("cpu", "clip"))
    monkeypatch.setattr(clipscore_reward, "_batcher", None)
    monkeypatch.setattr(clipscore_reward, "_batcher_loop", None)
    return inferencer


async def _score(index: int, scale_factor: float = 2.0) -> dict[str, float]:
    return await clipscore_reward.compute_score_clipscore(
        data_source="test",
        solution_image=torch.full((3, 1, 1), index, dtype=torch.uint8),
        ground_truth="shared prompt",
        extra_info={"media_kind": "image"},
        device="cpu",
        model_path="clip",
        scale_factor=scale_factor,
    )


async def _stop_consumer() -> None:
    await clipscore_reward._batcher.stop()


@pytest.mark.asyncio
async def test_compute_score_batches_burst_requests(fake_inferencer) -> None:
    inferencer = fake_inferencer

    scores = await asyncio.gather(*(_score(index) for index in range(4)))
    await _stop_consumer()

    assert scores == [
        {"score": 0.0, "clipscore_raw": 0.0},
        {"score": 2.0, "clipscore_raw": 2.0},
        {"score": 4.0, "clipscore_raw": 4.0},
        {"score": 6.0, "clipscore_raw": 6.0},
    ]
    assert len(inferencer.batches) == 1
    assert inferencer.batches[0][0] == ["shared prompt"] * 4


def test_compute_score_serves_requests_from_successive_event_loops(fake_inferencer) -> None:
    async def score_once(index):
        return await asyncio.wait_for(_score(index), timeout=5)

    assert asyncio.run(score_once(1)) == {"score": 2.0, "clipscore_raw": 2.0}
    assert asyncio.run(score_once(2)) == {"score": 4.0, "clipscore_raw": 4.0}


@pytest.mark.asyncio
async def test_compute_score_restarts_a_stopped_consumer(fake_inferencer) -> None:
    assert await _score(1) == {"score": 2.0, "clipscore_raw": 2.0}
    consumer = clipscore_reward._batcher.task
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert await asyncio.wait_for(_score(3), timeout=5) == {"score": 6.0, "clipscore_raw": 6.0}
    await _stop_consumer()
