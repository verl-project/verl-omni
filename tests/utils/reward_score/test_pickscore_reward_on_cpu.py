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
"""CPU tests for PickScore burst batching and prompt deduplication."""

import asyncio
import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image


def _load_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/pickscore_reward.py"
    spec = importlib.util.spec_from_file_location("pickscore_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pickscore_reward = _load_module()


class _FakeProcessor:
    def __init__(self):
        self.text_batches = []

    def __call__(self, *, images=None, text=None, **kwargs):
        if images is not None:
            return {"pixel_values": torch.stack([image.feature for image in images])}

        self.text_batches.append(text)
        features = {"same": [1.0, 0.0], "different": [0.0, 1.0]}
        return {"input_ids": torch.tensor([features[prompt] for prompt in text])}


class _FakeModel:
    logit_scale = torch.tensor(0.0)

    def get_image_features(self, pixel_values):
        return pixel_values

    def get_text_features(self, input_ids):
        return input_ids


class _FeatureImage:
    def __init__(self, feature):
        self.feature = torch.tensor(feature)


def test_inferencer_uses_platform_device_by_default(monkeypatch):
    class _FakeLoadedModel:
        def __init__(self):
            self.to_calls = []

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return self

    model = _FakeLoadedModel()
    monkeypatch.setattr(pickscore_reward, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(pickscore_reward, "get_device_id", lambda: 2)
    monkeypatch.setattr(pickscore_reward.CLIPProcessor, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pickscore_reward.CLIPModel, "from_pretrained", lambda *_args, **_kwargs: model)

    inferencer = pickscore_reward._PickScoreInferencer()

    assert inferencer.device == torch.device("cuda", 2)
    assert model.to_calls[0] == ((torch.device("cuda", 2),), {})


def test_score_encodes_duplicate_prompts_once_and_preserves_pairing():
    inferencer = object.__new__(pickscore_reward._PickScoreInferencer)
    inferencer.device = "cpu"
    inferencer.processor = _FakeProcessor()
    inferencer.model = _FakeModel()

    scores = inferencer.score(
        ["same", "same", "different"],
        [_FeatureImage([1.0, 0.0]), _FeatureImage([0.0, 1.0]), _FeatureImage([0.0, 1.0])],
    )

    assert inferencer.processor.text_batches == [["same", "different"]]
    torch.testing.assert_close(scores, torch.tensor([1.0 / 26, 0.0, 1.0 / 26]))


class _FakeInferencer:
    def __init__(self):
        self.batches = []

    def score(self, prompts, images):
        self.batches.append((list(prompts), list(images)))
        return torch.tensor([float(image.getpixel((0, 0))) for image in images])

    def infer(self, prompts, images):
        self.batches.append((list(prompts), list(images)))
        values = torch.tensor([float(image.getpixel((0, 0))) for image in images])
        return {
            "text_embeddings": torch.stack((values, torch.ones_like(values)), dim=-1),
            "image_embeddings": torch.stack((torch.ones_like(values), values), dim=-1),
            "logit_scale": torch.tensor(2.0),
        }


@pytest.mark.asyncio
async def test_native_model_batches_inference_and_closes_instance_consumer(monkeypatch):
    inferencer = _FakeInferencer()
    init_kwargs = {}

    def build_inferencer(**kwargs):
        init_kwargs.update(kwargs)
        return inferencer

    monkeypatch.setattr(pickscore_reward, "_PickScoreInferencer", build_inferencer)
    model = pickscore_reward.PickScoreNativeModel(model_path="/models/pickscore", device="cpu")

    assert init_kwargs["model_path"] == "/models/pickscore"
    assert init_kwargs["processor_path"] == pickscore_reward._PROCESSOR_PATH

    result_batches = await asyncio.gather(
        *(model.infer(["shared prompt"], [Image.new("L", (1, 1), index)]) for index in range(4))
    )

    assert [batch[0]["text_embeddings"].tolist() for batch in result_batches] == [
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
    ]
    assert len(inferencer.batches) == 1
    assert inferencer.batches[0][0] == ["shared prompt"] * 4

    await model.close()
    assert not model._consumer_task
    assert not hasattr(model, "_inferencer")


@pytest.mark.asyncio
async def test_native_reward_function_computes_score_from_model_output():
    class _NativeModelHandle:
        async def infer(self, prompts, images):
            assert prompts == ["prompt"]
            assert len(images) == 1
            return [
                {
                    "text_embeddings": torch.tensor([3.0, 4.0]),
                    "image_embeddings": torch.tensor([0.0, 5.0]),
                    "logit_scale": torch.tensor(98.0),
                }
            ]

    result = await pickscore_reward.compute_score_pickscore_native(
        data_source="test",
        solution_image=torch.zeros(3, 2, 2, dtype=torch.uint8),
        ground_truth="prompt",
        extra_info={},
        reward_model=_NativeModelHandle(),
    )

    expected_score = 98.0 * 0.8 / 26.0
    assert result == {"score": pytest.approx(expected_score), "pickscore_raw": pytest.approx(expected_score)}


@pytest.mark.asyncio
async def test_consumer_batches_burst_requests_and_preserves_order(monkeypatch):
    inferencer = _FakeInferencer()
    queue = asyncio.Queue()
    monkeypatch.setattr(pickscore_reward, "_score_queue", queue)
    monkeypatch.setattr(pickscore_reward, "_consumer_started", False)
    monkeypatch.setattr(pickscore_reward, "_consumer_task", None)
    monkeypatch.setattr(pickscore_reward, "_consumer_lock", asyncio.Lock())
    monkeypatch.setattr(pickscore_reward, "_PickScoreInferencer", lambda device: inferencer)

    scores = await asyncio.gather(
        *(
            pickscore_reward.compute_score_pickscore(
                data_source="test",
                solution_image=Image.new("L", (1, 1), index),
                ground_truth="shared prompt",
                extra_info={},
                device="cpu",
            )
            for index in range(4)
        )
    )
    queue.put_nowait((None, None, None))
    await pickscore_reward._consumer_task

    assert scores == [
        {"score": 0.0, "pickscore_raw": 0.0},
        {"score": 1.0, "pickscore_raw": 1.0},
        {"score": 2.0, "pickscore_raw": 2.0},
        {"score": 3.0, "pickscore_raw": 3.0},
    ]
    assert len(inferencer.batches) == 1
    assert inferencer.batches[0][0] == ["shared prompt"] * 4


@pytest.mark.asyncio
async def test_consumer_caps_each_micro_batch(monkeypatch):
    inferencer = _FakeInferencer()
    queue = asyncio.Queue()
    monkeypatch.setattr(pickscore_reward, "_inferencer", inferencer)
    monkeypatch.setattr(pickscore_reward, "_score_queue", queue)
    monkeypatch.setattr(pickscore_reward, "_MAX_BATCH_SIZE", 2)

    consumer = asyncio.create_task(pickscore_reward._consumer_loop())
    futures = []
    for index in range(5):
        future = asyncio.get_running_loop().create_future()
        futures.append(future)
        queue.put_nowait(("prompt", Image.new("L", (1, 1), index), future))
    queue.put_nowait((None, None, None))

    assert await asyncio.gather(*futures) == [0.0, 1.0, 2.0, 3.0, 4.0]
    await consumer
    assert [len(prompts) for prompts, _ in inferencer.batches] == [2, 2, 1]


@pytest.mark.asyncio
async def test_consumer_isolates_invalid_image_from_batch(monkeypatch):
    inferencer = _FakeInferencer()
    queue = asyncio.Queue()
    monkeypatch.setattr(pickscore_reward, "_inferencer", inferencer)
    monkeypatch.setattr(pickscore_reward, "_score_queue", queue)

    consumer = asyncio.create_task(pickscore_reward._consumer_loop())
    futures = [asyncio.get_running_loop().create_future() for _ in range(3)]
    queue.put_nowait(("prompt", Image.new("L", (1, 1), 1), futures[0]))
    queue.put_nowait(("prompt", object(), futures[1]))
    queue.put_nowait(("prompt", Image.new("L", (1, 1), 3), futures[2]))
    queue.put_nowait((None, None, None))

    results = await asyncio.gather(*futures, return_exceptions=True)
    await consumer

    assert results[0] == 1.0
    assert isinstance(results[1], AssertionError)
    assert results[2] == 3.0
    assert len(inferencer.batches) == 1
    assert inferencer.batches[0][0] == ["prompt", "prompt"]


@pytest.mark.asyncio
async def test_engine_reward_posts_openai_embedding_payloads(monkeypatch):
    requests = []

    class _Response:
        def __init__(self, embedding):
            self.embedding = embedding

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return {"data": [{"embedding": self.embedding}]}

    class _Session:
        def __init__(self, **kwargs):
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, url, json):
            requests.append((url, json))
            return _Response([3.0, 4.0] if len(requests) == 1 else [0.0, 5.0])

    monkeypatch.setattr(pickscore_reward.aiohttp, "ClientSession", _Session)

    result = await pickscore_reward.compute_score_pickscore_engine(
        data_source="test",
        solution_image=torch.zeros(3, 2, 2, dtype=torch.uint8),
        ground_truth="prompt",
        extra_info={},
        reward_router_address="router:8000",
        model_name="pickscore",
        logit_scale=98.0,
    )

    expected_score = 98.0 * 0.8 / 26.0
    assert result == {"score": pytest.approx(expected_score), "pickscore_raw": pytest.approx(expected_score)}
    assert [url for url, _ in requests] == ["http://router:8000/v1/embeddings"] * 2
    assert requests[0][1] == {"model": "pickscore", "input": "prompt", "encoding_format": "float"}
    assert requests[1][1]["encoding_format"] == "float"
    assert requests[1][1]["input"][0]["content"][0]["type"] == "image_url"
