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
"""Opt-in real-Ray streaming integration with native Torch inference.

Set STREAMING_TEST_TOKENIZER to a local tokenizer checkpoint and expose one GPU.
Run this file explicitly; it starts a Ray cluster and uses native GPU actors.
"""

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.protocol import DataProto

from verl_omni.reward_loop.reward_loop import OmniRewardLoopWorker
from verl_omni.reward_loop.streaming import StreamingRewardClient
from verl_omni.workers.config.reward import RewardModelSpec, StreamingRewardConfig


class LinearModel:
    """Small real Torch model with observable inference and disposal boundaries."""

    def __init__(self, device, scale=1.0):
        self.device = device
        self.model = torch.nn.Linear(1, 1, bias=False).to(device)
        with torch.no_grad():
            self.model.weight.fill_(scale)
        self.calls = []
        self.inflight = 0

    async def infer(self, value, delay):
        """Run deterministic tensor inference after a controlled scheduling delay."""
        self.calls.append(value)
        self.inflight += 1
        try:
            await asyncio.sleep(delay)
            with torch.inference_mode():
                return self.model(torch.tensor([[value]], dtype=torch.float32, device=self.device)).item()
        finally:
            self.inflight -= 1

    def close(self):
        """Refuse disposal during live inference."""
        assert self.inflight == 0
        del self.model


async def linear_score(solution_image, reward_model, delay=0.0, fail=False):
    """Use the native inference handle through the production score manager."""
    if fail:
        raise ValueError("deliberate scorer failure")
    value = float(solution_image[0, 0, 0])
    return {"score": await reward_model.infer(value=value, delay=delay)}


class ObservableWorker(OmniRewardLoopWorker):
    """Production reward worker with a test-only inference observation method."""

    async def stats(self):
        """Report the model's actual accepted calls and active inference count."""
        model = self.native_reward_executors["linear"]._model
        return {"calls": model.calls, "inflight": model.inflight}


@pytest.fixture(scope="module", autouse=True)
def cluster():
    ray.init(
        num_cpus=6,
        num_gpus=1,
        include_dashboard=False,
        _temp_dir="/tmp/vosray",
        object_store_memory=256 * 1024 * 1024,
    )
    yield
    ray.shutdown()


def _worker(key, *, weight=1.0, scale=1.0, delay=0.0, fail=False, required=True):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.actor_rollout_ref.model.tokenizer_path = os.environ["STREAMING_TEST_TOKENIZER"]
    config.reward.reward_manager.name = "MultiVisualRewardManager"
    config.reward.models = OmegaConf.create({"linear": {"backend": "native"}})
    config.reward.reward_functions = OmegaConf.create(
        {
            key: {
                "model": "linear",
                "path": "pkg://test_streaming_reward_ray",
                "name": "linear_score",
                "weight": weight,
                "delay": delay,
                "fail": fail,
                "required": required,
            }
        }
    )
    spec = RewardModelSpec(
        name="linear",
        backend="native",
        executor_config={
            "model": "test_streaming_reward_ray:LinearModel",
            "kwargs": {"scale": scale},
        },
    )
    worker = ray.remote(ObservableWorker).options(num_cpus=1, num_gpus=0.1).remote(config, None, {"linear": spec})
    ray.get(worker.wake_up_reward_model.remote("linear"))
    return worker


def _sample(value):
    return DataProto.from_dict(
        tensors={"responses": torch.full((1, 3, 4, 4), value, dtype=torch.uint8)},
        non_tensors={
            "data_source": np.array(["real_ray"]),
            "reward_model": np.array([{"ground_truth": "test"}], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        },
    )


@pytest.mark.asyncio
async def test_real_ray_fanout_weighting_order_and_close():
    a = _worker("a", weight=0.7, scale=2.0, delay=0.01)
    b = _worker("b", weight=0.3, scale=3.0)
    try:
        client = StreamingRewardClient({"a": (a,), "b": (b,)}, StreamingRewardConfig(max_inflight=2))
        results = await asyncio.gather(*(client.compute_score(_sample(i)) for i in range(1, 7)))
        assert [item["reward_score"] for item in results] == pytest.approx([2.3 * i for i in range(1, 7)])
        for i, item in enumerate(results, 1):
            assert item["reward_extra_info"]["reward/a"] == 2 * i
            assert item["reward_extra_info"]["reward/b"] == 3 * i
        await client.close()
        for worker in [a, b]:
            assert (await worker.stats.remote()) == {"calls": list(range(1, 7)), "inflight": 0}
            await worker.close_reward_model.remote()
    finally:
        ray.kill(a)
        ray.kill(b)


@pytest.mark.parametrize("kind", ["cancel", "timeout", "required", "optional", "worker_death", "submission"])
@pytest.mark.asyncio
async def test_real_ray_failure_paths_drain_before_model_close(kind):
    a = _worker("slow", delay=0.25)
    b = _worker(
        "other",
        fail=kind in {"required", "optional"},
        required=kind != "optional",
        delay=0.5 if kind == "worker_death" else 0.0,
    )
    try:
        other = b
        if kind == "submission":

            def reject(data):
                raise RuntimeError("submission failed")

            other = SimpleNamespace(compute_score=SimpleNamespace(remote=reject))
        client = StreamingRewardClient(
            {"slow": (a,), "other": (other,)},
            StreamingRewardConfig(max_inflight=1, timeout=0.02 if kind == "timeout" else None),
        )
        task = asyncio.create_task(client.compute_score(_sample(2)))
        while not (await a.stats.remote())["calls"]:
            await asyncio.sleep(0.001)
        if kind == "cancel":
            task.cancel()
            task.cancel()
        if kind == "worker_death":
            ray.kill(b)
        if kind == "optional":
            result = await task
            assert result["reward_score"] == 2
            assert result["reward_extra_info"]["reward/other/errors"] == 1
        else:
            exception = {
                "cancel": asyncio.CancelledError,
                "timeout": TimeoutError,
                "submission": RuntimeError,
            }.get(kind, ray.exceptions.RayError)
            with pytest.raises(exception):
                await task
        assert (await a.stats.remote())["inflight"] == 0
        await client.close()
        await a.close_reward_model.remote()
        if kind != "worker_death":
            await b.close_reward_model.remote()
    finally:
        ray.kill(a)
        ray.kill(b)
