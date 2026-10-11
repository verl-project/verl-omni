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
"""Real-Ray regressions for failed shutdown and late prompt registration."""

import asyncio
from types import SimpleNamespace

import pytest
import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopManagerTQ, DiffusionAgentLoopWorkerTQ
from verl_omni.reward_loop.streaming import StreamingRewardClient
from verl_omni.trainer.main_diffusion_v1 import DiffusionTaskRunnerV1
from verl_omni.workers.config.reward import StreamingRewardConfig


@ray.remote(num_cpus=0)
class GatedAgent(DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class):
    """Gate registration/close while retaining production prompt cancellation."""

    def __init__(self):
        self.rollout_config = OmegaConf.create(
            {"pipeline": {}, "algo": {}, "calculate_log_probs": False, "agent": {"default_agent_loop": "test"}}
        )
        self.streaming_reward_client = StreamingRewardClient({}, StreamingRewardConfig())
        self.background_tasks = set()
        self.gate = asyncio.Event()
        self.prompt_gate = asyncio.Event()
        self.blocked = False
        self.drained = False
        self.cancelled_prompts = 0

    async def generate_sequences(self, batch):
        """Delay entry to the production registration path."""
        self.blocked = True
        await self.gate.wait()
        await super().generate_sequences(batch)

    async def _run_prompt(self, *args, **kwargs):
        """Keep a registered prompt alive until production close cancels it."""
        try:
            await self.prompt_gate.wait()
        finally:
            self.cancelled_prompts += 1

    async def close_reward_streaming(self):
        """Expose whether the production prompt drain actually finished."""
        self.blocked = True
        await self.gate.wait()
        await super().close_reward_streaming()
        self.drained = True

    async def state(self):
        """Read registration and drain state without releasing the operation."""
        return {
            "blocked": self.blocked,
            "drained": self.drained,
            "prompts": len(self.background_tasks),
            "cancelled": self.cancelled_prompts,
        }

    async def release(self):
        """Let the pending registration or close finish."""
        self.gate.set()


@ray.remote(num_cpus=0)
class FailingAgent:
    """Fail after the other agent has entered the controlled interleaving."""

    def __init__(self, peer):
        self.peer = peer
        self.failed = False

    async def _fail(self):
        while not (await self.peer.state.remote())["blocked"]:
            await asyncio.sleep(0.01)
        self.failed = True
        raise RuntimeError("agent RPC failed")

    async def generate_sequences(self, batch):
        """Reject this chunk while the other chunk is registering prompts."""
        await self._fail()

    async def close_reward_streaming(self):
        """Reject close while the other agent is still draining."""
        await self._fail()

    async def has_failed(self):
        """Signal that the failure has happened before releasing the peer."""
        return self.failed


@pytest.fixture(scope="module", autouse=True)
def cluster():
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir="/tmp/vosr-acceptance",
        object_store_memory=128 * 1024 * 1024,
    )
    yield
    ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["generation", "close"])
async def test_failed_rpc_waits_for_peer_before_returning(monkeypatch, operation):
    peer = GatedAgent.remote()
    failed = FailingAgent.remote(peer)
    events = []

    async def close_models():
        events.append("models_closed")

    runner = DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
    manager = object.__new__(DiffusionAgentLoopManagerTQ)
    manager.agent_loop_workers = [failed, peer]
    trainer = SimpleNamespace(
        init=lambda: None,
        fit=lambda _: None,
        reward_loop_manager=SimpleNamespace(
            multi_reward_model_manager=SimpleNamespace(close_native_models=close_models)
        ),
    )
    runner.agent_loop_manager = manager
    monkeypatch.setattr(runner, "init_agent_loop_manager", lambda: None)
    monkeypatch.setattr("verl_omni.trainer.diffusion.v1.get_diffusion_trainer_cls", lambda mode: lambda config: trainer)
    monkeypatch.setattr(tq, "init", lambda config: events.append("tq_initialized"))
    monkeypatch.setattr(tq, "close", lambda: events.append("tq_closed"))
    config = OmegaConf.create(
        {
            "trainer": {"v1": {"trainer_mode": "sync"}},
            "transfer_queue": {"enable": True},
            "reward": {
                "streaming": {"_target_": "verl_omni.workers.config.reward.StreamingRewardConfig", "enabled": True}
            },
        }
    )
    if operation == "generation":
        prompts = TensorDict({"uid": NonTensorData("prompt")}, batch_size=[2])
        task = asyncio.create_task(asyncio.to_thread(manager.generate_sequences, prompts))
    else:
        task = asyncio.create_task(asyncio.to_thread(runner.run, config))
    try:
        async with asyncio.timeout(30):
            while not await failed.has_failed.remote():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            assert not task.done()
            assert "models_closed" not in events and "tq_closed" not in events
            await peer.release.remote()
            with pytest.raises(ray.exceptions.RayTaskError, match="agent RPC failed"):
                await task
            if operation == "generation":
                assert (await peer.state.remote())["prompts"] == 1
                await peer.close_reward_streaming.remote()
                state = await peer.state.remote()
                assert state["drained"] and state["prompts"] == 0 and state["cancelled"] == 1
            else:
                assert (await peer.state.remote())["drained"]
                assert events == ["tq_initialized"]
    finally:
        await peer.release.remote()
        await asyncio.gather(task, return_exceptions=True)
        ray.kill(failed)
        ray.kill(peer)
