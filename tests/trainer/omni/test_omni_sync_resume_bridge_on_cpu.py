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
"""Rollout-side admission-resume ordering for omni naive weight sync.

``AsyncOmni.sleep()`` sets an admission hold that worker-side ``rollout.resume()``
does not clear, and upstream verl's naive-backend ``CheckpointEngineManager``
restores weights without resuming generation. ``OmniCheckpointEngineManager``
owns ``resume_generation_replicas()`` on the naive weight-sync boundary so
``OmniPPOTrainerSync`` and other trainers inherit the stock hooks without
duplicating per-trainer resume bridges (#517).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("verl_omni.trainer.omni.ray_omni_trainer")

from verl.checkpoint_engine import CheckpointEngineManager  # noqa: E402
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync  # noqa: E402

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync  # noqa: E402
from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager  # noqa: E402


class _FakeReplica:
    """Emulates a rollout replica with an engine-side admission hold."""

    def __init__(self, calls: list[str], state: dict[str, bool]):
        self.calls = calls
        self.state = state

    async def sleep(self):
        self.calls.append("sleep_replicas")
        self.state["hold"] = True

    async def resume_generation(self):
        self.calls.append("resume_generation_replicas")
        self.state["hold"] = False


class _FakeActorWG:
    """Emulates the actor worker group's naive weight sync (does not lift hold)."""

    def __init__(self, calls: list[str]):
        self.calls = calls

    def update_weights(self, global_steps=None, mode="naive"):
        del global_steps, mode

        async def _sync():
            self.calls.append("actor_update_weights")

        return [_sync()]


def _make_harness():
    calls: list[str] = []
    state = {"hold": False}
    manager = object.__new__(OmniCheckpointEngineManager)
    manager.backend = "naive"
    manager.actor_wg = _FakeActorWG(calls)
    manager.replicas = [_FakeReplica(calls, state)]

    trainer = object.__new__(OmniPPOTrainerSync)
    trainer.checkpoint_manager = manager
    trainer.global_steps = 0
    trainer.timing_raw = {}

    def load_checkpoint():
        calls.append("load_checkpoint")

    def generate():
        assert not state["hold"], "generate() reached while admission is held"
        calls.append("generate")

    return SimpleNamespace(
        trainer=trainer,
        manager=manager,
        calls=calls,
        state=state,
        load_checkpoint=load_checkpoint,
        generate=generate,
    )


def test_omni_sync_trainer_inherits_stock_hooks_without_bridge():
    """OmniPPOTrainerSync delegates on_init_end/on_step_end directly to PPOTrainerSync."""
    assert OmniPPOTrainerSync.on_init_end is PPOTrainerSync.on_init_end
    assert OmniPPOTrainerSync.on_step_end is PPOTrainerSync.on_step_end


def test_on_init_end_resumes_after_naive_weight_sync():
    h = _make_harness()
    h.manager.sleep_replicas()
    h.load_checkpoint()
    h.trainer.on_init_end()
    h.generate()

    assert not h.state["hold"]
    assert h.calls == [
        "sleep_replicas",
        "load_checkpoint",
        "actor_update_weights",
        "resume_generation_replicas",
        "generate",
    ]


def test_on_step_end_resumes_after_naive_weight_sync():
    h = _make_harness()
    h.generate()
    h.manager.sleep_replicas()
    h.trainer.on_step_end()
    h.generate()

    assert not h.state["hold"]
    assert h.calls == [
        "generate",
        "sleep_replicas",
        "actor_update_weights",
        "resume_generation_replicas",
        "generate",
    ]


def test_stock_verl_checkpoint_manager_leaves_hold_set():
    """Contrast: upstream CheckpointEngineManager alone never resumes naive admission."""
    calls: list[str] = []
    state = {"hold": False}
    stock_manager = object.__new__(CheckpointEngineManager)
    stock_manager.backend = "naive"
    stock_manager.actor_wg = _FakeActorWG(calls)
    stock_manager.replicas = [_FakeReplica(calls, state)]

    trainer = object.__new__(OmniPPOTrainerSync)
    trainer.checkpoint_manager = stock_manager
    trainer.global_steps = 0
    trainer.timing_raw = {}

    stock_manager.sleep_replicas()
    trainer.on_init_end()
    assert state["hold"], "upstream CheckpointEngineManager does not resume naive admission"


async def test_server_wake_up_resumes_admission_after_ack():
    """The server's wake_up re-opens admission once the wake ACKs.

    wake_up_replicas() only runs after the weight sync completes, so resuming
    there is never mid-sync; it is what lets every trainer generate after
    sleep/wake without its own resume bridge (the engine keeps the sleep hold
    through its own wake_up).
    """
    server_module = pytest.importorskip("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server")
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.node_rank = 0
    server.replica_rank = 0

    from verl.workers.rollout.replica import RolloutMode

    server.rollout_mode = RolloutMode.COLOCATED
    server.config = SimpleNamespace(free_cache_engine=True)
    server._lora_request_cache = None

    calls: list[str] = []

    async def wake_up(stage_ids=None, tags=None):
        calls.append(f"wake_up(tags={tags})")

    async def resume_generation(stage_ids=None):
        calls.append("resume_generation")

    server.engine = SimpleNamespace(wake_up=wake_up)
    server.engine.resume_generation = resume_generation

    await server.wake_up()

    assert calls == ["wake_up(tags=['kv_cache', 'weights'])", "resume_generation"]
