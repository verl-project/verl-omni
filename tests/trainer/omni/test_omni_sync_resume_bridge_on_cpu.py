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
"""omni_sync resume-bridge ordering.

``AsyncOmni.sleep()`` sets an admission hold that ``wake_up()`` does not clear,
and the naive-backend ``update_weights`` restores weights without resuming
generation. Without the ``OmniPPOTrainerSync`` bridge the first ``generate()``
of init and of every step would block on the pause condition.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("verl_omni.trainer.omni.ray_omni_trainer")

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync  # noqa: E402

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync  # noqa: E402


class _HoldModel:
    """Emulates the engine-side admission hold + a recording checkpoint manager.

    ``sleep_replicas`` sets the hold; ``update_weights`` does not clear it;
    only ``resume_generation_replicas`` lifts it.
    """

    def __init__(self):
        self.calls: list[str] = []
        self.hold = False

    # -- CheckpointEngineManager surface (sync, as the sync trainer calls it) ----
    def update_weights(self, global_steps):
        del global_steps
        self.calls.append("update_weights")

    def sleep_replicas(self):
        self.calls.append("sleep_replicas")
        self.hold = True

    def resume_generation_replicas(self):
        self.calls.append("resume_generation_replicas")
        self.hold = False

    # -- trainer_base._load_checkpoint stand-in ---------------------------------
    def load_checkpoint(self):
        self.calls.append("load_checkpoint")

    # -- rollout generate stand-in ----------------------------------------------
    def generate(self):
        assert not self.hold, "generate() reached while admission is held"
        self.calls.append("generate")


def _trainer(model):
    trainer = object.__new__(OmniPPOTrainerSync)
    trainer.checkpoint_manager = model
    trainer.global_steps = 0
    trainer.timing_raw = {}
    return trainer


def test_on_init_end_resumes_after_update_weights():
    model = _HoldModel()
    _trainer(model).on_init_end()
    assert model.calls == ["update_weights", "resume_generation_replicas"]
    assert not model.hold


def test_on_step_end_resumes_after_update_weights():
    model = _HoldModel()
    _trainer(model).on_step_end()
    assert model.calls == ["update_weights", "resume_generation_replicas"]
    assert not model.hold


def test_init_sequence_generates_only_after_bridge_resume():
    """_setup tail -> _load_checkpoint -> on_init_end -> first generate (verl order)."""
    model = _HoldModel()
    trainer = _trainer(model)

    # trainer_base._setup() tail: sleep replicas to load the checkpoint.
    model.sleep_replicas()
    model.load_checkpoint()
    trainer.on_init_end()
    model.generate()  # must not raise: the bridge lifted the hold

    assert model.calls == [
        "sleep_replicas",
        "load_checkpoint",
        "update_weights",
        "resume_generation_replicas",
        "generate",
    ]


def test_step_sequence_generates_only_after_bridge_resume():
    """Per step: sample (generate) -> on_sample_end sleep -> update -> resume."""
    model = _HoldModel()
    trainer = _trainer(model)

    model.generate()
    model.sleep_replicas()  # on_sample_end inside _step_once
    trainer.on_step_end()  # actor update happened before this in fit()
    model.generate()  # next step's first action

    assert model.calls == [
        "generate",
        "sleep_replicas",
        "update_weights",
        "resume_generation_replicas",
        "generate",
    ]


def test_unbridged_parent_hooks_leave_hold_set():
    """Without the bridge (parent hooks alone) the hold survives and generate blocks."""
    model = _HoldModel()
    trainer = _trainer(model)

    model.sleep_replicas()
    PPOTrainerSync.on_init_end(trainer)
    PPOTrainerSync.on_step_end(trainer)
    assert model.hold, "parent hooks never resume generation"
    with pytest.raises(AssertionError, match="admission is held"):
        model.generate()


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

    assert calls == ["wake_up(tags=['weights'])", "resume_generation"]
