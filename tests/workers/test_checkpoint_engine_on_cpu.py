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

import pytest
from verl.checkpoint_engine import CheckpointEngineManager

from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager


def _manager_with_actor(actor_wg):
    manager = object.__new__(OmniCheckpointEngineManager)
    manager.actor_wg = actor_wg
    return manager


def test_lora_probe_allows_backend_without_capability():
    manager = _manager_with_actor(object())

    assert manager._fetch_actor_lora_peft_config() is None


def test_lora_probe_returns_first_configured_actor_result():
    expected = {"r": 8, "lora_alpha": 16}

    class ActorWorkerGroup:
        def get_lora_peft_config(self):
            return [None, expected]

    manager = _manager_with_actor(ActorWorkerGroup())

    assert manager._fetch_actor_lora_peft_config() is expected


def test_lora_probe_propagates_genuine_actor_failure():
    class ActorWorkerGroup:
        def get_lora_peft_config(self):
            raise RuntimeError("actor RPC failed")

    manager = _manager_with_actor(ActorWorkerGroup())

    with pytest.raises(RuntimeError, match="actor RPC failed"):
        manager._fetch_actor_lora_peft_config()


def test_naive_update_weights_resumes_generation_after_sync(monkeypatch):
    events: list[str] = []
    manager = object.__new__(OmniCheckpointEngineManager)
    manager.backend = "naive"
    manager.actor_wg = object()
    manager.replicas = []

    async def fake_super_update(self, global_steps=None):
        events.append(f"super_update({global_steps})")
        return {}

    async def fake_resume():
        events.append("resume_generation_replicas")

    monkeypatch.setattr(CheckpointEngineManager, "update_weights", fake_super_update)
    manager.resume_generation_replicas = fake_resume

    result = manager.update_weights(global_steps=3)

    assert result == {}
    assert events == ["super_update(3)", "resume_generation_replicas"]


def test_naive_update_weights_does_not_resume_when_sync_fails(monkeypatch):
    events: list[str] = []
    manager = object.__new__(OmniCheckpointEngineManager)
    manager.backend = "naive"
    manager.actor_wg = object()
    manager.replicas = []

    async def failing_super_update(self, global_steps=None):
        events.append(f"super_update({global_steps})")
        raise RuntimeError("weight sync failed")

    async def fake_resume():
        events.append("resume_generation_replicas")

    monkeypatch.setattr(CheckpointEngineManager, "update_weights", failing_super_update)
    manager.resume_generation_replicas = fake_resume

    with pytest.raises(RuntimeError, match="weight sync failed"):
        manager.update_weights(global_steps=1)

    assert events == ["super_update(1)"]


def test_non_naive_update_weights_returns_metrics_without_extra_resume(monkeypatch):
    events: list[str] = []
    expected_peft = {"r": 16}
    expected_metrics = {"sync_time": 0.42}

    class ActorWorkerGroup:
        def get_lora_peft_config(self):
            events.append("fetch_lora")
            return [expected_peft]

    manager = object.__new__(OmniCheckpointEngineManager)
    manager.backend = "omni_delta_sharded"
    manager.actor_wg = ActorWorkerGroup()
    manager.replicas = []

    async def fake_push(peft_config):
        events.append(f"push_lora({peft_config})")

    async def fake_super_update(self, global_steps=None):
        events.append(f"super_update({global_steps})")
        return expected_metrics

    async def fake_resume():
        events.append("resume_generation_replicas")

    manager._push_lora_peft_config_to_replicas = fake_push
    monkeypatch.setattr(CheckpointEngineManager, "update_weights", fake_super_update)
    manager.resume_generation_replicas = fake_resume

    result = manager.update_weights(global_steps=5)

    assert result == expected_metrics
    assert manager._lora_peft_config == expected_peft
    assert events == ["fetch_lora", f"push_lora({expected_peft})", "super_update(5)"]
