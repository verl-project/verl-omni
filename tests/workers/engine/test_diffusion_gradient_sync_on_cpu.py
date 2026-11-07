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
"""CPU tests for deferred FSDP gradient sync on the diffusion actor update."""

from contextlib import contextmanager
from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.utils.config import validate_config
from verl_omni.workers.engine.fsdp import diffusers_impl
from verl_omni.workers.engine_workers import ActorRolloutRefWorker


class _FSDP2Module:
    def __init__(self):
        self.events = []

    def set_requires_gradient_sync(self, enabled):
        self.events.append(enabled)


def _engine(monkeypatch):
    engine = object.__new__(diffusers_impl.PPODiffusersFSDPEngine)
    engine.ulysses_sequence_parallel_size = 1
    engine.get_data_parallel_group = lambda: None
    engine.postprocess_batch_func = lambda output_lst, indices, data: output_lst
    engine.forward_step = lambda micro_batch, loss_function, forward_only, step: (
        torch.tensor(1.0, requires_grad=True),
        {"model_output": {}, "loss": 1.0, "metrics": {}},
    )
    sync_states = []

    @contextmanager
    def record_sync(*, is_last_micro_batch):
        sync_states.append(is_last_micro_batch)
        yield

    engine._gradient_sync_context = record_sync
    monkeypatch.setattr(diffusers_impl, "get_device_id", lambda: "cpu")
    return engine, sync_states


def _timestep_batch(n_micro=2, n_steps=3):
    micro_batches = [TensorDict({"all_timesteps": torch.zeros(1, n_steps)}, batch_size=[1]) for _ in range(n_micro)]
    data = TensorDict({"all_timesteps": torch.zeros(n_micro, n_steps)}, batch_size=[n_micro])
    return data, micro_batches


def test_timestep_loop_defers_until_last_pair(monkeypatch):
    engine, sync_states = _engine(monkeypatch)
    data, micro_batches = _timestep_batch()
    tu.assign_non_tensor(data, use_no_sync_for_gradient_accumulation=True)
    monkeypatch.setattr(diffusers_impl, "prepare_micro_batches", lambda **_: (micro_batches, None))

    engine._run_forward_backward_batch(
        data, loss_function=lambda **_: None, forward_only=False, timesteps_key="all_timesteps"
    )

    assert sync_states == [False, False, False, False, False, True]


def test_flag_off_never_enters_sync_context(monkeypatch):
    engine, sync_states = _engine(monkeypatch)
    data, micro_batches = _timestep_batch()
    monkeypatch.setattr(diffusers_impl, "prepare_micro_batches", lambda **_: (micro_batches, None))

    engine._run_forward_backward_batch(
        data, loss_function=lambda **_: None, forward_only=False, timesteps_key="all_timesteps"
    )

    assert sync_states == []


def test_forward_only_never_enters_sync_context(monkeypatch):
    engine, sync_states = _engine(monkeypatch)
    data, micro_batches = _timestep_batch()
    tu.assign_non_tensor(data, use_no_sync_for_gradient_accumulation=True)
    monkeypatch.setattr(diffusers_impl, "prepare_micro_batches", lambda **_: (micro_batches, None))

    engine._run_forward_backward_batch(
        data, loss_function=lambda **_: None, forward_only=True, timesteps_key="all_timesteps"
    )

    assert sync_states == []


def test_gradient_sync_context_delegates_fsdp2(monkeypatch):
    engine = object.__new__(diffusers_impl.PPODiffusersFSDPEngine)
    engine.module = _FSDP2Module()
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.fsdp_version", lambda _: 2)

    with engine._gradient_sync_context(is_last_micro_batch=False):
        engine.module.events.append("backward")

    assert engine.module.events == [False, "backward", True]


def test_gradient_sync_context_restores_fsdp2_after_error(monkeypatch):
    engine = object.__new__(diffusers_impl.PPODiffusersFSDPEngine)
    engine.module = _FSDP2Module()
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.fsdp_version", lambda _: 2)

    with pytest.raises(RuntimeError, match="backward failed"):
        with engine._gradient_sync_context(is_last_micro_batch=False):
            raise RuntimeError("backward failed")

    assert engine.module.events == [False, True]


def test_gradient_sync_context_keeps_sync_on_last_step(monkeypatch):
    engine = object.__new__(diffusers_impl.PPODiffusersFSDPEngine)
    engine.module = _FSDP2Module()
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.fsdp_version", lambda _: 2)

    with engine._gradient_sync_context(is_last_micro_batch=True):
        engine.module.events.append("backward")

    assert engine.module.events == ["backward"]


@pytest.mark.parametrize("enabled", [False, True])
def test_hydra_actor_forwards_no_sync_flag(enabled):
    config_dir = Path(verl_omni.__file__).parent / "trainer/config"
    overrides = ["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2"]
    if enabled:
        overrides.append("actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=true")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="diffusion_trainer", overrides=overrides)
    validate_config(cfg)
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    assert actor.use_no_sync_for_gradient_accumulation is enabled

    received = []

    def train_mini_batch(data):
        received.append(tu.get_non_tensor_data(data, "use_no_sync_for_gradient_accumulation", default=None))
        return None

    worker = SimpleNamespace(config=cfg.actor_rollout_ref, actor=SimpleNamespace(train_mini_batch=train_mini_batch))
    batch = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor(batch, use_no_sync_for_gradient_accumulation=not enabled)
    unwrap(ActorRolloutRefWorker.update_actor)(worker, batch)
    assert received == [enabled]
