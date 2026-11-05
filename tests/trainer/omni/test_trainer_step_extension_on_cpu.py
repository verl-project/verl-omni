# Copyright 2026 verl-omni contributors
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

"""Native separate-async trainer routing for optional step extensions."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

from verl_omni.pipelines.proximal_reuse import rpc as proximal_rpc
from verl_omni.pipelines.proximal_reuse.strategy import ProximalActorReuse
from verl_omni.pipelines.step_extension import resolve_step_extension
from verl_omni.trainer.omni.ray_omni_trainer_separate_async import OmniPPOTrainerSeparateAsync


@pytest.fixture
def native_trainer(monkeypatch):
    calls = []

    def native(name):
        def call(_self, batch, metrics):
            calls.append((name, batch, metrics))
            return batch

        return call

    monkeypatch.setattr(PPOTrainerSeparateAsync, "_compute_old_log_prob", native("old"))
    monkeypatch.setattr(PPOTrainer, "_compute_advantage", native("advantage"))
    monkeypatch.setattr(PPOTrainer, "_update_actor", native("update"))
    trainer = object.__new__(OmniPPOTrainerSeparateAsync)
    trainer._step_extension = None
    return trainer, calls


def test_default_uses_native_methods_and_arguments(native_trainer):
    trainer, calls = native_trainer
    batch, metrics = object(), {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert trainer._compute_advantage(batch, metrics) is batch
    assert trainer._update_actor(batch, metrics) is batch
    assert calls == [("old", batch, metrics), ("advantage", batch, metrics), ("update", batch, metrics)]


def test_critic_warmup_falls_back_before_preparation(native_trainer):
    trainer, calls = native_trainer
    trainer.config = OmegaConf.create({"trainer": {"v1": {"reuse_proximal_actor_forward": True}, "critic_warmup": 2}})
    trainer.global_steps = 1
    trainer._step_extension = resolve_step_extension(trainer)
    assert isinstance(trainer._step_extension, ProximalActorReuse)
    batch, metrics = object(), {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert trainer._compute_advantage(batch, metrics) is batch
    assert trainer._update_actor(batch, metrics) is batch
    assert [name for name, _, _ in calls] == ["old", "advantage", "update"]


def test_unknown_partial_update_does_not_retry_native(native_trainer):
    trainer, calls = native_trainer
    strategy = ProximalActorReuse(trainer)
    strategy._omni_prepared_actor = object()

    def fail(*args):
        raise RuntimeError("partial update")

    strategy._update_proximal_actor = fail
    trainer._step_extension = strategy
    with pytest.raises(RuntimeError, match="partial update"):
        trainer._update_actor(object(), {})
    with pytest.raises(RuntimeError, match="has not finished"):
        trainer._compute_old_log_prob(object(), {})
    assert calls == []


@pytest.mark.parametrize("value", [None, "true", 1])
def test_invalid_opt_in_is_rejected(value):
    config = OmegaConf.create({"trainer": {"v1": {"reuse_proximal_actor_forward": value}}})
    with pytest.raises(ValueError, match="boolean"):
        resolve_step_extension(SimpleNamespace(config=config))


def test_absent_opt_in_selects_no_extension():
    config = OmegaConf.create({"trainer": {"v1": {}}})
    assert resolve_step_extension(SimpleNamespace(config=config)) is None


def test_named_step_extension_selects_proximal_reuse():
    config = OmegaConf.create({"trainer": {"v1": {"step_extension": "proximal_actor_reuse"}}})
    assert isinstance(resolve_step_extension(SimpleNamespace(config=config)), ProximalActorReuse)


@pytest.mark.parametrize("value", ["not_registered", "", 42, True])
def test_unknown_or_invalid_named_extension_is_rejected(value):
    config = OmegaConf.create({"trainer": {"v1": {"step_extension": value}}})
    with pytest.raises(ValueError, match="extension"):
        resolve_step_extension(SimpleNamespace(config=config))


@pytest.mark.parametrize("in_metadata", [True, False])
@pytest.mark.parametrize("phase", ["preparation", "update"])
def test_reserved_request_rejected_without_dispatch(monkeypatch, in_metadata, phase):
    data = TensorDict({"input_ids": torch.ones(1, 1)}, batch_size=[1])
    if not in_metadata:
        data[proximal_rpc.REQUEST] = torch.zeros(1, 1)
    metadata = KVBatchMeta(keys=["row"], tags=[{}], partition_id="train")
    if in_metadata:
        metadata.extra_info[proximal_rpc.REQUEST] = "injected"
    trainer = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=object()), actor_rollout_wg=None)
    strategy = ProximalActorReuse(trainer)
    strategy._check_proximal_algorithm = lambda batch: None
    monkeypatch.setattr(proximal_rpc, "actor_execution", lambda config: {})
    monkeypatch.setattr("verl_omni.pipelines.proximal_reuse.strategy.tq.kv_batch_get", lambda **kwargs: data)
    with pytest.raises(ValueError, match="Caller-supplied actor reuse request"):
        if phase == "preparation":
            strategy._prepare_proximal_actor(metadata)
        else:
            strategy._update_proximal_actor(metadata, {}, object())
