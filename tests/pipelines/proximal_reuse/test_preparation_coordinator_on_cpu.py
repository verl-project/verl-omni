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

"""Coordinator contracts using worker doubles; no Ray, model or real TQ."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl_omni.pipelines.proximal_reuse import rpc
from verl_omni.pipelines.proximal_reuse import strategy as trainer_module
from verl_omni.workers.utils.parameter_revision import ParameterVersion


class WorkerGroup:
    def __init__(self, *, hidden_error=False, failed_install=False):
        self.calls = []
        self.hidden_error = hidden_error
        self.failed_install = failed_install

    def _query_dispatch_info(self, mesh):
        assert mesh == "actor"
        return [0, 1, 0, 1]

    def stage_step_extension(self, metadata):
        assert metadata.extra_info.pop("_step_extension_selector") == ("proximal_actor_reuse", "prepare")
        return self.stage_proximal_actor(metadata)

    def control_step_extension(self, name, operation, payload=None):
        assert name == "proximal_actor_reuse"
        if operation == "get_preparation":
            return self.get_staged_proximal_actor()
        if operation == "discard_preparation":
            return self.discard_staged_proximal_actor(payload)
        if operation == "install":
            return self.install_proximal_actor(payload)
        raise AssertionError(operation)

    def stage_proximal_actor(self, metadata):
        self.calls.append("stage")
        self.request = metadata.extra_info[rpc.REQUEST]
        assert metadata.fields is None
        return [object() for _ in range(4)]

    def get_staged_proximal_actor(self):
        self.calls.append("collect-all")
        replies = []
        for rank, dp in enumerate(self.request.dp_mapping):
            binding = self.request.binding.shard([dp])
            receipt = rpc.RankReceipt(rank, dp, ParameterVersion(f"worker-{rank}", 0), binding, "d" * 64)
            replies.append({"request": self.request, "receipt": receipt})
        if self.hidden_error:
            replies[2] = {"error": "hidden replica rejected"}
        return replies

    def install_proximal_actor(self, prepared):
        self.calls.append("install-all")
        self.prepared = prepared
        return [True, True, not self.failed_install, True]

    def discard_staged_proximal_actor(self, cohort_id):
        self.calls.append("discard-all")
        assert cohort_id == self.request.cohort_id
        return [True] * 4


def setup_trainer(monkeypatch, **kwargs):
    full = TensorDict({"input_ids": torch.tensor([[1], [2]])}, batch_size=[2])
    monkeypatch.setattr(trainer_module.tq, "kv_batch_get", lambda **kwargs: full)
    monkeypatch.setattr(trainer_module.ray, "wait", lambda refs, **kwargs: (refs[:1], refs[1:]))
    monkeypatch.setattr(trainer_module.ray, "get", lambda ref, **kwargs: True)
    monkeypatch.setattr(rpc, "actor_execution", lambda config: {"temperature": 1.0})
    trainer = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(calculate_log_probs=False))),
        actor_rollout_wg=WorkerGroup(**kwargs),
    )
    strategy = trainer_module.ProximalActorReuse(trainer)
    strategy._check_proximal_algorithm = lambda batch: None
    batch = KVBatchMeta(keys=["k0", "k1"], tags=[{}, {}], partition_id="train")
    return strategy, batch


def test_coordinator_installs_only_after_all_rank_observations(monkeypatch):
    trainer, batch = setup_trainer(monkeypatch)
    prepared = trainer._prepare_proximal_actor(batch)
    assert trainer.trainer.actor_rollout_wg.calls == ["stage", "collect-all", "install-all"]
    assert prepared == trainer.trainer.actor_rollout_wg.prepared
    assert tuple(receipt.rank for receipt in prepared.ranks) == (0, 1, 2, 3)


def test_hidden_replica_rejection_prevents_installation(monkeypatch):
    trainer, batch = setup_trainer(monkeypatch, hidden_error=True)
    with pytest.raises(ValueError, match="hidden replica rejected"):
        trainer._prepare_proximal_actor(batch)
    assert trainer.trainer.actor_rollout_wg.calls == ["stage", "collect-all", "discard-all"]


def test_partial_install_is_an_error_not_success(monkeypatch):
    trainer, batch = setup_trainer(monkeypatch, failed_install=True)
    with pytest.raises(ValueError, match="every actor"):
        trainer._prepare_proximal_actor(batch)


def test_unknown_stage_outcome_is_not_discarded_or_retried(monkeypatch):
    trainer, batch = setup_trainer(monkeypatch)
    monkeypatch.setattr(trainer_module.ray, "wait", lambda refs, **kwargs: (refs, []))
    with pytest.raises(RuntimeError, match="Invalid actor preparation wait result"):
        trainer._prepare_proximal_actor(batch)
    assert trainer.trainer.actor_rollout_wg.calls == ["stage"]


def test_stage_error_is_not_assumed_safe_to_discard(monkeypatch):
    trainer, batch = setup_trainer(monkeypatch)

    def failed_get(ref, **kwargs):
        raise RuntimeError("uncertain actor task")

    monkeypatch.setattr(trainer_module.ray, "get", failed_get)
    with pytest.raises(RuntimeError, match="uncertain actor task"):
        trainer._prepare_proximal_actor(batch)
    assert trainer.trainer.actor_rollout_wg.calls == ["stage"]


def test_known_preparation_failure_does_not_wait_for_pending_peers(monkeypatch):
    trainer, batch = setup_trainer(monkeypatch)
    monkeypatch.setattr(trainer_module.ray, "wait", lambda refs, **kwargs: (refs[:1], refs[1:]))

    def failed_get(ref, **kwargs):
        raise RuntimeError("known preparation failure")

    monkeypatch.setattr(trainer_module.ray, "get", failed_get)
    with pytest.raises(RuntimeError, match="known preparation failure"):
        trainer._prepare_proximal_actor(batch)
    assert trainer.trainer.actor_rollout_wg.calls == ["stage"]
