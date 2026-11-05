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

"""Actual local update observations with CPU worker doubles."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.utils.losses import ppo_loss

from verl_omni.pipelines.proximal_reuse import rpc
from verl_omni.pipelines.proximal_reuse.anchor import proximal_ppo_loss
from verl_omni.workers.utils.cohort_data import CohortData
from verl_omni.workers.utils.cohort_lease import validate_observations
from verl_omni.workers.utils.parameter_revision import ParameterRevision


def prepared_update(monkeypatch):
    execution = {
        "temperature": 1.0,
        "global_batch_size": 4,
        "mini_batch_size": 4,
        "epochs": 1,
        "seed": 42,
        "dataloader_kwargs": {"shuffle": False},
        "calculate_entropy": True,
        "return_model_output": True,
    }
    monkeypatch.setattr(rpc, "actor_execution", lambda config: execution)
    monkeypatch.setattr(rpc, "_check_actual_loss", lambda worker: None)
    monkeypatch.setattr(rpc, "_engine_digest", lambda engine: engine.digest)
    raw = TensorDict({"input_ids": torch.tensor([[1], [2], [3], [4]])}, batch_size=[4])
    keys = ("k0", "k1", "k2", "k3")
    request = rpc.PrepareRequest("cohort", "train", CohortData.prepare(raw, keys, execution), (0, 1, 0, 1))
    workers, observations = [], []
    for rank, dp in enumerate(request.dp_mapping):
        tracker = ParameterRevision()
        engine = SimpleNamespace(
            digest="d" * 64,
            engine_config=SimpleNamespace(),
            get_data_parallel_rank=lambda dp=dp: dp,
            enable_parameter_tracking=lambda tracker=tracker: tracker,
        )
        worker = SimpleNamespace(rank=rank, world_size=4, actor=SimpleNamespace(engine=engine), config=object())
        receipt = rpc.RankReceipt(
            rank, dp, tracker.version(), request.binding.shard([2 * dp, 2 * dp + 1]), engine.digest
        )
        observations.append({"request": request, "receipt": receipt})
        workers.append(worker)
    prepared = rpc.validate_preparation_observations(request, observations)
    full = raw.clone()
    full["advantages"] = torch.tensor([[0.2], [-0.2], [0.5], [-0.5]])
    full["returns"] = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    update = prepared.bind(full, keys, execution)
    shard = full[:2].clone()
    tu.assign_non_tensor_data(shard, rpc.TRANSPORT, {"keys": keys[:2], "partition_id": "train"})
    for name, value in execution.items():
        tu.assign_non_tensor_data(shard, name, value)
    tu.assign_non_tensor_data(shard, rpc.REQUEST, update)
    workers[0]._prepared_proximal_cohort = prepared
    return workers[0], shard, update, prepared


def test_valid_actual_shard_matches_coordinator_plan(monkeypatch):
    worker, shard, update, prepared = prepared_update(monkeypatch)
    assert rpc.observe_actor_update(worker, shard, update, prepared, 0) == update.plan.shards[0]


@pytest.mark.parametrize("field", ["input_ids", "advantages", "returns"])
def test_changed_row_is_rejected_before_update(monkeypatch, field):
    worker, shard, update, prepared = prepared_update(monkeypatch)
    shard[field][0, 0] += 1
    with pytest.raises(ValueError, match="cohort data changed"):
        rpc.observe_actor_update(worker, shard, update, prepared, 0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enable_routing_replay", True),
        ("distillation_use_topk", True),
        ("forced_max_seqlen", 512),
        ("distillation_only", True),
        ("calculate_sum_pi_squared", True),
        ("local_cp_size", 2),
        ("force_group_size", 2),
        ("pad_mode", "right"),
        ("score_centering", True),
        ("no_lora_adapter", True),
    ],
)
def test_changed_batch_control_is_rejected_before_reservation(monkeypatch, key, value):
    worker, shard, _, _ = prepared_update(monkeypatch)
    tracker = worker.actor.engine.enable_parameter_tracking()
    tu.assign_non_tensor_data(shard, key, value)
    rpc.stage_actor_update(worker, shard)
    assert f"Unqualified actor batch control: {key}" in rpc.get_staged_actor_update(worker)["error"]
    assert rpc.discard_staged_actor_update(worker, "cohort")
    assert tracker.version().revision == 0
    with tracker.mutation("load"):
        pass


def test_explicit_native_default_batch_controls_remain_qualified(monkeypatch):
    worker, shard, update, prepared = prepared_update(monkeypatch)
    defaults = {
        "enable_routing_replay": False,
        "distillation_use_topk": False,
        "forced_max_seqlen": None,
        "distillation_only": False,
        "calculate_sum_pi_squared": False,
        "local_cp_size": None,
        "force_group_size": 1,
        "pad_mode": DatasetPadMode.NO_PADDING,
        "score_centering": False,
        "no_lora_adapter": False,
    }
    for key, value in defaults.items():
        tu.assign_non_tensor_data(shard, key, value)
    assert rpc.observe_actor_update(worker, shard, update, prepared, 0) == update.plan.shards[0]


def test_changed_transport_controls_and_engine_are_rejected(monkeypatch):
    worker, shard, update, prepared = prepared_update(monkeypatch)
    tu.assign_non_tensor_data(shard, rpc.TRANSPORT, {"keys": ("k0", "k1"), "partition_id": "other"})
    with pytest.raises(ValueError, match="shard changed"):
        rpc.observe_actor_update(worker, shard, update, prepared, 0)
    tu.assign_non_tensor_data(shard, rpc.TRANSPORT, {"keys": ("k0", "k1"), "partition_id": "train"})
    tu.assign_non_tensor_data(shard, "temperature", 0.8)
    with pytest.raises(ValueError, match="temperature"):
        rpc.observe_actor_update(worker, shard, update, prepared, 0)
    tu.assign_non_tensor_data(shard, "temperature", 1.0)
    worker.actor.engine.digest = "e" * 64
    with pytest.raises(ValueError, match="engine execution"):
        rpc.observe_actor_update(worker, shard, update, prepared, 0)


def test_stage_reserves_version_and_owns_its_input(monkeypatch):
    worker, shard, update, prepared = prepared_update(monkeypatch)
    tracker = worker.actor.engine.enable_parameter_tracking()
    rpc.stage_actor_update(worker, shard)
    observation = rpc.get_staged_actor_update(worker)
    assert observation["shard"] == update.plan.shards[0]
    replies = []
    for rank, dp in enumerate(prepared.request.dp_mapping):
        replies.append(
            {
                "cohort_id": "cohort",
                "partition_id": "train",
                "rank": rank,
                "dp_rank": dp,
                "version": update.plan.versions[rank],
                "shard": update.plan.shards[dp],
            }
        )
    replies[0] = observation
    validate_observations(update.plan, replies)
    with pytest.raises(RuntimeError, match="reserved"):
        with tracker.mutation("load"):
            pytest.fail("A staged version must be fenced")
    shard["input_ids"][0, 0] = 99
    frozen = worker._staged_proximal_update.data
    assert rpc.observe_actor_update(worker, frozen, update, prepared, 0) == update.plan.shards[0]
    with pytest.raises(ValueError, match="differs"):
        rpc.discard_staged_actor_update(worker, "other")
    assert rpc.discard_staged_actor_update(worker, "cohort")
    assert tracker.version().revision == 0
    assert worker._prepared_proximal_cohort == prepared


def test_bad_input_or_stale_version_does_not_enter_reservation(monkeypatch):
    worker, shard, _, _ = prepared_update(monkeypatch)
    tracker = worker.actor.engine.enable_parameter_tracking()
    original = shard["advantages"][0, 0].clone()
    shard["advantages"][0, 0] += 1
    rpc.stage_actor_update(worker, shard)
    assert "derived cohort data changed" in rpc.get_staged_actor_update(worker)["error"]
    assert rpc.discard_staged_actor_update(worker, "cohort")
    with tracker.mutation("load"):
        pass
    shard["advantages"][0, 0] = original
    rpc.stage_actor_update(worker, shard)
    assert "Stale" in rpc.get_staged_actor_update(worker)["error"]
    assert rpc.discard_staged_actor_update(worker, "cohort")
    assert tracker.version().revision == 1


def test_staging_rejects_mismatched_plan_and_duplicate_stage(monkeypatch):
    worker, shard, update, _ = prepared_update(monkeypatch)
    wrong = replace(update, plan=replace(update.plan, ranks_by_dp=((0, 1), (2, 3))))
    tu.assign_non_tensor_data(shard, rpc.REQUEST, wrong)
    rpc.stage_actor_update(worker, shard)
    assert "topology" in rpc.get_staged_actor_update(worker)["error"]
    with pytest.raises(RuntimeError, match="still staged"):
        rpc.stage_actor_update(worker, shard)
    assert rpc.discard_staged_actor_update(worker, "cohort")


def test_commit_consumes_frozen_shard_once_with_real_optimizer(monkeypatch):
    worker, shard, _, _ = prepared_update(monkeypatch)
    tracker = worker.actor.engine.enable_parameter_tracking()
    worker.config = SimpleNamespace(actor=SimpleNamespace(entropy_coeff=0, calculate_entropy=False))
    original_loss = partial(ppo_loss, config=object())
    worker.actor.loss_fn = original_loss
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    seen = []

    def train(*, data):
        assert worker.actor.loss_fn.func is proximal_ppo_loss
        seen.append(data["input_ids"][0, 0].item())
        optimizer.zero_grad()
        (parameter * 2).backward()
        with tracker.mutation("optimizer"):
            optimizer.step()
        return TensorDict({"value": parameter.detach().reshape(1)}, batch_size=[1])

    worker.actor.train_mini_batch = train
    rpc.stage_actor_update(worker, shard)
    shard["input_ids"][0, 0] = 99
    with pytest.raises(ValueError, match="matching"):
        rpc.commit_staged_actor_update(worker, "other")
    output = rpc.commit_staged_actor_update(worker, "cohort")
    assert seen == [1]
    assert output["value"].item() == pytest.approx(0.8)
    assert tracker.version().revision == 1
    assert worker.actor.loss_fn is original_loss
    assert worker._prepared_proximal_cohort is None
    with pytest.raises(ValueError, match="matching"):
        rpc.commit_staged_actor_update(worker, "cohort")


def test_failed_commit_poisoned_and_cannot_retry(monkeypatch):
    worker, shard, _, _ = prepared_update(monkeypatch)
    tracker = worker.actor.engine.enable_parameter_tracking()
    worker.config = SimpleNamespace(actor=SimpleNamespace(entropy_coeff=0, calculate_entropy=False))
    original_loss = partial(ppo_loss, config=object())
    worker.actor.loss_fn = original_loss

    def train(*, data):
        with tracker.mutation("optimizer"):
            raise RuntimeError("simulated partial optimizer failure")

    worker.actor.train_mini_batch = train
    rpc.stage_actor_update(worker, shard)
    with pytest.raises(RuntimeError, match="partial optimizer"):
        rpc.commit_staged_actor_update(worker, "cohort")
    assert worker.actor.loss_fn is original_loss
    with pytest.raises(RuntimeError, match="Partial actor operation"):
        tracker.version()
    with pytest.raises(ValueError, match="matching"):
        rpc.commit_staged_actor_update(worker, "cohort")


def test_failed_output_transfer_after_optimizer_poisoned(monkeypatch):
    worker, shard, _, _ = prepared_update(monkeypatch)
    tracker = worker.actor.engine.enable_parameter_tracking()
    worker.config = SimpleNamespace(actor=SimpleNamespace(entropy_coeff=0, calculate_entropy=False))
    original_loss = partial(ppo_loss, config=object())
    worker.actor.loss_fn = original_loss

    class BadOutput:
        def cpu(self):
            raise OSError("output transfer failed")

    def train(*, data):
        with tracker.mutation("optimizer"):
            pass
        return BadOutput()

    worker.actor.train_mini_batch = train
    rpc.stage_actor_update(worker, shard)
    with pytest.raises(OSError, match="output transfer failed"):
        rpc.commit_staged_actor_update(worker, "cohort")
    assert worker.actor.loss_fn is original_loss
    with pytest.raises(RuntimeError, match="Partial actor operation"):
        tracker.version()


def test_rejected_stage_cannot_commit(monkeypatch):
    worker, shard, _, _ = prepared_update(monkeypatch)
    shard["input_ids"][0, 0] = 99
    rpc.stage_actor_update(worker, shard)
    with pytest.raises(ValueError, match="rejected"):
        rpc.commit_staged_actor_update(worker, "cohort")
    assert rpc.discard_staged_actor_update(worker, "cohort")
