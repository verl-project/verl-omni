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

"""Native local preparation logic with CPU worker doubles, no model or Ray."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.proximal_reuse import rpc
from verl_omni.workers.utils.cohort_data import CohortData
from verl_omni.workers.utils.parameter_revision import ParameterRevision


def make_workers_and_data(monkeypatch):
    execution = {"epochs": 1, "dataloader_kwargs": {"shuffle": False}, "mini_batch_size": 4}
    monkeypatch.setattr(rpc, "actor_execution", lambda config: execution)
    monkeypatch.setattr(rpc, "_check_actual_loss", lambda worker: None)
    monkeypatch.setattr(rpc, "_engine_digest", lambda engine: "d" * 64)
    full = TensorDict({"input_ids": torch.tensor([[1], [2], [3], [4]])}, batch_size=[4])
    keys = ("k0", "k1", "k2", "k3")
    request = rpc.PrepareRequest("cohort", "train", CohortData.prepare(full, keys, execution), (0, 1, 0, 1))
    workers, shards = [], []
    for rank, dp in enumerate(request.dp_mapping):
        tracker = ParameterRevision()
        engine = SimpleNamespace(
            engine_config=SimpleNamespace(
                use_dynamic_bsz=False, micro_batch_size_per_gpu=2, infer_micro_batch_size_per_gpu=2
            ),
            get_data_parallel_size=lambda: 2,
            get_data_parallel_rank=lambda dp=dp: dp,
            enable_parameter_tracking=lambda tracker=tracker: tracker,
        )
        worker = SimpleNamespace(rank=rank, world_size=4, actor=SimpleNamespace(engine=engine), config=object())
        data = full[2 * dp : 2 * dp + 2].clone()
        tu.assign_non_tensor_data(data, rpc.REQUEST, request)
        tu.assign_non_tensor_data(data, rpc.TRANSPORT, {"keys": keys[2 * dp : 2 * dp + 2], "partition_id": "train"})
        workers.append(worker)
        shards.append(data)
    return request, workers, shards


def test_all_local_observations_are_validated_and_installed(monkeypatch):
    request, workers, shards = make_workers_and_data(monkeypatch)
    for worker, data in zip(workers, shards, strict=True):
        rpc.prepare_actor_locally(worker, data)
    prepared = rpc.validate_preparation_observations(request, [rpc.get_actor_preparation(w) for w in workers])
    for worker in workers:
        assert rpc.install_actor_preparation(worker, prepared)
        assert worker._prepared_proximal_cohort == prepared
        with pytest.raises(RuntimeError, match="no staged"):
            rpc.get_actor_preparation(worker)


def test_hidden_changed_input_is_reported_without_any_install(monkeypatch):
    request, workers, shards = make_workers_and_data(monkeypatch)
    shards[2]["input_ids"][0, 0] = 99
    for worker, data in zip(workers, shards, strict=True):
        rpc.prepare_actor_locally(worker, data)
    with pytest.raises(ValueError, match="inputs differ"):
        rpc.validate_preparation_observations(request, [rpc.get_actor_preparation(w) for w in workers])
    assert all(getattr(worker, "_prepared_proximal_cohort", None) is None for worker in workers)
    assert all(worker.actor.engine.enable_parameter_tracking().version().revision == 0 for worker in workers)


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
def test_unqualified_batch_controls_are_rejected_before_actor_preparation(monkeypatch, key, value):
    _, workers, shards = make_workers_and_data(monkeypatch)
    worker, shard = workers[0], shards[0]
    tracker = worker.actor.engine.enable_parameter_tracking()
    tu.assign_non_tensor_data(shard, key, value)
    rpc.prepare_actor_locally(worker, shard)
    assert f"Unqualified actor batch control: {key}" in rpc.get_actor_preparation(worker)["error"]
    assert tracker.version().revision == 0
    assert getattr(worker, "_prepared_proximal_cohort", None) is None


def test_install_rejects_intervening_writer_and_preserves_staged_observation(monkeypatch):
    request, workers, shards = make_workers_and_data(monkeypatch)
    for worker, data in zip(workers, shards, strict=True):
        rpc.prepare_actor_locally(worker, data)
    observations = [rpc.get_actor_preparation(w) for w in workers]
    prepared = rpc.validate_preparation_observations(request, observations)
    tracker = workers[1].actor.engine.enable_parameter_tracking()
    with tracker.mutation("load"):
        pass
    with pytest.raises(ValueError, match="Stale"):
        rpc.install_actor_preparation(workers[1], prepared)
    assert rpc.get_actor_preparation(workers[1]) == observations[1]
    assert getattr(workers[1], "_prepared_proximal_cohort", None) is None


def test_local_preparation_does_not_overwrite_existing_state(monkeypatch):
    _, workers, shards = make_workers_and_data(monkeypatch)
    worker = workers[0]
    rpc.prepare_actor_locally(worker, shards[0])
    first = rpc.get_actor_preparation(worker)
    with pytest.raises(RuntimeError, match="still staged"):
        rpc.prepare_actor_locally(worker, shards[0])
    assert rpc.get_actor_preparation(worker) == first
    with pytest.raises(ValueError, match="differs"):
        rpc.discard_actor_preparation(worker, "wrong")
    assert rpc.get_actor_preparation(worker) == first
    assert rpc.discard_actor_preparation(worker, "cohort")
    rpc.prepare_actor_locally(worker, shards[0])
    assert rpc.get_actor_preparation(worker) == first
