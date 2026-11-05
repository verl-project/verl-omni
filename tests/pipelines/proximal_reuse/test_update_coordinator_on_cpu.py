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

"""Coordinator failure semantics before wiring into the training loop."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.single_controller.base import decorator
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.proximal_reuse import strategy as coordinator
from verl_omni.workers.utils import cohort_lease
from verl_omni.workers.utils.training_metrics import get_training_output_metrics


class Group:
    def __init__(self):
        self.calls = []
        self.observations = [{"rank": 0}, {"rank": 1}]

    def stage_step_extension(self, metadata):
        assert metadata.extra_info.pop("_step_extension_selector") == ("proximal_actor_reuse", "update")
        return self.stage_proximal_update("batch")

    def control_step_extension(self, name, operation, payload=None):
        assert name == "proximal_actor_reuse"
        if operation == "get_update":
            return self.get_staged_proximal_update()
        if operation == "discard_update":
            return self.discard_staged_proximal_update(payload)
        raise AssertionError(operation)

    def commit_step_extension(self, name, operation, payload):
        assert (name, operation) == ("proximal_actor_reuse", "update")
        return self.commit_staged_proximal_update(payload)

    def stage_proximal_update(self, data):
        self.calls.append(("stage", data))
        return ["stage0", "stage1"]

    def get_staged_proximal_update(self):
        self.calls.append(("observe",))
        return self.observations

    def discard_staged_proximal_update(self, cohort_id):
        self.calls.append(("discard", cohort_id))
        return [True, True]

    def commit_staged_proximal_update(self, cohort_id):
        self.calls.append(("commit", cohort_id))
        return ["commit0", "commit1"]


@pytest.fixture
def setup(monkeypatch):
    plan = SimpleNamespace(versions=(0, 0), cohort_id="cohort", validate=lambda count: None)
    group = Group()
    monkeypatch.setattr(cohort_lease, "validate_observations", lambda plan, observations: None)
    monkeypatch.setattr(coordinator.ray, "wait", lambda refs, **kwargs: (refs[:1], refs[1:]))
    monkeypatch.setattr(coordinator.ray, "get", lambda ref, **kwargs: f"result:{ref}")
    # Use actual metadata, as required by the native shard dispatch.
    original = coordinator.coordinate_staged_proximal_update
    monkeypatch.setattr(
        coordinator,
        "coordinate_staged_proximal_update",
        lambda group, _batch, plan: original(group, KVBatchMeta(keys=["key"], tags=[{}], partition_id="train"), plan),
    )
    return group, plan


def test_success_waits_all_stages_before_commit(setup):
    group, plan = setup
    result = coordinator.coordinate_staged_proximal_update(group, "batch", plan)
    assert result == ["result:commit0", "result:commit1"]
    assert group.calls == [("stage", "batch"), ("observe",), ("commit", "cohort")]


@pytest.mark.parametrize("malformed", [None, [{"rank": 0}], [{"rank": 0}, None]])
def test_rejected_observations_discard_without_commit(setup, malformed):
    group, plan = setup
    group.observations = malformed
    with pytest.raises(ValueError, match="malformed"):
        coordinator.coordinate_staged_proximal_update(group, "batch", plan)
    assert group.calls == [("stage", "batch"), ("observe",), ("discard", "cohort")]


def test_known_version_mismatch_discards_without_commit(setup, monkeypatch):
    group, plan = setup

    def reject(plan, observations):
        raise ValueError("stale actor")

    monkeypatch.setattr(cohort_lease, "validate_observations", reject)
    with pytest.raises(ValueError, match="stale actor"):
        coordinator.coordinate_staged_proximal_update(group, "batch", plan)
    assert group.calls[-1] == ("discard", "cohort")


@pytest.mark.parametrize("phase", ["stage", "commit"])
def test_rpc_failure_has_unknown_outcome_and_never_retries_or_discards(setup, monkeypatch, phase):
    group, plan = setup

    def failed(ref, **kwargs):
        if ref == f"{phase}0":
            raise RuntimeError("unknown actor outcome")
        return ref

    monkeypatch.setattr(coordinator.ray, "get", failed)
    with pytest.raises(RuntimeError, match="unknown actor outcome"):
        coordinator.coordinate_staged_proximal_update(group, "batch", plan)
    assert not any(call[0] == "discard" for call in group.calls)
    assert sum(call[0] == "commit" for call in group.calls) == (phase == "commit")


def test_failed_commit_is_reported_without_waiting_for_a_stuck_peer(setup, monkeypatch):
    group, plan = setup
    waited = []

    def wait(refs, **kwargs):
        waited.append(list(refs))
        if refs == ["commit1"]:
            pytest.fail("the failed rank should be reported before waiting on a stuck peer")
        return refs[:1], refs[1:]

    def get(ref, **kwargs):
        if ref == "commit0":
            raise RuntimeError("partial update")
        return ref

    monkeypatch.setattr(coordinator.ray, "wait", wait)
    monkeypatch.setattr(coordinator.ray, "get", get)
    with pytest.raises(RuntimeError, match="partial update"):
        coordinator.coordinate_staged_proximal_update(group, "batch", plan)
    assert waited[-1] == ["commit0", "commit1"]
    assert group.calls == [("stage", "batch"), ("observe",), ("commit", "cohort")]


@pytest.mark.parametrize("mapping, mask", [([0, 1, 0, 1], [True, True, False, False])])
def test_output_collection_requires_one_source_in_dp_order(mapping, mask):
    plan = SimpleNamespace(versions=(0,) * 4, ranks_by_dp=((0, 2), (1, 3)), validate=lambda count: None)
    group = SimpleNamespace(_query_dispatch_info=lambda mesh: mapping, _query_collect_info=lambda mesh: mask)
    assert coordinator.validate_proximal_output_collection(group, plan) == tuple(mask)


@pytest.mark.parametrize(
    "mapping, mask",
    [
        ([0, 1, 0, 1], [True, True, True, False]),
        ([0, 1, 0, 1], [False, True, True, False]),
        ([0, True, 0, 1], [True, True, False, False]),
        ([0, 1, 0, 1], [1, True, False, False]),
    ],
)
def test_output_collection_rejects_duplicates_wrong_order_and_bad_types(mapping, mask):
    plan = SimpleNamespace(versions=(0,) * 4, ranks_by_dp=((0, 2), (1, 3)), validate=lambda count: None)
    group = SimpleNamespace(_query_dispatch_info=lambda mesh: mapping, _query_collect_info=lambda mesh: mask)
    with pytest.raises(ValueError):
        coordinator.validate_proximal_output_collection(group, plan)


def test_output_publish_writes_tokens_and_returns_metrics(monkeypatch):
    selected = TensorDict({"log_probs": torch.ones(2, 1), "entropy": torch.zeros(2, 1)}, batch_size=[2])
    tu.assign_non_tensor_data(selected, "metrics", {"loss": 0.125})
    batch = type("Batch", (), {"__len__": lambda self: 2, "keys": ["a", "b"], "partition_id": "train"})()
    monkeypatch.setattr(decorator, "collect_nd_compute_dataproto", lambda mask, group, outputs: selected)
    calls = []
    monkeypatch.setattr(coordinator.tq, "kv_batch_put", lambda **kwargs: calls.append(kwargs))
    result = coordinator.publish_proximal_outputs(object(), batch, [selected, None], [True, False])
    assert tu.get_non_tensor_data(result, "metrics", default=None) == {"loss": 0.125}
    assert calls[0]["keys"] == ["a", "b"]
    assert set(calls[0]["fields"].keys()) == {"log_probs", "entropy"}


def test_output_publish_rejects_missing_tokens_before_write(monkeypatch):
    selected = TensorDict({"log_probs": torch.ones(2, 1)}, batch_size=[2])
    batch = type("Batch", (), {"__len__": lambda self: 2})()
    monkeypatch.setattr(decorator, "collect_nd_compute_dataproto", lambda mask, group, outputs: selected)
    monkeypatch.setattr(coordinator.tq, "kv_batch_put", lambda **kwargs: pytest.fail("must not publish"))
    with pytest.raises(ValueError, match="entropy"):
        coordinator.publish_proximal_outputs(object(), batch, [selected], [True])


def test_collected_metrics_match_native_bridge_when_dp_ranks_differ():
    parts = []
    for loss in (1.0, 3.0):
        output = TensorDict({"log_probs": torch.ones(2, 1), "entropy": torch.ones(2, 1)}, batch_size=[2])
        tu.assign_non_tensor_data(output, "metrics", {"loss": [loss], "mfu": [loss / 10]})
        parts.append(output)
    selected = tu.concat_tensordict(parts)
    batch = KVBatchMeta(keys=["a", "b", "c", "d"], tags=[{}] * 4, partition_id="p", fields=None, extra_info={})
    native = KVBatchMeta(
        keys=batch.keys,
        tags=batch.tags,
        partition_id=batch.partition_id,
        fields=["log_probs", "entropy"],
        extra_info={"metrics": [{"loss": [1.0], "mfu": [0.1]}, {"loss": [3.0], "mfu": [0.3]}]},
    )
    expected = {"loss": [1.0], "mfu": [0.1]}
    assert get_training_output_metrics(selected, batch) == expected
    assert get_training_output_metrics(native, batch) == expected
