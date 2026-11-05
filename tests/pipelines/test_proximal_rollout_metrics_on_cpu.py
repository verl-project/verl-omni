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

"""Metrics-only rollout correction retains the native diagnostic values."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.protocol import DataProto
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.utils.debug.metrics import calculate_debug_metrics

from verl_omni.pipelines.proximal_reuse import rpc, strategy


def test_deferred_metrics_match_native_pre_update_calculation(monkeypatch):
    # Model and transport are mocked; the actual verl metrics helper runs in
    # both paths on exactly the same pre-update log-probs and mask.
    rows = TensorDict(
        {
            "old_log_probs": torch.tensor([[-0.3, -0.6, -1.0], [-0.7, -0.4, -0.2]]),
            "rollout_log_probs": torch.tensor([[-0.4, -0.5, -1.2], [-0.6, -0.2, -0.1]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        },
        batch_size=[2],
    )
    correction = {"rollout_is": None, "rollout_rs": None, "bypass_mode": False}
    expected_batch, expected = compute_rollout_correction_and_add_to_batch(
        DataProto(batch=rows.to_padded_tensor().clone()), correction
    )
    assert expected and torch.equal(expected_batch.batch["response_mask"], rows["response_mask"])

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            algorithm={"rollout_correction": correction},
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(calculate_log_probs=True)),
        ),
        actor_rollout_wg=object(),
    )
    candidate = strategy.ProximalActorReuse(trainer)
    monkeypatch.setattr(candidate, "_check_proximal_algorithm", lambda batch: None)
    monkeypatch.setattr(candidate, "_record_old_log_prob_output", lambda batch, metrics: batch)
    execution = {
        key: None
        for key in (
            "temperature",
            "global_batch_size",
            "mini_batch_size",
            "epochs",
            "seed",
            "dataloader_kwargs",
            "calculate_entropy",
            "return_model_output",
        )
    }
    monkeypatch.setattr(rpc, "actor_execution", lambda config: execution)
    monkeypatch.setattr(strategy.tq, "kv_batch_get", lambda **kwargs: rows)
    monkeypatch.setattr(strategy, "validate_proximal_output_collection", lambda group, plan: (True,))
    calls = []
    monkeypatch.setattr(strategy, "coordinate_staged_proximal_update", lambda *args: calls.append("update") or [])
    monkeypatch.setattr(strategy, "publish_proximal_outputs", lambda *args: object())
    monkeypatch.setattr(strategy, "get_training_output_metrics", lambda *args: {"mfu": 0.5})
    monkeypatch.setattr(strategy, "reduce_metrics", lambda values: values)
    batch = KVBatchMeta(keys=["a", "b"], tags=[{}, {}], partition_id="test")
    prepared = SimpleNamespace(bind=lambda *args: SimpleNamespace(plan=object()))
    actual = {}
    assert candidate._update_proximal_actor(batch, actual, prepared) is batch
    assert calls == ["update"]
    assert {key for key in actual if key.startswith("rollout_corr/")} == set(expected)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value)


@pytest.mark.parametrize("malformed", ["empty_mask", "rollout_shape"])
def test_invalid_diagnostics_fail_before_actor_preparation(monkeypatch, malformed):
    mask = torch.ones(2, 3, dtype=torch.int64)
    rollout = torch.full((2, 3), -0.5)
    if malformed == "empty_mask":
        mask.zero_()
    else:
        rollout = torch.full((2, 2), -0.5)
    rows = TensorDict(
        {
            "old_log_probs": torch.full((2, 3), -0.3),
            "rollout_log_probs": rollout,
            "response_mask": mask,
            "responses": torch.ones(2, 3, dtype=torch.int64),
        },
        batch_size=[2],
    )
    correction = {"rollout_is": None, "rollout_rs": None, "bypass_mode": False}
    with pytest.raises(ValueError):
        compute_rollout_correction_and_add_to_batch(DataProto(batch=rows.clone()), correction)

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            algorithm={"rollout_correction": correction},
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(calculate_log_probs=True)),
        ),
        actor_rollout_wg=SimpleNamespace(_query_dispatch_info=lambda _: [0]),
    )
    candidate = strategy.ProximalActorReuse(trainer)
    monkeypatch.setattr(candidate, "_check_proximal_algorithm", lambda batch: None)
    monkeypatch.setattr(rpc, "actor_execution", lambda config: {})
    monkeypatch.setattr(strategy.tq, "kv_batch_get", lambda **kwargs: rows.exclude("old_log_probs"))
    monkeypatch.setattr(strategy.worker_calls, "stage", lambda *args: pytest.fail("unexpected actor preparation"))
    batch = KVBatchMeta(keys=["a", "b"], tags=[{}, {}], partition_id="test")
    with pytest.raises(ValueError):
        candidate._prepare_proximal_actor(batch)


def test_valid_diagnostics_preflight_preserves_inputs():
    rows = TensorDict(
        {
            "rollout_log_probs": torch.full((2, 3), -0.5),
            "response_mask": torch.ones(2, 3),
            "responses": torch.ones(2, 3, dtype=torch.int64),
        },
        batch_size=[2],
    )
    before = rows.clone()
    strategy.validate_deferred_rollout_inputs(rows, require_nonempty=True)
    assert torch.equal(rows["response_mask"], before["response_mask"])
    assert torch.equal(rows["rollout_log_probs"], before["rollout_log_probs"])


def test_debug_only_bad_shape_fails_before_actor_preparation(monkeypatch):
    rows = TensorDict(
        {
            "old_log_probs": torch.full((2, 3), -0.3),
            "rollout_log_probs": torch.full((2, 2), -0.5),
            "response_mask": torch.ones(2, 3),
            "responses": torch.ones(2, 3, dtype=torch.int64),
        },
        batch_size=[2],
    )
    with pytest.raises(RuntimeError):
        calculate_debug_metrics(DataProto(batch=rows.clone()))
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            algorithm={"rollout_correction": None},
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(calculate_log_probs=True)),
        ),
        actor_rollout_wg=SimpleNamespace(_query_dispatch_info=lambda _: [0]),
    )
    candidate = strategy.ProximalActorReuse(trainer)
    monkeypatch.setattr(candidate, "_check_proximal_algorithm", lambda batch: None)
    monkeypatch.setattr(rpc, "actor_execution", lambda config: {})
    monkeypatch.setattr(strategy.tq, "kv_batch_get", lambda **kwargs: rows.exclude("old_log_probs"))
    monkeypatch.setattr(strategy.worker_calls, "stage", lambda *args: pytest.fail("unexpected actor preparation"))
    batch = KVBatchMeta(keys=["a", "b"], tags=[{}, {}], partition_id="test")
    with pytest.raises(ValueError):
        candidate._prepare_proximal_actor(batch)


@pytest.mark.parametrize("correction", [None, {"rollout_is": None, "rollout_rs": None, "bypass_mode": False}])
@pytest.mark.parametrize("response_width", [1, 2])
def test_bad_response_width_fails_before_actor_preparation(monkeypatch, correction, response_width):
    rows = TensorDict(
        {
            "old_log_probs": torch.full((2, 3), -0.3),
            "rollout_log_probs": torch.full((2, 3), -0.5),
            "response_mask": torch.ones(2, 3),
            "responses": torch.ones(2, response_width, dtype=torch.int64),
        },
        batch_size=[2],
    )
    if response_width == 2:
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            calculate_debug_metrics(DataProto(batch=rows.clone()))
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            algorithm={"rollout_correction": correction},
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(calculate_log_probs=True)),
        ),
        actor_rollout_wg=SimpleNamespace(_query_dispatch_info=lambda _: [0]),
    )
    candidate = strategy.ProximalActorReuse(trainer)
    monkeypatch.setattr(candidate, "_check_proximal_algorithm", lambda batch: None)
    monkeypatch.setattr(rpc, "actor_execution", lambda config: {})
    monkeypatch.setattr(strategy.tq, "kv_batch_get", lambda **kwargs: rows.exclude("old_log_probs"))
    monkeypatch.setattr(strategy.worker_calls, "stage", lambda *args: pytest.fail("unexpected actor preparation"))
    batch = KVBatchMeta(keys=["a", "b"], tags=[{}, {}], partition_id="test")
    with pytest.raises(ValueError, match="responses and response_mask"):
        candidate._prepare_proximal_actor(batch)


@pytest.mark.parametrize("rollout_shape", [(2, 2), (2,)])
def test_empty_mask_without_correction_matches_native_debug_early_return(rollout_shape):
    rows = TensorDict(
        {
            "old_log_probs": torch.full((2, 3), -0.3),
            "rollout_log_probs": torch.full(rollout_shape, -0.5),
            "response_mask": torch.zeros(2, 3),
            "responses": torch.ones(2, 3, dtype=torch.int64),
        },
        batch_size=[2],
    )
    assert calculate_debug_metrics(DataProto(batch=rows.clone()))["training/rollout_probs_diff_valid"] == 0
    strategy.validate_deferred_rollout_inputs(rows.exclude("old_log_probs"), require_nonempty=False)


def test_valid_jagged_rollout_probabilities_are_not_rejected_or_mutated():
    offsets = torch.tensor([0, 2, 5])
    mask = torch.nested.nested_tensor_from_jagged(torch.ones(5, dtype=torch.int64), offsets=offsets)
    rollout = torch.nested.nested_tensor_from_jagged(torch.full((5,), -0.5), offsets=offsets)
    responses = torch.nested.nested_tensor_from_jagged(torch.ones(5, dtype=torch.int64), offsets=offsets)
    rows = TensorDict({"response_mask": mask, "rollout_log_probs": rollout, "responses": responses}, batch_size=[2])
    before_mask, before_rollout = mask.values().clone(), rollout.values().clone()
    strategy.validate_deferred_rollout_inputs(rows, require_nonempty=True)
    assert torch.equal(mask.values(), before_mask)
    assert torch.equal(rollout.values(), before_rollout)
    assert torch.equal(mask.offsets(), offsets) and torch.equal(rollout.offsets(), offsets)


@pytest.mark.parametrize("correction,calculate_log_probs", [(None, False), ({}, False)])
def test_preparation_does_not_require_unrequested_rollout_diagnostics(monkeypatch, correction, calculate_log_probs):
    rows = TensorDict({"input_ids": torch.ones(2, 3)}, batch_size=[2])
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            algorithm={"rollout_correction": correction},
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(calculate_log_probs=calculate_log_probs)),
        ),
        actor_rollout_wg=SimpleNamespace(_query_dispatch_info=lambda _: [0]),
    )
    candidate = strategy.ProximalActorReuse(trainer)
    monkeypatch.setattr(candidate, "_check_proximal_algorithm", lambda batch: None)
    monkeypatch.setattr(rpc, "actor_execution", lambda config: {})
    monkeypatch.setattr(strategy.tq, "kv_batch_get", lambda **kwargs: rows)

    def stage(*args):
        raise RuntimeError("actor preparation reached")

    monkeypatch.setattr(strategy.worker_calls, "stage", stage)
    batch = KVBatchMeta(keys=["a", "b"], tags=[{}, {}], partition_id="test")
    with pytest.raises(RuntimeError, match="actor preparation reached"):
        candidate._prepare_proximal_actor(batch)
