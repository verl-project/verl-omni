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

"""Single-update strategy eligibility and conservative fallback."""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.proximal_reuse import rpc
from verl_omni.pipelines.proximal_reuse import strategy as strategy_module


def test_shared_worker_keeps_native_ppo_loss_by_default():
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "verl_omni/workers/engine_workers.py").read_text())
    imports = [
        (node.module, alias.name, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if (alias.asname or alias.name) == "ppo_loss"
    ]
    assert imports == [("verl.workers.utils.losses", "ppo_loss", None)]


def _config():
    return OmegaConf.create(
        {
            "trainer": {"v1": {"reuse_proximal_actor_forward": True}, "critic_warmup": 0},
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_epochs": 1,
                    "shuffle": False,
                    "use_prefix_grouper": False,
                    "calculate_sum_pi_squared": False,
                    "policy_loss": {"loss_mode": "gspo"},
                    "ppo_mini_batch_size": 2,
                },
                "rollout": {"n": 1},
            },
        }
    )


@pytest.fixture
def prepared_strategy():
    trainer = SimpleNamespace(
        config=_config(), global_steps=1, parameter_sync_step=1, use_critic=False, use_teacher_policy=False
    )
    return strategy_module.ProximalActorReuse(trainer)


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("selector", ["legacy", "name"])
def test_eligibility_before_skipping_native_old(monkeypatch, prepared_strategy, supported, selector):
    strategy = prepared_strategy
    if selector == "name":
        strategy.trainer.config.trainer.v1.pop("reuse_proximal_actor_forward")
        strategy.trainer.config.trainer.v1.step_extension = "proximal_actor_reuse"
    group = SimpleNamespace(_query_dispatch_info=lambda mesh: [0, 1])
    strategy.trainer.actor_rollout_wg = group
    strategy._check_proximal_algorithm = lambda batch: None
    monkeypatch.setattr(
        strategy_module.worker_calls,
        "control",
        lambda worker_group, operation: [
            {"supported": supported, "world_size": 2, "world_rank": rank} for rank in range(2)
        ],
    )
    ticket = object()
    strategy._prepare_proximal_actor = lambda batch: ticket
    calls = []
    batch, metrics = object(), {}
    assert strategy._compute_old_log_prob(batch, metrics, lambda batch, metrics: calls.append("old") or batch) is batch
    assert calls == ([] if supported else ["old"])
    assert metrics["training/proximal_anchor_reuse"] == int(supported)
    assert strategy._omni_prepared_actor is (ticket if supported else None)


@pytest.mark.parametrize("field,value", [("ppo_epochs", 2), ("shuffle", True), ("use_prefix_grouper", True)])
def test_unqualified_update_schedule_rejected(prepared_strategy, field, value):
    prepared_strategy.trainer.config.actor_rollout_ref.actor[field] = value
    with pytest.raises(ValueError, match="one qualified"):
        prepared_strategy._check_proximal_algorithm([1, 2])


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
def test_unqualified_batch_control_falls_back_before_actor_queries(monkeypatch, prepared_strategy, key, value):
    class Batch:
        extra_info = {key: value}

        def __len__(self):
            return 2

    batch = Batch()
    metrics = {}
    monkeypatch.setattr(strategy_module.worker_calls, "control", lambda *args: pytest.fail("Unexpected actor query"))
    result = prepared_strategy._compute_old_log_prob(batch, metrics, lambda input_batch, _: input_batch)
    assert result is batch and metrics["training/proximal_anchor_reuse"] == 0


def test_materialized_unqualified_control_falls_back_before_staging(monkeypatch, prepared_strategy):
    class Batch:
        extra_info = {}
        keys = ["k0", "k1"]
        partition_id = "train"

        def __len__(self):
            return len(self.keys)

    batch = Batch()
    data = TensorDict({"input_ids": torch.tensor([[1], [2]])}, batch_size=[2])
    tu.assign_non_tensor_data(data, "force_group_size", 2)
    prepared_strategy.trainer.actor_rollout_wg = SimpleNamespace(_query_dispatch_info=lambda mesh: [0, 1])
    monkeypatch.setattr(
        strategy_module.worker_calls,
        "control",
        lambda worker_group, operation: [{"supported": True, "world_size": 2, "world_rank": rank} for rank in range(2)],
    )
    monkeypatch.setattr(strategy_module.worker_calls, "stage", lambda *args: pytest.fail("Unexpected actor staging"))
    monkeypatch.setattr(strategy_module.tq, "kv_batch_get", lambda **kwargs: data)
    monkeypatch.setattr(rpc, "actor_execution", lambda config: {})
    metrics = {}
    result = prepared_strategy._compute_old_log_prob(batch, metrics, lambda input_batch, _: input_batch)
    assert result is batch and metrics["training/proximal_anchor_reuse"] == 0


@pytest.mark.parametrize("error", [TimeoutError, KeyboardInterrupt])
def test_unknown_preparation_outcome_fences_strategy_reentry(monkeypatch, prepared_strategy, error):
    class Batch:
        extra_info = {}
        keys = ["k0", "k1"]
        tags = [{}, {}]
        partition_id = "train"

        def __len__(self):
            return len(self.keys)

    from verl_omni.workers.utils.cohort_data import CohortData

    strategy = prepared_strategy
    batch = Batch()
    strategy.trainer.config.actor_rollout_ref.rollout.calculate_log_probs = False
    strategy.trainer.actor_rollout_wg = SimpleNamespace(_query_dispatch_info=lambda mesh: [0, 1])
    strategy._check_proximal_algorithm = lambda batch: None
    strategy._select_proximal_actor = lambda batch, metrics: strategy._prepare_proximal_actor(batch)
    monkeypatch.setattr(strategy_module.tq, "kv_batch_get", lambda **kwargs: {})
    monkeypatch.setattr(rpc, "actor_execution", lambda config: {})
    monkeypatch.setattr(CohortData, "prepare", lambda *args: object())
    dispatches = []

    def unknown_outcome(*args):
        dispatches.append("stage")
        raise error("unknown actor outcome")

    monkeypatch.setattr(strategy_module.worker_calls, "stage", unknown_outcome)

    def native(*args):
        pytest.fail("Unexpected native work")

    with pytest.raises(error, match="unknown actor outcome"):
        strategy._compute_old_log_prob(batch, {}, native)
    for method in (strategy._compute_old_log_prob, strategy._compute_advantage, strategy._update_actor):
        with pytest.raises(RuntimeError, match="previous actor reuse operation failed"):
            method(batch, {}, native)
    assert dispatches == ["stage"]


def test_unknown_update_outcome_fences_strategy_reentry(prepared_strategy):
    strategy = prepared_strategy
    strategy._omni_prepared_actor = object()
    strategy._omni_inflight = True
    updates = []

    def unknown_outcome(*args):
        updates.append("update")
        raise TimeoutError("unknown actor outcome")

    strategy._update_proximal_actor = unknown_outcome

    def native(*args):
        pytest.fail("Unexpected native work")

    with pytest.raises(TimeoutError, match="unknown actor outcome"):
        strategy._update_actor(object(), {}, native)
    for method in (strategy._compute_old_log_prob, strategy._compute_advantage, strategy._update_actor):
        with pytest.raises(RuntimeError, match="previous actor reuse operation failed"):
            method(object(), {}, native)
    assert updates == ["update"]


def test_publication_failure_after_commit_fences_strategy_reentry(prepared_strategy, monkeypatch):
    from transfer_queue import KVBatchMeta

    strategy = prepared_strategy
    strategy.trainer.actor_rollout_wg = object()
    strategy._omni_inflight = True
    strategy._omni_prepared_actor = SimpleNamespace(bind=lambda *_args: SimpleNamespace(plan=object()))
    strategy._check_proximal_algorithm = lambda _batch: None
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
    monkeypatch.setattr(rpc, "actor_execution", lambda _config: execution)
    monkeypatch.setattr(strategy_module.tq, "kv_batch_get", lambda **_kwargs: {})
    monkeypatch.setattr(strategy_module, "validate_proximal_output_collection", lambda *_args: (True,))
    commits = []
    monkeypatch.setattr(
        strategy_module, "coordinate_staged_proximal_update", lambda *_args: commits.append("commit") or [object()]
    )

    def fail_publication(*_args):
        raise OSError("publication outcome unknown")

    monkeypatch.setattr(strategy_module, "publish_proximal_outputs", fail_publication)
    batch = KVBatchMeta(keys=["row"], tags=[{}], partition_id="train")
    native = lambda *_args: pytest.fail("must not retry through native path")
    with pytest.raises(OSError, match="publication outcome unknown"):
        strategy._update_actor(batch, {}, native)
    with pytest.raises(RuntimeError, match="previous actor reuse operation failed"):
        strategy._update_actor(batch, {}, native)
    assert commits == ["commit"]


def test_native_fallback_failure_does_not_fence_strategy(prepared_strategy):
    strategy = prepared_strategy
    strategy._select_proximal_actor = lambda *args: None

    def fail_native(*args):
        raise ValueError("native error")

    with pytest.raises(ValueError, match="native error"):
        strategy._compute_old_log_prob(object(), {}, fail_native)
    assert strategy._compute_old_log_prob(object(), {}, lambda *args: "ok") == "ok"


@pytest.mark.parametrize("method_name", ["_compute_old_log_prob", "_compute_advantage", "_update_actor"])
def test_concurrent_hooks_are_rejected_before_any_second_operation(prepared_strategy, method_name):
    strategy = prepared_strategy
    entered, resume = Event(), Event()
    batch, ticket = object(), object()

    def select(actual, metrics):
        assert actual is batch
        entered.set()
        if not resume.wait(timeout=5):
            raise TimeoutError("test synchronization failed")
        return ticket

    strategy._select_proximal_actor = select
    native_calls = []

    def native(*args):
        native_calls.append(args)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(strategy._compute_old_log_prob, batch, {}, native)
        assert entered.wait(timeout=5)
        try:
            concurrent = pool.submit(getattr(strategy, method_name), object(), {}, native)
            with pytest.raises(RuntimeError, match="Concurrent actor reuse operations"):
                concurrent.result(timeout=2)
        finally:
            resume.set()
        assert first.result(timeout=5) is batch
    assert not native_calls
    assert strategy._omni_prepared_actor is ticket


@pytest.mark.parametrize("worlds,ranks", [([3, 3], [0, 1]), ([2, 2], [0, 0])])
def test_mismatched_world_falls_back_before_preparation(monkeypatch, prepared_strategy, worlds, ranks):
    strategy = prepared_strategy
    strategy.trainer.actor_rollout_wg = SimpleNamespace(_query_dispatch_info=lambda mesh: [0, 1])
    strategy._check_proximal_algorithm = lambda batch: None
    strategy._prepare_proximal_actor = lambda batch: pytest.fail("unsupported topology prepared")
    monkeypatch.setattr(
        strategy_module.worker_calls,
        "control",
        lambda group, operation: [
            {"supported": True, "world_size": world, "world_rank": rank}
            for world, rank in zip(worlds, ranks, strict=True)
        ],
    )
    calls, metrics = [], {}
    batch = object()
    assert strategy._compute_old_log_prob(batch, metrics, lambda batch, metrics: calls.append("old") or batch) is batch
    assert calls == ["old"] and metrics["training/proximal_anchor_reuse"] == 0


@pytest.mark.parametrize("initialized", [True, False])
def test_qualification_reports_actual_world(monkeypatch, initialized):
    monkeypatch.setattr(rpc, "_check_actual_loss", lambda worker: None)
    monkeypatch.setattr(rpc, "_engine_digest", lambda engine: "digest")
    monkeypatch.setattr(rpc.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(rpc.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(rpc.dist, "get_rank", lambda: 2)
    worker = SimpleNamespace(actor=SimpleNamespace(engine=object()), enable_routing_replay=False)
    result = rpc.qualify_actor(worker)
    assert result["supported"] is initialized
    if initialized:
        assert (result["world_size"], result["world_rank"]) == (4, 2)
    else:
        assert "not initialized" in result["reason"]


def test_unknown_engine_rejected_before_qualification():
    with pytest.raises(ValueError, match="does not support"):
        rpc._engine_digest(object())


def test_execution_config_accepts_dataclass_instances_but_not_classes():
    @dataclass
    class Configuration:
        value: int = 3

    assert rpc._plain(Configuration()) == {"value": 3}
    with pytest.raises(TypeError, match="actual resolved configuration"):
        rpc._plain(Configuration)
