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
"""CPU regression tests for diffusion V1 parameter-sync cycles."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta
from verl import DataProto

from verl_omni.trainer.diffusion.v1 import trainer_base as trainer_base_module
from verl_omni.trainer.diffusion.v1.metrics import DiffusionMetricsAggregator
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
from verl_omni.trainer.diffusion.v1.trainer_separate_async import (
    PolicyGradientDiffusionTrainerV1SeparateAsync,
)
from verl_omni.workers import detach_actor_worker as detach_actor_worker_module
from verl_omni.workers.detach_actor_worker import DiffusionDetachActorWorker


def _separate_async_config(*, train_batch_size=8, ppo_mini_batch_size=2, parameter_sync_step=4):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": train_batch_size},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": ppo_mini_batch_size},
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 1,
                    "checkpoint_engine": {"backend": "nccl"},
                },
            },
            "trainer": {
                "v1": {
                    "separate_async": {"parameter_sync_step": parameter_sync_step},
                }
            },
        }
    )


def test_separate_async_validates_parameter_sync_batches(monkeypatch):
    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "__init__", lambda self, config: None)
    PolicyGradientDiffusionTrainerV1SeparateAsync(_separate_async_config())

    invalid_configs = [
        (_separate_async_config(train_batch_size=7), r"parameter_sync_step \* ppo_mini_batch_size"),
        (_separate_async_config(parameter_sync_step=0), "must be positive"),
    ]
    for config, error in invalid_configs:
        with pytest.raises(AssertionError, match=error):
            PolicyGradientDiffusionTrainerV1SeparateAsync(config)


def test_separate_async_replay_buffer_counts_outer_step_versions():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "v1": {
                    "separate_async": {"parameter_sync_step": 4},
                    "sampler": {
                        "max_off_policy_threshold": 8,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                        "max_incomplete_group_refill_rounds": 3,
                    },
                }
            }
        }
    )

    replay_buffer = trainer._build_replay_buffer()

    assert replay_buffer.parameter_sync_step == 1


def test_base_step_samples_one_mini_batch_per_local_update():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 8}})
    trainer.parameter_sync_step = 4
    sample_sizes = []
    local_steps = []

    trainer._add_batch_to_generate = lambda: None

    def step_once(iter_metrics, timing_raw, sample_batch_size):
        del timing_raw
        sample_sizes.append(sample_batch_size)
        local_steps.append(trainer.local_trigger_step)
        iter_metrics["actor/loss/mean"] = float(trainer.local_trigger_step)
        iter_metrics["training/rollout_failure/refilled_prompts"] = 1
        return KVBatchMeta(
            partition_id="train",
            keys=[f"sample-{trainer.local_trigger_step}"],
            tags=[{"is_padding": False}],
        )

    trainer._step_once = step_once
    metrics = {}
    batch = PolicyGradientDiffusionTrainerV1.step(trainer, metrics, {})

    assert sample_sizes == [2, 2, 2, 2]
    assert local_steps == [0, 1, 2, 3]
    assert batch.keys == ["sample-0", "sample-1", "sample-2", "sample-3"]
    assert metrics["actor/loss/mean"] == pytest.approx(1.5)
    assert metrics["training/rollout_failure/refilled_prompts"] == 4


@pytest.mark.parametrize(
    ("batch_size", "dp_size", "message"),
    [
        (2, 1, "refusing to pad copied trajectories"),
        (4, 3, "must be divisible by actor DP size 3"),
    ],
)
def test_separate_async_refuses_to_pad_invalid_local_batch(batch_size, dp_size, message):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    if dp_size == 1:
        trainer.actor_rollout_wg = SimpleNamespace()
    else:
        trainer.actor_rollout_wg = SimpleNamespace(_query_dispatch_info=lambda _mesh_name: list(range(dp_size)))
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 2},
                "rollout": {"n": 2},
            }
        }
    )
    data = DataProto.from_dict(tensors={"value": torch.zeros(batch_size)})

    with pytest.raises(ValueError, match=message):
        trainer._balance_batch(data, {})


def test_separate_async_refills_stale_groups_without_padding():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.global_steps = 5
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "trainer": {"v1": {"sampler": {"max_incomplete_group_refill_rounds": 3}}},
        }
    )
    responses = [
        (
            KVBatchMeta(
                partition_id="train",
                keys=["kept_0_0", "kept_1_0"],
                tags=[{"global_steps": 5}, {"global_steps": 5}],
            ),
            {
                "training/off_policy/dropped_samples": 2,
                "training/off_policy/dropped_samples_staleness/mean": 3.0,
            },
        ),
        (
            KVBatchMeta(
                partition_id="train",
                keys=["replacement_0_0", "replacement_1_0"],
                tags=[{"global_steps": 5}, {"global_steps": 5}],
            ),
            {},
        ),
    ]
    sample_sizes = []
    trainer.replay_buffer = SimpleNamespace(
        sample=lambda **kwargs: sample_sizes.append(kwargs["batch_size"]) or responses.pop(0)
    )
    submitted = []
    trainer._generation_batch_size = lambda: 4
    trainer._add_prompts_to_generate = lambda count: submitted.append(count) or count

    batch, metrics = trainer._sample_training_batch(2)

    assert sample_sizes == [2, 1]
    assert submitted == [4]
    assert batch.keys == ["kept_0_0", "kept_1_0", "replacement_0_0", "replacement_1_0"]
    assert metrics["training/off_policy/dropped_samples"] == 2
    assert metrics["training/off_policy/dropped_samples_staleness/mean"] == pytest.approx(3.0)


def test_separate_async_refills_incomplete_groups_without_padding(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.global_steps = 1
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "trainer": {"v1": {"sampler": {"max_incomplete_group_refill_rounds": 3}}},
        }
    )
    responses = [
        (
            KVBatchMeta(
                partition_id="train",
                keys=["partial_0_0"],
                tags=[{"global_steps": 1}],
            ),
            {},
        ),
        (
            KVBatchMeta(
                partition_id="train",
                keys=["replacement_0_0", "replacement_1_0"],
                tags=[{"global_steps": 1}, {"global_steps": 1}],
            ),
            {},
        ),
    ]
    sample_sizes = []
    trainer.replay_buffer = SimpleNamespace(
        sample=lambda **kwargs: sample_sizes.append(kwargs["batch_size"]) or responses.pop(0)
    )
    submitted = []
    cleared = []
    trainer._generation_batch_size = lambda: 1
    trainer._add_prompts_to_generate = lambda count: submitted.append(count) or count
    monkeypatch.setattr(
        trainer_base_module.tq,
        "kv_clear",
        lambda *, partition_id, keys: cleared.append((partition_id, list(keys))),
    )

    batch, metrics = trainer._sample_training_batch(1)

    assert sample_sizes == [1, 1]
    assert submitted == [1]
    assert cleared == [("train", ["partial_0_0"])]
    assert batch.keys == ["replacement_0_0", "replacement_1_0"]
    assert metrics["training/rollout_failure/evicted_groups"] == 1
    assert metrics["training/rollout_failure/evicted_trajectories"] == 1
    assert metrics["training/rollout_failure/refilled_prompts"] == 1
    assert metrics["training/rollout_failure/refill_rounds"] == 1


def test_metrics_aggregator_sums_rollout_failure_counts():
    aggregator = DiffusionMetricsAggregator()
    aggregator.add_step_metrics(
        {
            "training/rollout_failure/refilled_prompts": 1,
            "training/off_policy/dropped_samples": 2,
            "training/off_policy/dropped_samples_staleness/mean": 10.0,
            "actor/loss/mean": 2.0,
            "actor/lr": 1e-4,
        },
        sample_count=1,
    )
    aggregator.add_step_metrics(
        {
            "training/rollout_failure/refilled_prompts": 3,
            "training/off_policy/dropped_samples": 6,
            "training/off_policy/dropped_samples_staleness/mean": 2.0,
            "actor/loss/mean": 4.0,
            "actor/lr": 2e-4,
        },
        sample_count=3,
    )

    metrics = aggregator.get_aggregated_metrics()
    assert metrics["training/rollout_failure/refilled_prompts"] == 4
    assert metrics["training/off_policy/dropped_samples"] == 8
    assert metrics["training/off_policy/dropped_samples_staleness/mean"] == pytest.approx(4.0)
    assert metrics["actor/loss/mean"] == pytest.approx(3.5)
    assert metrics["actor/lr"] == pytest.approx(2e-4)


class _FakeSnapshotWorkerGroup:
    def __init__(self):
        self.current = "W0"
        self.snapshots = {}

    def save_model_to_cpu(self, snapshot_id):
        self.snapshots[snapshot_id] = self.current

    def restore_model_from_cpu(self, snapshot_id):
        self.current = self.snapshots[snapshot_id]

    def clear_cpu_model(self, snapshot_id):
        self.snapshots.pop(snapshot_id, None)


def test_old_policy_is_stable_across_local_updates(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.parameter_sync_step = 3
    trainer.actor_rollout_wg = _FakeSnapshotWorkerGroup()

    def compute_old_log_prob(self, data):
        del data
        return DataProto(meta_info={"policy_version": self.actor_rollout_wg.current})

    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "_compute_old_log_prob", compute_old_log_prob)

    versions = []
    for local_step, current_version in enumerate(("W0", "W1", "W2")):
        trainer.local_trigger_step = local_step
        trainer.actor_rollout_wg.current = current_version
        result = trainer._compute_old_log_prob(DataProto())
        versions.append(result.meta_info["policy_version"])
        assert trainer.actor_rollout_wg.current == current_version

    assert versions == ["W0", "W0", "W0"]
    assert trainer.actor_rollout_wg.snapshots == {}


def test_on_step_end_syncs_every_outer_step():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    events = []
    trainer.global_steps = 1
    trainer.timing_raw = {}
    trainer.standalone_checkpoint_manager = SimpleNamespace(
        update_weights=lambda global_steps: events.append(("sync", global_steps))
    )
    trainer.sync_compatible = True
    trainer._standalone_paused = True

    def resume():
        events.append(("resume",))
        trainer._standalone_paused = False

    trainer._resume_standalone_generation = resume

    trainer.on_step_end()
    trainer.global_steps = 2
    trainer.on_step_end()

    assert events == [("sync", 1), ("resume",), ("sync", 2)]


@pytest.mark.parametrize(
    ("strategy", "save_handler_name", "restore_handler_name"),
    [
        ("fsdp", "_fsdp1_sharded_save_to_cpu", "_fsdp1_sharded_load_from_cpu"),
        ("fsdp2", "fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu"),
        ("veomni", "fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu"),
    ],
)
def test_snapshot_worker_selects_sharded_strategy_handlers(strategy, save_handler_name, restore_handler_name):
    worker = object.__new__(DiffusionDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": strategy}})
    worker._strategy_handlers = None

    save_handler, restore_handler = worker._get_strategy_handlers()

    assert save_handler.__name__ == save_handler_name
    assert restore_handler.__name__ == restore_handler_name


def test_snapshot_worker_materializes_and_reoffloads_parameters(monkeypatch):
    worker = object.__new__(DiffusionDetachActorWorker)
    worker.config = OmegaConf.create({"actor": {"strategy": "fsdp2"}})
    module = torch.nn.Linear(1, 1, bias=False)
    device_transitions = []

    class _FakeEngine:
        is_param_offload_enabled = True

        def __init__(self):
            self.module = module

        def to(self, device, *, model, optimizer, grad):
            device_transitions.append((device, model, optimizer, grad))

    worker.actor = SimpleNamespace(engine=_FakeEngine())
    worker._strategy_handlers = (
        lambda actor_module: (actor_module.weight.detach().clone(), "test-spec"),
        lambda actor_module, state, _global_spec: actor_module.weight.data.copy_(state),
    )
    worker.cpu_saved_models = {}
    monkeypatch.setattr(detach_actor_worker_module, "get_device_name", lambda: "cuda")

    worker.save_model_to_cpu(0)
    worker.restore_model_from_cpu(0)

    assert device_transitions == [
        ("cuda", True, False, False),
        ("cpu", True, False, False),
        ("cuda", True, False, False),
        ("cpu", True, False, False),
    ]
