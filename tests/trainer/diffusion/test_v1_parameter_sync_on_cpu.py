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
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

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
                    "separate_async": {
                        "parameter_sync_step": parameter_sync_step,
                        "num_warmup_batches": 0,
                        "sync_compatible": True,
                    },
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
    warmup_config = _separate_async_config()
    warmup_config.trainer.v1.separate_async.num_warmup_batches = 1
    invalid_configs.append((warmup_config, "requires num_warmup_batches=0"))
    for config, error in invalid_configs:
        with pytest.raises(AssertionError, match=error):
            PolicyGradientDiffusionTrainerV1SeparateAsync(config)


def test_separate_async_uses_upstream_async_replay_buffer():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer._add_prompts_to_generate = lambda count: count
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

    assert isinstance(replay_buffer, ReplayBufferAsync)
    assert replay_buffer.refill_fn is trainer._add_prompts_to_generate


def test_base_step_samples_one_mini_batch_per_local_update():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 8}})
    trainer.parameter_sync_step = 4
    trainer.sync_compatible = False
    sample_sizes = []
    local_steps = []

    trainer._add_batch_to_generate = lambda: None

    def step_once(iter_metrics, timing_raw, sample_batch_size):
        del timing_raw
        sample_sizes.append(sample_batch_size)
        local_steps.append(trainer.local_trigger_step)
        iter_metrics["actor/loss/mean"] = float(trainer.local_trigger_step)
        iter_metrics["training/off_policy/evicted_samples"] = 1
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
    assert metrics["training/off_policy/evicted_samples"] == 4


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


def test_separate_async_uses_single_prompt_generation_batches_for_exact_refill():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8, "gen_batch_size": 8},
            "trainer": {"v1": {"sampler": {"drop_incomplete_groups": False}}},
        }
    )

    assert trainer._generation_batch_size() == 1


def test_sync_compatible_prefetches_all_local_batches_before_training():
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 4}})
    trainer.parameter_sync_step = 2
    trainer.sync_compatible = True
    events = []
    trainer._add_batch_to_generate = lambda: events.append("feed")
    trainer.on_sample_begin = lambda: events.append("sample_begin")
    trainer.on_sample_end = lambda: events.append("sample_end")

    def sample(batch_size):
        step = trainer.local_trigger_step
        events.append(f"sample_{step}")
        return (
            KVBatchMeta(
                partition_id="train",
                keys=[f"sample-{step}"],
                tags=[{"is_padding": False}],
            ),
            {},
        )

    def train(iter_metrics, _timing_raw, batch):
        events.append(f"train_{trainer.local_trigger_step}")
        iter_metrics["actor/loss/mean"] = float(trainer.local_trigger_step)
        return batch

    trainer._sample_training_batch = sample
    trainer._train_sampled_batch = train

    metrics = {}
    batch = PolicyGradientDiffusionTrainerV1.step(trainer, metrics, {})

    assert events == ["feed", "sample_begin", "sample_0", "sample_1", "sample_end", "train_0", "train_1"]
    assert batch.keys == ["sample-0", "sample-1"]
    assert metrics["actor/loss/mean"] == pytest.approx(0.5)


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
        ("fsdp", "fsdp1_sharded_save_to_cpu", "fsdp1_sharded_load_from_cpu"),
        ("fsdp2", "fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu"),
        ("veomni", "fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu"),
    ],
)
def test_snapshot_worker_selects_sharded_strategy_handlers(strategy, save_handler_name, restore_handler_name):
    assert issubclass(DiffusionDetachActorWorker, DetachActorWorker)
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
        lambda actor_module: (actor_module.weight.detach(), "test-spec"),
        lambda actor_module, state, _global_spec: actor_module.weight.data.copy_(state),
    )
    worker.cpu_saved_models = {}
    monkeypatch.setattr(detach_actor_worker_module, "get_device_name", lambda: "cuda")

    original_weight = module.weight.detach().clone()
    worker.save_model_to_cpu(0)
    with torch.no_grad():
        module.weight.add_(1)
    assert torch.equal(worker.cpu_saved_models[0][0], original_weight)
    worker.restore_model_from_cpu(0)

    assert torch.equal(module.weight, original_weight)
    assert device_transitions == [
        ("cuda", True, False, False),
        ("cpu", True, False, False),
        ("cuda", True, False, False),
        ("cpu", True, False, False),
    ]
