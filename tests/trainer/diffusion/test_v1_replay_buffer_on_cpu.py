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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync

from verl_omni.trainer.diffusion.v1 import trainer_base as trainer_base_module
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1


class _FakeTransferQueue:
    def __init__(self, items):
        self.items = {"train": items, "val": {}}

    def kv_list(self):
        return deepcopy(self.items)

    def kv_clear(self, *, partition_id, keys):
        for key in keys:
            self.items.setdefault(partition_id, {}).pop(key, None)

    def add_group(
        self,
        uid: str,
        *,
        status: str,
        trajectories: int,
        global_steps: int = 1,
    ):
        self.items["train"][uid] = {"is_prompt": True, "status": status, "global_steps": global_steps}
        for session_id in range(trajectories):
            self.items["train"][f"{uid}_{session_id}_0"] = {
                "is_prompt": False,
                "global_steps": global_steps,
                "seq_len": 1,
            }


def _make_config(
    *,
    drop_incomplete_groups: bool,
    trainer_mode: str = "sync",
    max_refill_rounds: int = 3,
    train_batch_size: int = 3,
):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": train_batch_size},
            "trainer": {
                "v1": {
                    "trainer_mode": trainer_mode,
                    trainer_mode: {"parameter_sync_step": 1},
                    "sampler": {
                        "max_off_policy_threshold": 8,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                        "drop_incomplete_groups": drop_incomplete_groups,
                        "max_incomplete_group_refill_rounds": max_refill_rounds,
                    },
                }
            },
        }
    )


def _make_upstream_buffer(trainer_mode="sync"):
    return ReplayBuffer(
        trainer_mode=trainer_mode,
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0,
    )


def _make_trainer(
    *,
    refill_fn,
    drop_incomplete_groups=True,
    max_refill_rounds=3,
    train_batch_size=3,
    trainer_mode="sync",
):
    return SimpleNamespace(
        config=_make_config(
            drop_incomplete_groups=drop_incomplete_groups,
            trainer_mode=trainer_mode,
            max_refill_rounds=max_refill_rounds,
            train_batch_size=train_batch_size,
        ),
        trainer_mode=trainer_mode,
        global_steps=1,
        replay_buffer=_make_upstream_buffer(trainer_mode),
        _add_prompts_to_generate=refill_fn,
        _trajectory_uid=PolicyGradientDiffusionTrainerV1._trajectory_uid,
    )


def _patch_transfer_queue(monkeypatch, fake_tq):
    monkeypatch.setattr(trainer_base_module.tq, "kv_list", fake_tq.kv_list)
    monkeypatch.setattr(trainer_base_module.tq, "kv_clear", fake_tq.kv_clear)


def _sample(trainer, batch_size):
    return PolicyGradientDiffusionTrainerV1._sample_training_batch(trainer, batch_size)


def test_wandb_validation_logging_rejects_non_uint8_images():
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"log_val_generations": 1, "logger": ["wandb"]}}),
    )

    with pytest.raises(ValueError, match=r"Expected a uint8 image tensor, got torch\.float32\."):
        PolicyGradientDiffusionTrainerV1._maybe_log_val_generations(
            trainer,
            inputs=["prompt"],
            outputs=torch.zeros(1, 3, 8, 8),
            scores=[0.0],
        )


def test_v1_generation_dump_rejects_non_uint8_outputs_before_submission():
    trainer = SimpleNamespace()

    with pytest.raises(ValueError, match=r"Expected generation outputs to be a uint8 tensor, got torch\.float32\."):
        PolicyGradientDiffusionTrainerV1._dump_generations(
            trainer,
            inputs=["prompt"],
            outputs=torch.zeros(1, 3, 8, 8),
            gts=[""],
            scores=[0.0],
            reward_extra_infos_dict={},
            dump_path="unused",
        )


def test_sample_evicts_partial_failure_and_refills_exact_prompt_count(monkeypatch):
    main_tq = _FakeTransferQueue({})
    main_tq.add_group("finished", status="finished", trajectories=2)
    main_tq.add_group("partial", status="failure", trajectories=1)
    _patch_transfer_queue(monkeypatch, main_tq)

    main_batch, main_metrics = _make_upstream_buffer().sample(global_steps=1, partition_id="train", batch_size=2)

    assert {key.rsplit("_", 2)[0] for key in main_batch.keys} == {"finished", "partial"}
    assert len(main_batch.keys) == 3
    assert main_metrics == {}

    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("finished", status="finished", trajectories=2)
    fake_tq.add_group("partial", status="failure", trajectories=1)
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        fake_tq.add_group("replacement", status="finished", trajectories=2)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    batch, metrics = _sample(_make_trainer(refill_fn=refill), batch_size=2)

    assert refill_calls == [1]
    assert {key.rsplit("_", 2)[0] for key in batch.keys} == {"finished", "replacement"}
    assert "partial" not in fake_tq.items["train"]
    assert "partial_0_0" not in fake_tq.items["train"]
    assert metrics == {
        "training/rollout_failure/evicted_groups": 1,
        "training/rollout_failure/evicted_trajectories": 1,
        "training/rollout_failure/refilled_prompts": 1,
        "training/rollout_failure/refill_rounds": 1,
    }


def test_sample_refills_multiple_failed_groups_in_one_exact_call(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("finished", status="finished", trajectories=1)
    fake_tq.add_group("failed-a", status="failure", trajectories=0)
    fake_tq.add_group("failed-b", status="failure", trajectories=1)
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        for index in range(num_prompts):
            fake_tq.add_group(f"replacement-{index}", status="finished", trajectories=1)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    batch, metrics = _sample(_make_trainer(refill_fn=refill), batch_size=3)

    assert refill_calls == [2]
    assert len({key.rsplit("_", 2)[0] for key in batch.keys}) == 3
    assert metrics["training/rollout_failure/evicted_groups"] == 2
    assert metrics["training/rollout_failure/refilled_prompts"] == 2


def test_sample_stops_after_bounded_failed_refill_rounds(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed-initial", status="failure", trajectories=0)
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        fake_tq.add_group(f"failed-refill-{len(refill_calls)}", status="failure", trajectories=0)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    trainer = _make_trainer(refill_fn=refill, max_refill_rounds=2)

    with pytest.raises(RuntimeError, match="Exceeded max_incomplete_group_refill_rounds=2"):
        _sample(trainer, batch_size=1)

    assert refill_calls == [1, 1]


def test_sample_accumulates_metrics_across_successive_refill_rounds(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed-initial", status="failure", trajectories=1)
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        if len(refill_calls) == 1:
            fake_tq.add_group("failed-again", status="failure", trajectories=0)
        else:
            fake_tq.add_group("replacement", status="finished", trajectories=1)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    batch, metrics = _sample(_make_trainer(refill_fn=refill), batch_size=1)

    assert batch.keys == ["replacement_0_0"]
    assert refill_calls == [1, 1]
    assert metrics == {
        "training/rollout_failure/evicted_groups": 2,
        "training/rollout_failure/evicted_trajectories": 1,
        "training/rollout_failure/refilled_prompts": 2,
        "training/rollout_failure/refill_rounds": 2,
    }


def test_sample_does_not_refill_unselected_failed_groups(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("old-finished", status="finished", trajectories=1, global_steps=0)
    fake_tq.add_group("new-failed", status="failure", trajectories=0, global_steps=1)
    refill_calls = []
    _patch_transfer_queue(monkeypatch, fake_tq)

    batch, metrics = _sample(
        _make_trainer(refill_fn=lambda count: refill_calls.append(count), train_batch_size=1),
        batch_size=1,
    )

    assert batch.keys == ["old-finished_0_0"]
    assert metrics == {}
    assert refill_calls == []
    assert "new-failed" in fake_tq.items["train"]


def test_validation_preserves_upstream_failure_sampling(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.items["val"] = {
        "failed": {"is_prompt": True, "status": "failure", "global_steps": 1},
        "failed_0_0": {"is_prompt": False, "global_steps": 1, "seq_len": 1},
    }
    _patch_transfer_queue(monkeypatch, fake_tq)

    batch, metrics = _make_upstream_buffer().sample(global_steps=1, partition_id="val", batch_size=1)

    assert batch.keys == ["failed_0_0"]
    assert metrics == {}


def test_disabled_sync_policy_preserves_upstream_training_failure_sampling(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed", status="failure", trajectories=1)
    _patch_transfer_queue(monkeypatch, fake_tq)
    trainer = _make_trainer(refill_fn=None, drop_incomplete_groups=False)

    batch, metrics = _sample(trainer, batch_size=1)

    assert batch.keys == ["failed_0_0"]
    assert metrics == {}


def test_separate_async_uses_upstream_exact_eviction_and_refill(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("stale", status="finished", trajectories=1, global_steps=0)
    fake_tq.add_group("failed", status="failure", trajectories=1, global_steps=5)
    fake_tq.add_group("fresh", status="finished", trajectories=1, global_steps=5)
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        for index in range(num_prompts):
            fake_tq.add_group(f"replacement-{index}", status="finished", trajectories=1, global_steps=5)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    config = _make_config(
        drop_incomplete_groups=False,
        trainer_mode="separate_async",
        train_batch_size=3,
    )
    config.trainer.v1.sampler.max_off_policy_threshold = 2
    trainer = SimpleNamespace(
        config=config,
        trainer_mode="separate_async",
        _add_prompts_to_generate=refill,
    )
    replay_buffer = PolicyGradientDiffusionTrainerV1._build_replay_buffer(trainer)
    replay_buffer.poll_interval = 0

    batch, metrics = replay_buffer.sample(global_steps=5, partition_id="train", batch_size=3)

    assert refill_calls == [2]
    assert {key.rsplit("_", 2)[0] for key in batch.keys} == {"fresh", "replacement-0", "replacement-1"}
    assert metrics["training/off_policy/evicted_samples"] == 1
    assert metrics["training/off_policy/evicted_samples_staleness/mean"] == pytest.approx(6.0)
    assert metrics["training/rollout_failure/evicted_samples"] == 1


@pytest.mark.parametrize(
    ("trainer_mode", "drop_incomplete_groups", "expected_type"),
    [
        ("sync", False, ReplayBuffer),
        ("separate_async", False, ReplayBufferAsync),
        ("sync", True, ReplayBuffer),
    ],
)
def test_trainer_factory_uses_upstream_replay_buffer(trainer_mode, drop_incomplete_groups, expected_type):
    config = _make_config(
        drop_incomplete_groups=drop_incomplete_groups,
        trainer_mode=trainer_mode,
    )
    trainer = SimpleNamespace(config=config, trainer_mode=trainer_mode, _add_prompts_to_generate=lambda count: count)

    replay_buffer = PolicyGradientDiffusionTrainerV1._build_replay_buffer(trainer)

    assert type(replay_buffer) is expected_type


def test_sample_rejects_non_exact_refill_result(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed", status="failure", trajectories=0)
    _patch_transfer_queue(monkeypatch, fake_tq)

    with pytest.raises(RuntimeError, match="refill submitted 0 prompts, expected 1"):
        _sample(_make_trainer(refill_fn=lambda _count: 0), batch_size=1)


def test_factory_rejects_non_sync_policy_and_invalid_round_limit():
    config = _make_config(drop_incomplete_groups=True, trainer_mode="separate_async")
    trainer = SimpleNamespace(config=config, trainer_mode="separate_async")
    with pytest.raises(ValueError, match="only supported with trainer_mode='sync'"):
        PolicyGradientDiffusionTrainerV1._build_replay_buffer(trainer)

    config = _make_config(drop_incomplete_groups=True, max_refill_rounds=0)
    trainer = SimpleNamespace(config=config, trainer_mode="sync")
    with pytest.raises(ValueError, match="must be a positive integer"):
        PolicyGradientDiffusionTrainerV1._build_replay_buffer(trainer)
