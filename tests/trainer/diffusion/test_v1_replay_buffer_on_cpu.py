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

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


def _load_replay_buffer_module():
    path = Path(__file__).parents[3] / "verl_omni/trainer/diffusion/v1/replay_buffer.py"
    spec = importlib.util.spec_from_file_location("diffusion_v1_replay_buffer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


replay_buffer_module = _load_replay_buffer_module()
DiffusionReplayBuffer = replay_buffer_module.DiffusionReplayBuffer


class _FakeTransferQueue:
    def __init__(self, items):
        self.items = {"train": items, "val": {}}

    def kv_list(self):
        return deepcopy(self.items)

    def kv_clear(self, *, partition_id, keys):
        for key in keys:
            self.items.setdefault(partition_id, {}).pop(key, None)

    def add_group(self, uid: str, *, status: str, trajectories: int, reason: str | None = None):
        prompt_tag = {"is_prompt": True, "status": status, "global_steps": 1}
        if reason is not None:
            prompt_tag["failure_reason"] = reason
        self.items["train"][uid] = prompt_tag
        for session_id in range(trajectories):
            self.items["train"][f"{uid}_{session_id}_0"] = {
                "is_prompt": False,
                "global_steps": 1,
                "seq_len": 1,
            }


def _make_buffer(*, refill_fn, max_refill_rounds=3, train_batch_size=3):
    return DiffusionReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0,
        refill_fn=refill_fn,
        drop_incomplete_groups=True,
        max_incomplete_group_refill_rounds=max_refill_rounds,
        train_batch_size=train_batch_size,
    )


def _make_default_buffer():
    return DiffusionReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0,
    )


def _make_upstream_buffer():
    return ReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0,
    )


def _patch_transfer_queue(monkeypatch, fake_tq):
    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", fake_tq.kv_list)
    monkeypatch.setattr(replay_buffer_module.tq, "kv_clear", fake_tq.kv_clear)


def test_sample_evicts_partial_failure_and_refills_exact_prompt_count(monkeypatch):
    main_tq = _FakeTransferQueue({})
    main_tq.add_group("finished", status="finished", trajectories=2)
    main_tq.add_group("partial", status="failure", trajectories=1, reason="runtime_error")
    _patch_transfer_queue(monkeypatch, main_tq)

    main_batch, main_metrics = _make_upstream_buffer().sample(
        global_steps=1,
        partition_id="train",
        batch_size=2,
    )

    assert {key.rsplit("_", 2)[0] for key in main_batch.keys} == {"finished", "partial"}
    assert len(main_batch.keys) == 3
    assert main_metrics == {}

    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("finished", status="finished", trajectories=2)
    fake_tq.add_group("partial", status="failure", trajectories=1, reason="runtime_error")
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        for index in range(num_prompts):
            fake_tq.add_group(f"refill-{index}", status="finished", trajectories=2)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    batch, metrics = _make_buffer(refill_fn=refill).sample(global_steps=1, partition_id="train", batch_size=2)

    assert refill_calls == [1]
    assert {key.rsplit("_", 2)[0] for key in batch.keys} == {"finished", "refill-0"}
    assert "partial" not in fake_tq.items["train"]
    assert "partial_0_0" not in fake_tq.items["train"]
    assert metrics == {
        "training/rollout_failure/evicted_groups": 1,
        "training/rollout_failure/evicted_trajectories": 1,
        "training/rollout_failure/reason/runtime_error_groups": 1,
        "training/rollout_failure/refilled_prompts": 1,
        "training/rollout_failure/refill_rounds": 1,
    }


def test_sample_refills_multiple_failed_groups_in_one_exact_call(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("finished", status="finished", trajectories=1)
    fake_tq.add_group("failed-a", status="failure", trajectories=0, reason="timeout")
    fake_tq.add_group("failed-b", status="failure", trajectories=1, reason="cancelled")
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        for index in range(num_prompts):
            fake_tq.add_group(f"replacement-{index}", status="finished", trajectories=1)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    batch, metrics = _make_buffer(refill_fn=refill).sample(global_steps=1, partition_id="train", batch_size=3)

    assert refill_calls == [2]
    assert len({key.rsplit("_", 2)[0] for key in batch.keys}) == 3
    assert metrics["training/rollout_failure/evicted_groups"] == 2
    assert metrics["training/rollout_failure/refilled_prompts"] == 2
    assert metrics["training/rollout_failure/reason/timeout_groups"] == 1
    assert metrics["training/rollout_failure/reason/cancelled_groups"] == 1


def test_sample_stops_after_bounded_failed_refill_rounds(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed-initial", status="failure", trajectories=0, reason="runtime_error")
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        fake_tq.add_group(f"failed-refill-{len(refill_calls)}", status="failure", trajectories=0, reason="timeout")
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    buffer = _make_buffer(refill_fn=refill, max_refill_rounds=2)

    with pytest.raises(RuntimeError, match="Exceeded max_incomplete_group_refill_rounds=2"):
        buffer.sample(global_steps=1, partition_id="train", batch_size=1)

    assert refill_calls == [1, 1]


def test_sample_accumulates_metrics_across_successive_refill_rounds(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed-initial", status="failure", trajectories=1, reason="runtime_error")
    refill_calls = []

    def refill(num_prompts):
        refill_calls.append(num_prompts)
        if len(refill_calls) == 1:
            fake_tq.add_group("failed-again", status="failure", trajectories=0, reason="timeout")
        else:
            fake_tq.add_group("replacement", status="finished", trajectories=1)
        return num_prompts

    _patch_transfer_queue(monkeypatch, fake_tq)
    batch, metrics = _make_buffer(refill_fn=refill).sample(global_steps=1, partition_id="train", batch_size=1)

    assert batch.keys == ["replacement_0_0"]
    assert refill_calls == [1, 1]
    assert metrics["training/rollout_failure/evicted_groups"] == 2
    assert metrics["training/rollout_failure/evicted_trajectories"] == 1
    assert metrics["training/rollout_failure/refilled_prompts"] == 2
    assert metrics["training/rollout_failure/refill_rounds"] == 2
    assert metrics["training/rollout_failure/reason/runtime_error_groups"] == 1
    assert metrics["training/rollout_failure/reason/timeout_groups"] == 1


def test_validation_preserves_upstream_failure_sampling_without_refill(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.items["val"] = {
        "failed": {"is_prompt": True, "status": "failure", "global_steps": 1},
        "failed_0_0": {"is_prompt": False, "global_steps": 1, "seq_len": 1},
    }
    refill_calls = []
    _patch_transfer_queue(monkeypatch, fake_tq)

    batch, metrics = _make_buffer(refill_fn=lambda count: refill_calls.append(count)).sample(
        global_steps=1,
        partition_id="val",
        batch_size=1,
    )

    assert batch.keys == ["failed_0_0"]
    assert metrics == {}
    assert refill_calls == []


def test_disabled_policy_preserves_upstream_training_failure_sampling(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed", status="failure", trajectories=1, reason="runtime_error")
    _patch_transfer_queue(monkeypatch, fake_tq)

    batch, metrics = _make_default_buffer().sample(global_steps=1, partition_id="train", batch_size=1)

    assert batch.keys == ["failed_0_0"]
    assert metrics == {}


def test_sample_rejects_non_exact_refill_result(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    fake_tq.add_group("failed", status="failure", trajectories=0, reason="runtime_error")
    _patch_transfer_queue(monkeypatch, fake_tq)

    with pytest.raises(RuntimeError, match="refill_fn submitted 0 prompts, expected 1"):
        _make_buffer(refill_fn=lambda _count: 0).sample(global_steps=1, partition_id="train", batch_size=1)


def test_sample_rejects_refill_above_total_prompt_budget(monkeypatch):
    fake_tq = _FakeTransferQueue({})
    for index in range(4):
        fake_tq.add_group(f"failed-{index}", status="failure", trajectories=0, reason="runtime_error")
    refill_calls = []
    _patch_transfer_queue(monkeypatch, fake_tq)

    with pytest.raises(RuntimeError, match="exceeding the bounded budget 3"):
        _make_buffer(refill_fn=lambda count: refill_calls.append(count), train_batch_size=1).sample(
            global_steps=1,
            partition_id="train",
            batch_size=1,
        )

    assert refill_calls == []


def test_init_rejects_missing_refill_fn_and_invalid_round_limit():
    with pytest.raises(ValueError, match="drop_incomplete_groups requires refill_fn"):
        DiffusionReplayBuffer(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=8,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            drop_incomplete_groups=True,
            train_batch_size=1,
        )

    with pytest.raises(ValueError, match="must be a positive integer"):
        DiffusionReplayBuffer(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=8,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            max_incomplete_group_refill_rounds=0,
        )
