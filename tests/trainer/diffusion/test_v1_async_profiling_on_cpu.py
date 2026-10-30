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
"""CPU tests for the always-on async profiling of issue #712.

Covers the sample-wait decomposition of ``ProfiledReplayBufferAsync``
(slow-generation vs coarse-polling counterexamples, completion-tag
transitions, threshold-crossing detection, sync-disjoint segments), the
rollout-side TQ write-tag round trip, the trainer-side payload byte count,
and the per-phase weight-sync timing mixin against a stub manager whose call
shapes mirror the pinned verl dependency (no Ray or engine backend needed).
"""

import asyncio
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from verl.trainer.ppo.v1 import replay_buffer as replay_buffer_module

from verl_omni.trainer.diffusion.v1.async_profiling import (
    TQ_PAYLOAD_BYTES_TAG,
    TQ_PUT_AT_TAG,
    TQ_PUT_SECONDS_TAG,
    ProfiledReplayBufferAsync,
    aggregate_tq_write_stats,
    derive_tq_put_rate,
    tensordict_payload_bytes,
)
from verl_omni.workers.checkpoint_engine import (
    TimedCheckpointEngineManager,
    WeightSyncPhaseTimerMixin,
)


class _FakeTransferQueue:
    """Minimal TransferQueue stand-in covering the replay buffer's read path."""

    def __init__(self, items):
        self.items = {"train": items, "val": {}}

    def kv_list(self):
        return deepcopy(self.items)

    def kv_clear(self, *, partition_id, keys):
        for key in keys:
            self.items.setdefault(partition_id, {}).pop(key, None)

    def add_finished_group(self, uid: str, trajectories: int = 1, global_steps: int = 1, put_at: float | None = None):
        self.items["train"][uid] = {"is_prompt": True, "status": "finished", "global_steps": global_steps}
        for session_id in range(trajectories):
            tag = {
                "is_prompt": False,
                "global_steps": global_steps,
                "seq_len": 1,
            }
            if put_at is not None:
                tag[TQ_PUT_AT_TAG] = put_at
            self.items["train"][f"{uid}_{session_id}_0"] = tag


def _make_profiled_buffer(poll_interval: float = 0.01) -> ProfiledReplayBufferAsync:
    return ProfiledReplayBufferAsync(
        trainer_mode="separate_async",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=poll_interval,
    )


def _patch_transfer_queue(monkeypatch, fake_tq):
    # Both verl's replay_buffer and verl-omni resolve `tq` to the same
    # transfer_queue module object, so patching it once covers the buffer.
    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", fake_tq.kv_list)
    monkeypatch.setattr(replay_buffer_module.tq, "kv_clear", fake_tq.kv_clear)


def test_sample_wait_decomposition_segments_sum_to_total(monkeypatch):
    """The published segments must be consistent after a polling sample."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)

    def finish_group_after_delay():
        time.sleep(0.05)
        fake_tq.add_finished_group("uida", put_at=time.time())

    import threading

    finisher = threading.Thread(target=finish_group_after_delay)
    finisher.start()

    buffer = _make_profiled_buffer(poll_interval=0.01)
    meta, _eviction = buffer.sample(global_steps=1, partition_id="train", batch_size=1)
    finisher.join()

    timing = buffer.last_sample_timing
    assert len(meta.keys) == 1
    # All accounting keys present and non-negative.
    for key in (
        "sample_wait/generation_bound",
        "sample_wait/detection",
        "sample_wait/metadata_sync",
        "sample_wait/eviction",
        "sample_wait/total",
        "sample_wait/polls",
    ):
        assert key in timing, f"missing {key} in {sorted(timing)}"
        assert timing[key] >= 0.0
    # generation_bound is the residual of total minus the measured segments.
    measured = sum(timing[f"sample_wait/{s}"] for s in ("detection", "metadata_sync", "eviction"))
    assert timing["sample_wait/generation_bound"] == pytest.approx(
        max(0.0, timing["sample_wait/total"] - measured), abs=1e-6
    )
    # We polled at least once before the group appeared, and every poll did a
    # metadata sync that the decomposition actually timed.
    assert timing["sample_wait/polls"] >= 1.0
    assert timing["sample_wait/metadata_sync"] > 0.0
    # Buffer-depth telemetry for the polls we waited through.
    assert timing["sample_wait/buffer_finished_max"] >= 0.0


def test_sample_wait_no_poll_when_data_ready(monkeypatch):
    """A ready buffer must report zero polls, wait, and detection latency."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    fake_tq.add_finished_group("uida", put_at=time.time())

    buffer = _make_profiled_buffer()
    buffer.sample(global_steps=1, partition_id="train", batch_size=1)

    timing = buffer.last_sample_timing
    assert timing["sample_wait/polls"] == 0.0
    assert timing["sample_wait/detection"] == 0.0
    assert timing["sample_wait/generation_bound"] == pytest.approx(0.0, abs=0.05)


def test_arrival_during_metadata_sync_is_not_double_counted(monkeypatch):
    """An arrival inside a slow metadata RPC is that RPC's service time.

    The undiscovered window of a row published mid-RPC lies inside the sync's
    own duration: charging it as detection on top of metadata_sync would book
    the same second twice. Detection stays at zero and the segment sums stay
    within the wall time.
    """
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    real_kv_list = fake_tq.kv_list

    def kv_list_publishing_row_mid_rpc():
        time.sleep(0.05)
        # The row lands about 0.03s into this RPC, well before its return.
        fake_tq.add_finished_group("uida", put_at=time.time() - 0.03)
        return real_kv_list()

    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", kv_list_publishing_row_mid_rpc)

    buffer = _make_profiled_buffer()
    buffer.sample(global_steps=1, partition_id="train", batch_size=1)

    timing = buffer.last_sample_timing
    assert timing["sample_wait/polls"] == 0.0
    assert timing["sample_wait/detection"] == 0.0
    assert timing["sample_wait/metadata_sync"] >= 0.04
    measured = sum(timing[f"sample_wait/{s}"] for s in ("detection", "metadata_sync", "eviction"))
    assert measured <= timing["sample_wait/total"] + 1e-9


def test_slow_generation_is_generation_bound_not_detection(monkeypatch):
    """Counterexample 1: slow generation must land in generation_bound.

    A group that finishes well into the polling wait is generation-bound:
    detection (completion-to-discovery) stays below one inter-sync window.
    """
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)

    def finish_group_late():
        time.sleep(0.2)
        fake_tq.add_finished_group("uida", put_at=time.time())

    import threading

    finisher = threading.Thread(target=finish_group_late)
    finisher.start()
    buffer = _make_profiled_buffer(poll_interval=0.02)
    buffer.sample(global_steps=1, partition_id="train", batch_size=1)
    finisher.join()

    timing = buffer.last_sample_timing
    assert timing["sample_wait/generation_bound"] >= 0.12
    # Detection is bounded by one poll cycle, not by the generation time.
    assert timing["sample_wait/detection"] <= 0.06
    assert timing["sample_wait/generation_bound"] > timing["sample_wait/detection"]


def test_coarse_polling_is_detection_not_generation_bound(monkeypatch):
    """Counterexample 2: coarse polling must land in detection.

    A group that finishes almost immediately is discovered only at the next
    poll boundary: most of the wait is detection latency, recoverable by a
    smaller poll interval.
    """
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)

    def finish_group_early():
        time.sleep(0.02)
        fake_tq.add_finished_group("uida", put_at=time.time())

    import threading

    finisher = threading.Thread(target=finish_group_early)
    finisher.start()
    buffer = _make_profiled_buffer(poll_interval=0.3)
    buffer.sample(global_steps=1, partition_id="train", batch_size=1)
    finisher.join()

    timing = buffer.last_sample_timing
    assert timing["sample_wait/detection"] >= 0.15
    assert timing["sample_wait/generation_bound"] <= 0.1
    assert timing["sample_wait/detection"] > timing["sample_wait/generation_bound"]


def test_rows_without_arrival_timestamp_fall_back_to_generation_bound(monkeypatch):
    """Rows from older writers carry no tq_put_at; their wait stays combined."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)

    def finish_group_late():
        time.sleep(0.1)
        fake_tq.add_finished_group("uida")  # no put_at stamp

    import threading

    finisher = threading.Thread(target=finish_group_late)
    finisher.start()
    buffer = _make_profiled_buffer(poll_interval=0.02)
    buffer.sample(global_steps=1, partition_id="train", batch_size=1)
    finisher.join()

    timing = buffer.last_sample_timing
    assert timing["sample_wait/detection"] == 0.0
    assert timing["sample_wait/generation_bound"] >= 0.06


def test_detection_charged_at_sampleability_transition(monkeypatch):
    """Staggered arrivals: only the completing group's discovery is detection.

    Two groups are required. A completes early and is discovered mid-wait,
    B completes the batch last: A's undiscovered window is generation-bound
    wait (the trainer could not have sampled anyway), so detection must cover
    only B's publication-to-discovery latency -- not the sum over both.
    """
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    real_kv_list = fake_tq.kv_list
    start = time.time()

    def kv_list_staggered():
        now = time.time()
        elapsed = now - start
        if elapsed > 0.10 and "uida" not in fake_tq.items["train"]:
            # A completed near the 0.02 boundary but is discovered only here.
            fake_tq.add_finished_group("uida", put_at=start + 0.02)
        if elapsed > 0.16 and "uidb" not in fake_tq.items["train"]:
            # B completes the requested batch right before this poll.
            fake_tq.add_finished_group("uidb", put_at=now - 0.01)
        return real_kv_list()

    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", kv_list_staggered)

    buffer = _make_profiled_buffer(poll_interval=0.02)
    meta, _eviction = buffer.sample(global_steps=1, partition_id="train", batch_size=2)

    timing = buffer.last_sample_timing
    assert len(meta.keys) == 2
    # B's latency only (~0.01s): charging A's ~0.08s of undiscovered wait
    # here would mislabel a rollout bottleneck as polling overhead.
    assert timing["sample_wait/detection"] < 0.04
    assert timing["sample_wait/generation_bound"] >= 0.05
    assert timing["sample_wait/generation_bound"] > timing["sample_wait/detection"]


def test_completion_tag_transition_counts_detection(monkeypatch):
    """A completion tag that lands on an already-visible group must stage
    exactly once, while the batch is still waiting for the others.

    The writer publishes rows, then the terminal status, then the telemetry
    tags -- so a poll can see a finished group without ``tq_put_at``. That
    group is sampleable already (no false wait), but once the batch needs
    more groups, its tag transition must not re-stage or drop it, and the
    completing group's discovery still carries the detection.
    """
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    real_kv_list = fake_tq.kv_list
    state = {"syncs": 0}

    def kv_list_publish_untagged_then_tag():
        state["syncs"] += 1
        if state["syncs"] == 1:
            # A's status published, its completion tag not yet landed.
            fake_tq.add_finished_group("uida")
        elif state["syncs"] == 2:
            # A's tag put lands mid-gap; B is still generating.
            for tag in fake_tq.items["train"].values():
                if not tag.get("is_prompt"):
                    tag[TQ_PUT_AT_TAG] = time.time()
        elif state["syncs"] == 3:
            # B completes the requested batch right before this poll.
            fake_tq.add_finished_group("uidb", put_at=time.time() - 0.005)
        return real_kv_list()

    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", kv_list_publish_untagged_then_tag)

    buffer = _make_profiled_buffer(poll_interval=0.02)
    meta, _eviction = buffer.sample(global_steps=1, partition_id="train", batch_size=2)

    timing = buffer.last_sample_timing
    assert len(meta.keys) == 2
    # B's publication-to-discovery latency (~0.005s) is the detection; A's
    # late-landing tag must not add a second charge on top of it.
    assert 0.0 < timing["sample_wait/detection"] <= 0.06


def test_detection_uses_the_group_that_completed_the_batch(monkeypatch):
    """Target 1 with two same-poll arrivals: detection runs from the earliest.

    A and B complete inside the same polling gap (at 0.10 and 0.20) and are
    discovered together at the 0.30 sync. The batch was ready at A, so
    detection is ~0.20 -- reading the latest arrival instead would report
    ~0.10 and bury a polling bottleneck.
    """
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    real_kv_list = fake_tq.kv_list
    start = time.time()

    def kv_list_surplus_group():
        if time.time() - start > 0.29 and "uida" not in fake_tq.items["train"]:
            fake_tq.add_finished_group("uida", put_at=start + 0.10)
            fake_tq.add_finished_group("uidb", put_at=start + 0.20)
        return real_kv_list()

    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", kv_list_surplus_group)

    buffer = _make_profiled_buffer(poll_interval=0.3)
    meta, _eviction = buffer.sample(global_steps=1, partition_id="train", batch_size=1)

    timing = buffer.last_sample_timing
    assert len(meta.keys) == 1
    assert timing["sample_wait/detection"] >= 0.15
    assert timing["sample_wait/generation_bound"] <= 0.12
    assert timing["sample_wait/detection"] > timing["sample_wait/generation_bound"]


def test_detection_crosses_threshold_for_target_two_of_three(monkeypatch):
    """Target 2 with three same-poll arrivals: the second earliest completes
    the batch.

    Arrivals at 0.10 / 0.20 / 0.28, all discovered together at the 0.30 sync.
    The threshold crossing is the second arrival (0.20), so detection is
    ~0.10 -- not the ~0.02 of the latest arrival, and not the ~0.30 of
    counting every discovered group as needed.
    """
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    real_kv_list = fake_tq.kv_list
    start = time.time()

    def kv_list_three_groups():
        if time.time() - start > 0.29 and "uida" not in fake_tq.items["train"]:
            fake_tq.add_finished_group("uida", put_at=start + 0.10)
            fake_tq.add_finished_group("uidb", put_at=start + 0.20)
            fake_tq.add_finished_group("uidc", put_at=start + 0.28)
        return real_kv_list()

    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", kv_list_three_groups)

    buffer = _make_profiled_buffer(poll_interval=0.3)
    meta, _eviction = buffer.sample(global_steps=1, partition_id="train", batch_size=2)

    timing = buffer.last_sample_timing
    assert len(meta.keys) == 2
    assert 0.06 <= timing["sample_wait/detection"] <= 0.15


def test_incidental_metadata_syncs_are_not_profiled(monkeypatch):
    """get_sampleable_count runs outside the profiling window by design."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    fake_tq.add_finished_group("uida", put_at=time.time())

    buffer = _make_profiled_buffer()
    buffer.get_sampleable_count(1, "train")  # syncs metadata, must not accumulate
    assert buffer.last_sample_timing == {}


def test_aggregate_tq_write_stats_sums_tags():
    meta = SimpleNamespace(
        tags=[
            {TQ_PUT_SECONDS_TAG: 0.5, TQ_PAYLOAD_BYTES_TAG: 100},
            {TQ_PUT_SECONDS_TAG: 1.5, TQ_PAYLOAD_BYTES_TAG: 300},
            {"status": "success"},  # rows from older writers are skipped
            {TQ_PUT_SECONDS_TAG: 1.0, TQ_PAYLOAD_BYTES_TAG: 0},
        ]
    )
    stats = aggregate_tq_write_stats(meta)
    assert stats["tq/put/rows"] == 3.0
    assert stats["tq/put/seconds"] == pytest.approx(3.0)
    assert stats["tq/put/bytes"] == 400.0
    # The rate is a whole-step property derived from the summed counters, not
    # something the aggregator could average per update.
    assert "tq/put/bytes_per_second" not in stats


def test_sample_profile_depth_stats_combine_across_local_updates():
    """parameter_sync_step > 1 records one profile per local update into the
    same ``timing_raw``: the wall-time segments and poll count sum, but the
    buffer-depth stats must keep their own semantics -- min stays the minimum,
    max the maximum, and the means combine weighted by each sample's polls.
    """
    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

    def record(timing_raw, sample_timing):
        trainer = SimpleNamespace(replay_buffer=SimpleNamespace(last_sample_timing=sample_timing))
        PolicyGradientDiffusionTrainerV1._record_async_sample_profile(
            trainer, {}, timing_raw, SimpleNamespace(tags=[])
        )

    first = {
        "sample_wait/detection": 0.2,
        "sample_wait/metadata_sync": 0.05,
        "sample_wait/generation_bound": 1.0,
        "sample_wait/total": 1.25,
        "sample_wait/polls": 3.0,
        "sample_wait/buffer_finished_mean": 2.0,
        "sample_wait/buffer_finished_min": 1.0,
        "sample_wait/buffer_finished_max": 4.0,
        "sample_wait/buffer_pending_mean": 5.0,
        "sample_wait/buffer_running_mean": 7.0,
    }
    second = {
        "sample_wait/detection": 0.1,
        "sample_wait/metadata_sync": 0.05,
        "sample_wait/generation_bound": 0.5,
        "sample_wait/total": 0.65,
        "sample_wait/polls": 1.0,
        "sample_wait/buffer_finished_mean": 10.0,
        "sample_wait/buffer_finished_min": 8.0,
        "sample_wait/buffer_finished_max": 12.0,
        "sample_wait/buffer_pending_mean": 3.0,
        "sample_wait/buffer_running_mean": 9.0,
    }

    timing_raw = {}
    record(timing_raw, first)
    record(timing_raw, second)

    assert timing_raw["sample_wait/total"] == pytest.approx(1.9)
    assert timing_raw["sample_wait/polls"] == 4.0
    # The depth stats keep their semantics instead of summing: summing turned
    # min into min1+min2 (9.0) and doubled every mean.
    assert timing_raw["sample_wait/buffer_finished_min"] == 1.0
    assert timing_raw["sample_wait/buffer_finished_max"] == 12.0
    assert timing_raw["sample_wait/buffer_finished_mean"] == pytest.approx((2.0 * 3 + 10.0 * 1) / 4)
    assert timing_raw["sample_wait/buffer_pending_mean"] == pytest.approx((5.0 * 3 + 3.0 * 1) / 4)
    assert timing_raw["sample_wait/buffer_running_mean"] == pytest.approx((7.0 * 3 + 9.0 * 1) / 4)


def test_tq_put_counters_sum_across_local_updates():
    """With parameter_sync_step > 1 the counters must sum, not weighted-average.

    Two one-row updates carrying 100 bytes in 1 s and 2 s must log 2 rows /
    200 bytes / 3 s, with the rate derived from those sums -- averaging the
    per-update rates (66.7 and 50 B/s) would misstate the throughput.
    """
    from verl.trainer.ppo.v1.utils import MetricsAggregator

    from verl_omni.trainer.diffusion.v1.trainer_base import configure_step_metrics_aggregator

    aggregator = configure_step_metrics_aggregator(MetricsAggregator())
    aggregator.add_step_metrics(
        {"tq/put/rows": 1.0, "tq/put/seconds": 1.0, "tq/put/bytes": 100.0, "tq/get/bytes": 500.0},
        sample_count=1,
    )
    aggregator.add_step_metrics(
        {"tq/put/rows": 1.0, "tq/put/seconds": 2.0, "tq/put/bytes": 100.0, "tq/get/bytes": 1500.0},
        sample_count=1,
    )
    step_metrics = aggregator.get_aggregated_metrics()
    derive_tq_put_rate(step_metrics)
    assert step_metrics["tq/put/rows"] == 2.0
    assert step_metrics["tq/put/seconds"] == pytest.approx(3.0)
    assert step_metrics["tq/put/bytes"] == 200.0
    assert step_metrics["tq/get/bytes"] == 2000.0
    assert step_metrics["tq/put/bytes_per_second"] == pytest.approx(200.0 / 3.0)


def test_aggregate_tq_write_stats_empty_and_untagged():
    assert aggregate_tq_write_stats(SimpleNamespace(tags=[])) == {
        "tq/put/rows": 0.0,
        "tq/put/seconds": 0.0,
        "tq/put/bytes": 0.0,
    }
    # Missing tags attribute (sync-mode metas) degrades to zeros, not an error.
    stats = aggregate_tq_write_stats(SimpleNamespace())
    assert stats["tq/put/bytes"] == 0.0


def test_tensordict_payload_bytes_counts_tensor_columns():
    from tensordict import TensorDict

    batch = TensorDict(
        {
            "latents": torch.zeros(4, 8, 8, dtype=torch.float32),
            "ids": torch.zeros(4, 16, dtype=torch.int64),
        },
        batch_size=4,
    )
    # 4*8*8*4 + 4*16*8 = 1024 + 512
    assert tensordict_payload_bytes(batch) == 1536


def test_tq_write_accounting_reaches_the_queue(monkeypatch):
    """The write-side accounting tags must survive the producer->queue round trip.

    The fake queue copies tags at the send boundary (like the real controller).
    The payload upload and the terminal status publication each take time, and
    ``_publish_completion_tags`` runs after the status put -- so the completion
    stamps provably postdate the boundary the replay buffer waits on, and the
    payload put carries no accounting tags of its own beyond the byte count.
    """
    import transfer_queue as tq_module

    store: dict[str, dict] = {}
    calls: list[dict] = []
    payload_completed_at = 0.0
    status_completed_at = 0.0

    async def fake_async_kv_batch_put(*, keys, partition_id, fields=None, tags=None, **kwargs):
        nonlocal payload_completed_at
        if fields is not None:
            await asyncio.sleep(0.05)  # the payload upload takes time
            payload_completed_at = time.time()
        calls.append({"keys": list(keys), "fields": fields, "tags": [dict(t) for t in (tags or [])]})
        # Controller-side semantics: a put merges the new tag into the stored
        # row tag (TransferQueue BatchMeta.update_custom_meta).
        for key, tag in zip(keys, tags or [{}] * len(keys)):
            store.setdefault(key, {}).update(tag)

    async def fake_async_kv_put(*, key, partition_id, tag, **kwargs):
        nonlocal status_completed_at
        await asyncio.sleep(0.05)  # the terminal status publication takes time too
        status_completed_at = time.time()
        store.setdefault(key, {}).update(tag)

    monkeypatch.setattr(tq_module, "async_kv_batch_put", fake_async_kv_batch_put)
    monkeypatch.setattr(tq_module, "async_kv_put", fake_async_kv_put)
    from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ

    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class

    outputs = [
        (
            0,
            SimpleNamespace(
                prompt_ids=torch.zeros(1, 3, dtype=torch.long),
                response_diffusion_output=torch.zeros(1, 2, 2),
                num_turns=1,
                uid="uida",
                response_logprobs=None,
                reward_score=None,
                extra_fields={},
            ),
        ),
        (
            1,
            SimpleNamespace(
                prompt_ids=torch.zeros(1, 3, dtype=torch.long),
                response_diffusion_output=torch.zeros(1, 2, 2),
                num_turns=1,
                uid="uida",
                response_logprobs=None,
                reward_score=None,
                extra_fields={},
            ),
        ),
    ]
    worker = object.__new__(worker_cls)
    put_groups = asyncio.run(
        worker._write_trajectories_to_tq(
            outputs,
            uid="uida",
            trajectory={"step": 1},
            validate=False,
            global_steps=1,
        )
    )

    # The write itself is payload-only; the returned put groups carry the
    # rows and the measured upload time for the later stamping.
    assert len(calls) == 1
    assert calls[0]["fields"] is not None
    assert [keys for keys, _seconds in put_groups] == [calls[0]["keys"]]
    assert all(seconds > 0.0 for _keys, seconds in put_groups)
    for tag in calls[0]["tags"]:
        assert tag[TQ_PAYLOAD_BYTES_TAG] > 0

    # _run_prompt's post-status step: publish the terminal status, then stamp.
    asyncio.run(fake_async_kv_put(key="uida", partition_id="train", tag={"status": "finished"}))
    asyncio.run(worker._publish_completion_tags("uida", "train", put_groups))

    completion_put = calls[1]
    assert completion_put["fields"] is None
    assert completion_put["keys"] == calls[0]["keys"]
    # The completion update carries the put duration and the terminal-
    # publication boundary -- stamped after the status put returned, so the
    # trainer's detection never bills the status/tag round trips.
    for tag in completion_put["tags"]:
        assert tag[TQ_PUT_SECONDS_TAG] > 0.0
        assert tag[TQ_PUT_AT_TAG] >= status_completed_at > payload_completed_at
    for key in ("uida_0_0", "uida_1_0"):
        assert store[key][TQ_PAYLOAD_BYTES_TAG] > 0
        assert store[key][TQ_PUT_SECONDS_TAG] > 0.0
        assert store[key][TQ_PUT_AT_TAG] >= status_completed_at


# -- weight-sync phase timer -------------------------------------------------


class _StubManagerBase:
    """Stand-in for ``CheckpointEngineManager`` with the pinned dependency's
    call shapes: ``build_process_group`` is a plain synchronous method invoked
    without ``await`` from the async ``update_weights``; the other phases are
    async and awaited.
    """

    def __init__(self):
        self.replicas = [SimpleNamespace(), SimpleNamespace()]
        self.calls = []

    async def abort_replicas(self):
        self.calls.append("abort")
        await asyncio.sleep(0.01)

    async def release_kv_cache_replicas(self):
        self.calls.append("kv_release")
        await asyncio.sleep(0.01)

    def build_process_group(self, rollout):
        self.calls.append("topology")
        time.sleep(0.01)

    async def resume_kv_cache_replicas(self):
        self.calls.append("kv_resume")
        await asyncio.sleep(0.01)

    async def resume_generation_replicas(self):
        self.calls.append("generation_resume")
        await asyncio.sleep(0.01)

    async def update_weights(self, global_steps=None):
        # The pinned pipeline's call order: every phase method is invoked
        # through self, so the mixin's overrides intercept; topology is called
        # synchronously, and the inline transfer window is the sleep below.
        await self.abort_replicas()
        await self.release_kv_cache_replicas()
        self.build_process_group(rollout=None)
        await asyncio.sleep(0.02)  # worker-group build + transfer + finalize
        await self.resume_kv_cache_replicas()
        await self.resume_generation_replicas()
        return {"engine/changed_ratio": 0.25}


class _StubTimedManager(WeightSyncPhaseTimerMixin, _StubManagerBase):
    async def update_weights(self, global_steps=None):
        return await self._profiled_update_weights(_StubManagerBase.update_weights, global_steps)


def test_weight_sync_phase_timer_decomposes_pipeline():
    manager = _StubTimedManager()
    metrics = asyncio.run(manager.update_weights(global_steps=3))

    # Engine metrics keep their own keys; phase timings are added alongside.
    # (Windows timer resolution makes sleep(0.01) land up to ~0.03s, so the
    # tolerance only separates phases from the 0.02s residual window.)
    assert metrics["engine/changed_ratio"] == 0.25
    for phase in ("abort", "kv_release", "topology", "kv_resume", "generation_resume"):
        assert metrics[f"weight_sync/phase/{phase}_s"] == pytest.approx(0.01, abs=0.02)
    # Residual window holds the un-wrapped inline phases (modeled as 0.02s).
    assert metrics["weight_sync/phase/transfer_and_finalize_s"] == pytest.approx(0.02, abs=0.02)
    assert metrics["weight_sync/total_s"] == pytest.approx(0.07, abs=0.1)
    assert metrics["weight_sync/replicas"] == 2.0
    # The phases ran through the mixin's overrides in pipeline order. If the
    # topology override were async again, the un-awaited sync call site above
    # would drop its coroutine and "topology" would be missing here.
    assert manager.calls == ["abort", "kv_release", "topology", "kv_resume", "generation_resume"]


def test_timed_manager_sync_call_style_still_works():
    """The async phase overrides keep the dual sync/async contract of @auto_await."""
    manager = _StubTimedManager()
    # Calling the coroutine function via asyncio.run from a sync context.
    manager.abort_replicas()  # not awaited: auto_await runs it in a thread pool
    assert manager.calls == ["abort"]


def test_timed_checkpoint_engine_manager_is_constructible_subclass():
    # The Ray-facing classes must still subclass verl's manager for isinstance
    # checks and the trainer's duck typing.
    from verl.checkpoint_engine import CheckpointEngineManager

    assert issubclass(TimedCheckpointEngineManager, CheckpointEngineManager)
