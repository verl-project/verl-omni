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
(including the slow-generation vs coarse-polling counterexamples), the
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


def test_row_arriving_before_first_sync_counts_detection(monkeypatch):
    """A row published inside the window counts detection on the first sync."""
    fake_tq = _FakeTransferQueue({})
    _patch_transfer_queue(monkeypatch, fake_tq)
    real_kv_list = fake_tq.kv_list

    def kv_list_publishing_row():
        # The first sync stalls mid-flight and the row lands inside that gap,
        # so its unseen time (put_at -> discovery) is real and in-window.
        time.sleep(0.02)
        fake_tq.add_finished_group("uida", put_at=time.time() - 0.01)
        return real_kv_list()

    monkeypatch.setattr(replay_buffer_module.tq, "kv_list", kv_list_publishing_row)

    buffer = _make_profiled_buffer()
    buffer.sample(global_steps=1, partition_id="train", batch_size=1)

    timing = buffer.last_sample_timing
    assert timing["sample_wait/polls"] == 0.0
    # Published between the window start and the first sync: the arrival gap
    # is recorded instead of silently marking the row as seen.
    assert timing["sample_wait/detection"] > 0.0


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

    The fake queue copies tags at the send boundary (like the real controller)
    and the payload put itself takes time, so the completion stamps provably
    exclude the transfer duration; mutating the local dicts after the put
    would leave these assertions empty.
    """
    import transfer_queue as tq_module

    store: dict[str, dict] = {}
    calls: list[dict] = []
    payload_completed_at = 0.0

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

    monkeypatch.setattr(tq_module, "async_kv_batch_put", fake_async_kv_batch_put)
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
    asyncio.run(
        worker._write_trajectories_to_tq(
            outputs,
            uid="uida",
            trajectory={"step": 1},
            validate=False,
            global_steps=1,
        )
    )

    # One payload put plus one tag-only completion update per field-signature group.
    assert len(calls) == 2
    payload_put, completion_put = calls
    assert payload_put["fields"] is not None
    assert completion_put["fields"] is None
    assert completion_put["keys"] == payload_put["keys"]
    # Bytes rode the payload put itself.
    for tag in payload_put["tags"]:
        assert tag[TQ_PAYLOAD_BYTES_TAG] > 0
    # The completion update carries the put duration and the publication
    # boundary -- stamped after the transfer finished, not when it started,
    # so the trainer's detection never bills the upload to polling.
    for tag in completion_put["tags"]:
        assert tag[TQ_PUT_SECONDS_TAG] > 0.0
        assert tag[TQ_PUT_AT_TAG] >= payload_completed_at
    for key in ("uida_0_0", "uida_1_0"):
        assert store[key][TQ_PAYLOAD_BYTES_TAG] > 0
        assert store[key][TQ_PUT_SECONDS_TAG] > 0.0
        assert store[key][TQ_PUT_AT_TAG] >= payload_completed_at


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
    assert metrics["engine/changed_ratio"] == 0.25
    for phase in ("abort", "kv_release", "topology", "kv_resume", "generation_resume"):
        assert metrics[f"weight_sync/phase/{phase}_s"] == pytest.approx(0.01, abs=0.008)
    # Residual window holds the un-wrapped inline phases (modeled as 0.02s).
    assert metrics["weight_sync/phase/transfer_and_finalize_s"] == pytest.approx(0.02, abs=0.01)
    assert metrics["weight_sync/total_s"] == pytest.approx(0.07, abs=0.02)
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
