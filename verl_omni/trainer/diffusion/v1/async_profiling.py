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
"""Always-on accounting for the async diffusion V1 trainers (issue #712).

Three complementary pieces feed the per-step metrics dict:

- :class:`ProfiledReplayBufferAsync` decomposes ``sample()`` wall time into
  generation-bound wait, completion-detection latency, TransferQueue metadata
  sync, and eviction. The trainer reads ``last_sample_timing`` after each
  sample and merges it into ``timing_raw``, so every segment shows up as a
  ``timing_s/sample_wait/*`` metric with no new export channel.
- :func:`aggregate_tq_write_stats` sums the per-row ``tq_put_s`` /
  ``tq_payload_bytes`` tags the rollout-side TQ writer records, attributing
  rollout-side transfer cost and bytes to the batch that actually pays for
  them (the sampled one).
- :func:`tensordict_payload_bytes` measures the trainer-side read payload so
  ``tq/get/bytes`` pairs with ``timing_s/tq_get`` for a GB/s figure.

Everything here is metadata-only: timers that were already being taken (or
tags that were already being written) promoted to metrics, no behavior
change.
"""

import time
from collections import defaultdict

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

# Tag keys written by ``DiffusionAgentLoopWorkerTQ`` and summed by
# :func:`aggregate_tq_write_stats`. ``tq_put_at`` (wall-clock seconds) is the
# terminal-publication boundary the wait decomposition keys on: the rollout
# writer stamps it only after the prompt's ``status=finished`` put has
# returned, because the buffer cannot sample the group before that status.
# Keep in sync with the rollout writer's stamping.
TQ_PUT_SECONDS_TAG = "tq_put_s"
TQ_PAYLOAD_BYTES_TAG = "tq_payload_bytes"
TQ_PUT_AT_TAG = "tq_put_at"

# Per-local-update totals that must sum across the ``parameter_sync_step``
# updates of one outer step: ``MetricsAggregator``'s default rule is
# weighted_avg, which would undercount every one of them. The step-level
# rate is derived from the summed bytes and seconds by
# :func:`derive_tq_put_rate` -- averaging per-update rates misstates
# throughput when updates carry different durations.
TQ_COUNTER_METRICS = ("tq/put/rows", "tq/put/seconds", "tq/put/bytes", "tq/get/bytes")


class ProfiledReplayBufferAsync(ReplayBufferAsync):
    """``ReplayBufferAsync`` that accounts for where ``sample()`` time went.

    The upstream poll loop sleeps a fixed ``poll_interval`` between
    ``kv_list`` metadata syncs, evicts stale/filtered groups, and re-checks
    sampleability -- all inside one opaque wait. Because the sleep is how the
    loop waits for generation, sleep time alone cannot distinguish a
    generation-throughput bottleneck from polling-quantization waste: with a
    60 s generation and a 2 s interval, nearly the whole wait is sleep. This
    subclass therefore keys on arrival events (the ``tq_put_at`` row tags)
    instead of the sleep:

    - ``sample_wait/generation_bound`` -- wait while the needed trajectories
      were still generating (residual: total minus the segments below);
    - ``sample_wait/detection`` -- batch-readiness discovery latency: how long
      the group whose completion made the requested batch sampleable sat
      between its terminal publication and the metadata sync that discovered
      it (this is the part a smaller ``poll_interval`` recovers);
    - ``sample_wait/metadata_sync`` -- ``kv_list`` round trips;
    - ``sample_wait/eviction`` -- stale/DAPO/failure eviction passes.

    Detection is charged once per window, at the sampleability transition, and
    only for the crossing group: among the eligible terminal groups first
    tagged by the committing sync, the ``(target - already_sampleable)``-th
    earliest arrival is the one that completed the batch -- earlier-completing
    groups were waited on for their generation, so their undiscovered time
    stays in ``generation_bound``, and a window that opens with the batch
    already sampleable charges nothing. The arrival timestamp is sampled
    before the discovering sync's ``kv_list`` call, so detection can never
    overlap the metadata segment: an arrival inside a metadata RPC is that
    RPC's service time, and the segment sums stay within the wall time.
    Groups published by older writers carry no ``tq_put_at`` and fall back
    into ``generation_bound``; groups tagged before the window started are
    likewise excluded from detection, since they arrived while the trainer
    was not polling for them. A completion tag that appears on an already
    visible group is what stages the arrival -- the writer publishes payload
    rows before the telemetry tags, so keying on tag transitions rather than
    row novelty is what keeps such a group's wait from collapsing to zero.

    ``last_sample_timing`` (seconds per segment, plus poll count and the
    finished-group depth seen while waiting) is refreshed on every
    ``wait_for_sampleable`` / ``sample`` call. Accumulation only happens
    inside those calls, so incidental metadata syncs from
    ``get_sampleable_count`` (hybrid switch decisions) are not mixed in.

    The queue-depth snapshots answer "was the trainer waiting because the
    buffer was drained or because everything was still running" -- pending vs
    running vs finished counts per poll -- which is the difference between a
    generation-throughput problem and a scheduling one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._profiling_active = False
        self._window_depth = 0
        self._segment_timings: dict[str, float] = defaultdict(float)
        self._poll_depths: list[tuple[int, int, int]] = []
        # Completion-tag bookkeeping per partition: uid -> staged arrival. A
        # uid is staged once its rows first carry ``tq_put_at`` (which may be
        # several syncs after the rows themselves became visible) and never
        # re-staged while its rows persist.
        self._counted_arrivals: dict[str, dict[str, float]] = {}
        self._window_started_at: float | None = None
        self._last_sync_wall_time: float | None = None
        # Batch-readiness attribution: each profiled sync stages the eligible
        # arrivals it discovered as ``{uid: arrival}`` plus the pre-sync clock
        # reading and the clamp cap; ``_has_enough_samples`` commits the
        # threshold-crossing arrival only when that sync's check turns the
        # window from waiting to ready.
        self._sync_detection_candidates: dict[str, tuple[dict[str, float], float, float]] = {}
        self._sampleable_latched = False
        self._sampleable_at_window_open = False
        self.last_sample_timing: dict[str, float] = {}

    # -- timed segments ----------------------------------------------------

    def _sync_metadata_from_transfer_queue(self):
        start = time.perf_counter() if self._profiling_active else None
        # Detection's reference clock is read before the RPC below: an arrival
        # that lands inside the metadata call is that call's service time, so
        # detection and metadata_sync stay disjoint and their sums stay within
        # the wall time.
        now = time.time()
        try:
            return super()._sync_metadata_from_transfer_queue()
        finally:
            if start is not None:
                self._segment_timings["metadata_sync"] += time.perf_counter() - start
            if self._profiling_active:
                # Arrivals are counted from the window open (older rows were
                # published while the trainer was not polling for them), but
                # detection is clamped to the discovering gap: an undiscovered
                # stretch that spans earlier syncs overlaps those RPCs, so it
                # stays in the generation_bound residual instead.
                window_open = self._window_started_at if self._window_started_at is not None else now
                prev_sync_end = self._last_sync_wall_time if self._last_sync_wall_time is not None else window_open
                cap = max(0.0, now - prev_sync_end)
                for partition_id, partition in self.partitions.items():
                    counted = self._counted_arrivals.setdefault(partition_id, {})
                    finished_uids = self.finished_keys[partition_id]
                    latest_arrival: dict[str, float] = {}
                    uids_with_rows: set[str] = set()
                    for key, tag in partition.items():
                        uid = key.split("_")[0]
                        uids_with_rows.add(uid)
                        if uid in counted or uid not in finished_uids or not isinstance(tag, dict):
                            continue
                        put_at = tag.get(TQ_PUT_AT_TAG)
                        if isinstance(put_at, int | float) and float(put_at) >= window_open:
                            arrival = float(put_at)
                            if uid not in latest_arrival or arrival > latest_arrival[uid]:
                                latest_arrival[uid] = arrival
                    self._sync_detection_candidates[partition_id] = (latest_arrival, now, cap)
                    counted.update(latest_arrival)
                    # Groups whose rows vanished (evicted or cleared) are
                    # forgotten, so a refilled re-publication counts afresh.
                    for uid in list(counted):
                        if uid not in uids_with_rows:
                            del counted[uid]
                self._last_sync_wall_time = now

    def _evict_terminal_groups(self, global_steps, partition_id, eviction_reasons):
        if not self._profiling_active:
            return super()._evict_terminal_groups(global_steps, partition_id, eviction_reasons)
        start = time.perf_counter()
        try:
            return super()._evict_terminal_groups(global_steps, partition_id, eviction_reasons)
        finally:
            self._segment_timings["eviction"] += time.perf_counter() - start

    def _wait_for_next_poll(self, partition_id, last_debug_time):
        if not self._profiling_active:
            return super()._wait_for_next_poll(partition_id, last_debug_time)
        self._poll_depths.append(
            (
                len(self.pending_keys[partition_id]),
                len(self.running_keys[partition_id]),
                len(self.finished_keys[partition_id]),
            )
        )
        start = time.perf_counter()
        try:
            return super()._wait_for_next_poll(partition_id, last_debug_time)
        finally:
            self._segment_timings["poll_sleep"] += time.perf_counter() - start

    # -- profiling window --------------------------------------------------

    def _begin_profile_window(self, global_steps: int, partition_id: str, target_count: int) -> None:
        # sample() wraps wait_for_sampleable(); the nested call must not reset
        # the outer window's accumulators.
        self._window_depth += 1
        if self._window_depth > 1:
            return
        self._segment_timings = defaultdict(float)
        self._poll_depths = []
        self._counted_arrivals = {}
        self._window_started_at = time.time()
        self._last_sync_wall_time = self._window_started_at
        self._sync_detection_candidates = {}
        self._sampleable_latched = False
        # Sampleability of the last-synced state: a window that opens with the
        # batch already ready never waits, so it must not charge detection for
        # rows it happens to discover. Cache-only reads -- the wait loop's own
        # first iteration evaluates the same state.
        eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
        self._sampleable_at_window_open = (
            len(self._sampleable_terminal_keys(partition_id, eviction_reasons)) >= target_count
        )
        self._profiling_active = True

    def _end_profile_window(self, total_seconds: float) -> None:
        self._window_depth = max(0, self._window_depth - 1)
        if self._window_depth > 0:
            return
        self._profiling_active = False
        # Report every segment every time (0.0 when absent) so dashboards see
        # a constant metric schema across steps regardless of what ran.
        # poll_sleep is an internal measure (it cannot distinguish generation
        # wait from quantization); the public decomposition is the residual
        # generation_bound plus the event-based detection above.
        measured_names = ("detection", "metadata_sync", "eviction")
        measured = sum(self._segment_timings[name] for name in measured_names)
        timings = {f"sample_wait/{name}": self._segment_timings[name] for name in measured_names}
        timings["sample_wait/generation_bound"] = max(0.0, total_seconds - measured)
        timings["sample_wait/total"] = total_seconds
        timings["sample_wait/polls"] = float(len(self._poll_depths))
        if self._poll_depths:
            finished = [depth[2] for depth in self._poll_depths]
            timings["sample_wait/buffer_finished_mean"] = sum(finished) / len(finished)
            timings["sample_wait/buffer_finished_min"] = float(min(finished))
            timings["sample_wait/buffer_finished_max"] = float(max(finished))
            pending = [depth[0] for depth in self._poll_depths]
            running = [depth[1] for depth in self._poll_depths]
            timings["sample_wait/buffer_pending_mean"] = sum(pending) / len(pending)
            timings["sample_wait/buffer_running_mean"] = sum(running) / len(running)
        self.last_sample_timing = timings

    def _has_enough_samples(
        self, global_steps: int, partition_id: str, batch_size: int, sampleable_keys: set[str]
    ) -> bool:
        result = super()._has_enough_samples(global_steps, partition_id, batch_size, sampleable_keys)
        if self._profiling_active and result and not self._sampleable_latched:
            self._sampleable_latched = True
            if not self._sampleable_at_window_open:
                # The sync that turned the window ready stages the arrivals it
                # discovered; restricted to the eligible set, the batch needed
                # ``len(sampleable) - discovered`` groups that earlier syncs
                # already knew about, so the ``needed``-th earliest staged
                # arrival is the group whose completion crossed the
                # sampleability threshold. Surplus, nonterminal, and evictable
                # groups never enter this ordering, and an untagged crossing
                # group simply leaves detection at zero (its wait stays in the
                # generation_bound residual).
                staged, now, cap = self._sync_detection_candidates.get(partition_id, ({}, 0.0, 0.0))
                eligible_arrivals = sorted(t for uid, t in staged.items() if uid in sampleable_keys)
                needed = batch_size - (len(sampleable_keys) - len(eligible_arrivals))
                if 0 < needed <= len(eligible_arrivals):
                    detection = min(max(now - eligible_arrivals[needed - 1], 0.0), cap)
                    self._segment_timings["detection"] += detection
        return result

    def wait_for_sampleable(self, global_steps: int, partition_id: str, target_count: int):
        self._begin_profile_window(global_steps, partition_id, target_count)
        start = time.perf_counter()
        try:
            return super().wait_for_sampleable(global_steps, partition_id, target_count)
        finally:
            self._end_profile_window(time.perf_counter() - start)

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        self._begin_profile_window(global_steps, partition_id, batch_size)
        start = time.perf_counter()
        try:
            return super().sample(global_steps, partition_id, batch_size)
        finally:
            self._end_profile_window(time.perf_counter() - start)


def aggregate_tq_write_stats(batch_meta) -> dict[str, float]:
    """Sum rollout-side TQ write cost recorded in row tags for one batch.

    ``DiffusionAgentLoopWorkerTQ`` stamps every written row with the wall time
    of its ``kv_batch_put`` and the payload bytes it carried. Rows written by
    older writers (or tag-only prompt rows filtered out before this point)
    simply lack the keys and are skipped, so the aggregation degrades to zero
    rather than error. The totals are per local update: the trainer's
    aggregator sums them across the step (:data:`TQ_COUNTER_METRICS`) and
    :func:`derive_tq_put_rate` derives the step-level rate from the sums --
    per-update rates weighted-averaged by the aggregator would misstate
    throughput, so this function emits none.
    """
    put_seconds = 0.0
    payload_bytes = 0
    tagged_rows = 0
    for tag in getattr(batch_meta, "tags", None) or []:
        if not isinstance(tag, dict):
            continue
        seconds = tag.get(TQ_PUT_SECONDS_TAG)
        if seconds is not None:
            put_seconds += float(seconds)
            tagged_rows += 1
        payload_bytes += int(tag.get(TQ_PAYLOAD_BYTES_TAG, 0) or 0)
    return {
        "tq/put/rows": float(tagged_rows),
        "tq/put/seconds": put_seconds,
        "tq/put/bytes": float(payload_bytes),
    }


def derive_tq_put_rate(step_metrics: dict[str, float]) -> None:
    """Fill ``tq/put/bytes_per_second`` from the step-summed bytes and seconds.

    Call after the aggregator has summed the per-update counters; the rate is
    a property of the whole outer step, not of any single local update.
    """
    put_bytes = step_metrics.get("tq/put/bytes", 0.0)
    put_seconds = step_metrics.get("tq/put/seconds", 0.0)
    if put_bytes > 0 and put_seconds > 0:
        step_metrics["tq/put/bytes_per_second"] = put_bytes / put_seconds


def tensordict_payload_bytes(data) -> int:
    """Payload bytes of a tensor batch fetched from TransferQueue.

    Counts the tensor columns only: non-tensor columns (uid, data_source, ...)
    ride the tag/metadata channel, not the payload transfer the RFC's byte
    counters are about.
    """
    total = 0
    for value in data.values():
        if hasattr(value, "element_size"):
            total += value.numel() * value.element_size()
    return total
