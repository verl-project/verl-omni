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

:class:`ProfiledReplayBufferAsync` decomposes ``sample()`` wall time into
generation-bound wait, detection latency, TransferQueue metadata sync, and
eviction (read via ``last_sample_timing``); :func:`aggregate_tq_write_stats`
sums the per-row ``tq_put_s`` / ``tq_payload_bytes`` tags the rollout-side
TQ writer records; :func:`tensordict_payload_bytes` measures the
trainer-side read payload. Full metric catalog: ``docs/perf/profiler.md``.
"""

import time
from collections import defaultdict

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

# ``tq_put_at`` marks terminal publication (the finished-status put's
# return); keep in sync with the rollout writer's stamping.
TQ_PUT_SECONDS_TAG = "tq_put_s"
TQ_PAYLOAD_BYTES_TAG = "tq_payload_bytes"
TQ_PUT_AT_TAG = "tq_put_at"

# Per-update totals that must sum across the step's local updates: the
# aggregator's weighted_avg would undercount; derive the rate from sums.
TQ_COUNTER_METRICS = ("tq/put/rows", "tq/put/seconds", "tq/put/bytes")


class ProfiledReplayBufferAsync(ReplayBufferAsync):
    """``ReplayBufferAsync`` that decomposes ``sample()`` wait time.

    The upstream poll loop's fixed sleep cannot distinguish slow generation
    from coarse polling, so this subclass keys on ``tq_put_at`` arrival
    events instead and exports ``last_sample_timing`` with these segments
    (all under ``sample_wait/``):

    - ``generation_bound`` -- the needed trajectories were still generating
      (residual: total minus the segments below);
    - ``detection`` -- the batch-completing group waited between its
      terminal publication and the sync that discovered it (what a smaller
      ``poll_interval`` recovers);
    - ``metadata_sync`` -- ``kv_list`` round trips;
    - ``eviction`` -- stale/DAPO/failure eviction passes;
    - ``polls`` and ``buffer_*`` -- poll count and pending/running/finished
      group depths.

    Detection is charged once per window, at the sampleability transition,
    to the group whose completion crossed the batch threshold, clamped to
    the gap since the previous sync returned so it never overlaps
    ``metadata_sync``; untagged groups (older writers, pre-window tags)
    stay in ``generation_bound``. Full attribution rules live in
    ``docs/perf/profiler.md``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._profiling_active = False
        self._window_depth = 0
        self._segment_timings: dict[str, float] = defaultdict(float)
        self._poll_depths: list[tuple[int, int, int]] = []
        # uid -> staged arrival; staged once rows first carry ``tq_put_at``,
        # never re-staged while the rows persist.
        self._counted_arrivals: dict[str, dict[str, float]] = {}
        self._window_started_at: float | None = None
        self._last_sync_wall_time: float | None = None
        # Per-sync staged arrivals with the pre-sync clock reading and the
        # clamp cap; ``_has_enough_samples`` commits the crossing arrival.
        self._sync_detection_candidates: dict[str, tuple[dict[str, float], float, float]] = {}
        self._sampleable_latched = False
        self._sampleable_at_window_open = False
        self.last_sample_timing: dict[str, float] = {}

    # -- timed segments ----------------------------------------------------

    def _sync_metadata_from_transfer_queue(self):
        start = time.perf_counter() if self._profiling_active else None
        # Sampled before the RPC below: an arrival inside it is that call's
        # service time, keeping detection disjoint from metadata_sync.
        now = time.time()
        try:
            return super()._sync_metadata_from_transfer_queue()
        finally:
            sync_end = time.time()
            if start is not None:
                self._segment_timings["metadata_sync"] += time.perf_counter() - start
            if self._profiling_active:
                # Clamp detection to the gap since the previous sync's RPC
                # returned: a wider clamp would double-book that RPC's tail.
                window_open = self._window_started_at if self._window_started_at is not None else sync_end
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
                self._last_sync_wall_time = sync_end

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
        return super()._wait_for_next_poll(partition_id, last_debug_time)

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
        # A window that opens already-sampleable never waited and must not
        # charge detection; cache-only reads match the wait loop's first pass.
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
        # Emit every segment every time (0.0 when absent) so the metric
        # schema is constant across steps; generation_bound is the residual.
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
                # The ``needed``-th earliest staged eligible arrival is the
                # crossing group; an untagged one leaves detection at zero.
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

    Rows from older writers lack the tag keys and are skipped (totals degrade
    to zero, never error). Per-update totals by design: the aggregator sums
    them (:data:`TQ_COUNTER_METRICS`) and :func:`derive_tq_put_rate` derives
    the step rate from the sums -- per-update rates would misstate throughput.
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

    Tensor columns only: non-tensor columns are bookkeeping scalars beside
    the payload that dominates the read time.
    """
    total = 0
    for value in data.values():
        if hasattr(value, "element_size"):
            total += value.numel() * value.element_size()
    return total
