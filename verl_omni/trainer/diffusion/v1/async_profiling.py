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

# Tag keys written by ``DiffusionAgentLoopWorkerTQ._write_trajectories_to_tq``
# and summed by :func:`aggregate_tq_write_stats`. ``tq_put_at`` (wall-clock
# seconds) is the completed-publication boundary the wait decomposition keys
# on; keep in sync with the rollout writer's stamping.
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
      the group that completed the requested batch sat unseen between its
      publication and the sync at which the batch became sampleable (this is
      the part a smaller ``poll_interval`` recovers);
    - ``sample_wait/metadata_sync`` -- ``kv_list`` round trips;
    - ``sample_wait/eviction`` -- stale/DAPO/failure eviction passes.

    Detection is charged once per window, at the sampleability transition:
    rows discovered before the batch is sampleable were waited on for their
    generation, so their undiscovered time stays in ``generation_bound``, and
    a window that opens with the batch already sampleable charges nothing.
    Rows written by older writers carry no ``tq_put_at`` and fall back into
    ``generation_bound``; rows tagged before the window started are likewise
    excluded from detection, since they arrived while the trainer was not
    polling for them.

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
        self._seen_trajectory_keys: dict[str, set[str]] = {}
        self._last_sync_wall_time: float | None = None
        # Batch-readiness attribution: each profiled sync stages the detection
        # it discovered per partition; ``_has_enough_samples`` commits it only
        # when that sync's check turns the window from waiting to ready, so
        # discoveries the batch does not need yet stay in generation_bound.
        self._sync_detection_candidates: dict[str, float] = {}
        self._sampleable_latched = False
        self._sampleable_at_window_open = False
        self.last_sample_timing: dict[str, float] = {}

    # -- timed segments ----------------------------------------------------

    def _sync_metadata_from_transfer_queue(self):
        start = time.perf_counter() if self._profiling_active else None
        try:
            return super()._sync_metadata_from_transfer_queue()
        finally:
            if start is not None:
                self._segment_timings["metadata_sync"] += time.perf_counter() - start
            # Arrival bookkeeping runs on every sync (profiled or not) so rows
            # that arrived via incidental syncs are never re-detected as new
            # arrivals inside a later profiling window.
            now = time.time()
            for partition_id, partition in self.partitions.items():
                current = set(partition)
                seen = self._seen_trajectory_keys.setdefault(partition_id, set())
                if self._profiling_active:
                    window_start = self._last_sync_wall_time or now
                    arrivals = [
                        float(tag[TQ_PUT_AT_TAG])
                        for key in current - seen
                        if isinstance(tag := partition.get(key), dict)
                        and isinstance(tag.get(TQ_PUT_AT_TAG), int | float)
                        and float(tag[TQ_PUT_AT_TAG]) >= window_start
                    ]
                    if arrivals:
                        # Clamping bounds clock skew between the rollout writer
                        # and this process to the inter-sync window.
                        detection = min(max(now - max(arrivals), 0.0), now - window_start)
                        self._sync_detection_candidates[partition_id] = detection
                    else:
                        self._sync_detection_candidates[partition_id] = 0.0
                self._seen_trajectory_keys[partition_id] = current
            if self._profiling_active:
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
        self._last_sync_wall_time = time.time()
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
                # The sync that discovered the completing group stages its
                # detection in the candidate; waits for earlier-completing
                # groups were generation-bound and stay in the residual.
                self._segment_timings["detection"] += self._sync_detection_candidates.get(partition_id, 0.0)
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
