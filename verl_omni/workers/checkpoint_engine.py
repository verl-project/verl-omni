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
import time
from collections import defaultdict

import ray
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineRegistry
from verl.utils.ray_utils import auto_await

# verl's CheckpointEngineWorker gates the "delta_sharded" backend to sglang
# rollouts (its own consumer rides the sglang custom weight loader / the vLLM
# weight-transfer engine of newer pins), and at this pin verl's vLLM
# ServerAdapter only streams the named_tensors bucketed wire (the delta_flush
# dispatch arrives with verl#7227). verl-omni therefore registers a thin
# subclass of verl's DeltaShardedCheckpointEngine under its own backend name:
# the subclass re-declares the wire as named_tensors and flattens the flush
# stream into sentinel-named pairs. verl's worker and the inherited vLLM
# ServerAdapter weight sender drive the sync; the omni adapter adds checked
# replies for IPC updates. Importing this module performs the
# registration (see verl_omni/__init__.py).
try:
    from verl.checkpoint_engine import DeltaShardedCheckpointEngine

    if DeltaShardedCheckpointEngine is not None:

        class OmniDeltaShardedCheckpointEngine(DeltaShardedCheckpointEngine):
            """verl's delta engine presenting its flushes on the stock named_tensors wire.

            ``receive_weights`` flattens the ``(named, is_last)`` flush stream into
            ``(name#<flush>, tensor)`` pairs: the stock bucketed sender keys each
            bucket's metadata dict by tensor name, so the suffix keeps same-named
            sentinels (``__delta_spec__`` / ``__positions__`` / ``__values__``) of
            separate flushes sharing a bucket from overwriting each other's entry;
            the omni rollout worker parses the suffix back off. Everything else --
            the seed/steady state machine, snapshot priming, sparse gather, wire
            encoding -- is verl's, inherited unchanged.
            """

            wire_format = "named_tensors"

            def receive_weights(self, global_steps: int | None = None):
                """Yield the flush stream flattened into ``(name#flush, tensor)`` pairs.

                Every rank must drain this generator to the end: the receive loop's
                collective broadcasts deadlock otherwise. Dropping the per-flush
                ``is_last`` is safe -- flush boundaries are keyed off the sentinel
                ordering, and the bucketed channel marks its final bucket after this
                generator is drained, which is what completes the receiver.
                """
                yield from _flatten_flush_stream(super().receive_weights(global_steps))

        CheckpointEngineRegistry.register("omni_delta_sharded")(OmniDeltaShardedCheckpointEngine)
except ImportError:  # verl records the failure; Registry.get reports it on use
    pass


def _flatten_flush_stream(flushes):
    """Yield ``(name#flush, tensor)`` from ``(named_tensors, is_last)`` flushes.

    ``is_last`` is dropped on purpose: the bucketed sender marks the final
    bucket after this generator is drained, and that is what completes the
    receiver. The suffix keeps same-named sentinels of separate flushes that
    share a bucket from overwriting each other.
    """
    for flush_idx, (named, _is_last) in enumerate(flushes):
        for name, tensor in named:
            yield f"{name}#{flush_idx}", tensor


class OmniCheckpointEngineManager(CheckpointEngineManager):
    """``CheckpointEngineManager`` subclass that forwards the actor's LoRA
    ``peft_config`` to standalone rollout replicas for separate-async NCCL
    weight sync.
    """

    @auto_await
    async def update_weights(self, global_steps: int = None):
        """Fetch the actor's LoRA ``peft_config`` and stash it on the rollout
        workers before delegating to the parent ``update_weights``. The
        parent's engine sync metrics are propagated so callers (and the
        ``WeightSyncPhaseTimerMixin`` subclass) can merge them into step
        metrics.
        """
        if self.backend != "naive":
            peft_config = self._fetch_actor_lora_peft_config()
            self._lora_peft_config = peft_config
            await self._push_lora_peft_config_to_replicas(peft_config)
        return await super().update_weights(global_steps=global_steps)

    async def _push_lora_peft_config_to_replicas(self, peft_config: dict | None) -> None:
        """Fetch ``peft_config`` from the actor (collective-free) and stash it
        on every standalone rollout replica's worker extension.

        """
        futures = [
            replica.server_handle.collective_rpc.remote(
                "set_pending_lora_peft_config",
                kwargs={"peft_config": peft_config},
            )
            for replica in self.replicas
            if replica.server_handle is not None
        ]
        if futures:
            ray.get(futures)

    def _fetch_actor_lora_peft_config(self):
        """Return the actor's LoRA ``peft_config`` dict, or ``None``."""
        if not hasattr(self.actor_wg, "get_lora_peft_config"):
            return None
        results = self.actor_wg.get_lora_peft_config()
        for result in results or []:
            if result is not None:
                return result
        return None


# verl's ``CheckpointEngineManager.update_weights`` runs an eight-phase
# pipeline (abort -> temp worker group -> KV release -> process-group build ->
# transfer -> finalize -> KV resume -> generation resume), and the trainer
# only ever sees the total as ``timing_s/update_weights``. The mixin below
# times each phase by wrapping the very manager methods the pipeline calls,
# without duplicating any of verl's internals; the inline phases (worker-group
# construction, the transfer ``ray.get`` and ``finalize``) fall out as the
# residual and are reported together as ``transfer_and_finalize`` because the
# transfer dominates that window. The async phase overrides keep verl's
# ``@auto_await`` dual sync/async call style and await the parent's raw
# coroutine via ``__wrapped__`` so the decorator never nests; the sync caller
# gets the thread-pool path -- exactly the parent's contract.
# ``build_process_group`` is a plain synchronous method in the pinned verl and
# is called un-awaited from the async ``update_weights``, so its override must
# stay synchronous: an ``auto_await`` override would hand that caller a
# coroutine it never awaits, silently skipping topology/process-group setup.
class WeightSyncPhaseTimerMixin:
    """Report per-phase weight-sync timings through ``update_weights``'s metrics.

    The metrics dict returned by ``update_weights`` (already merged into the
    trainer's step metrics via ``_pending_sync_metrics``) gains
    ``weight_sync/phase/<name>_s`` for each timed phase plus
    ``weight_sync/total_s`` and ``weight_sync/replicas``; engine-reported
    sync metrics keep their own keys unchanged.
    """

    _sync_phase_timings: dict[str, float]

    def __init__(self, *args, **kwargs):
        # Phase methods can be called outside update_weights (switch_to_trainer
        # aborts/sleeps replicas directly), so the accumulator must exist from
        # construction, not just inside a profiled sync.
        super().__init__(*args, **kwargs)
        self._sync_phase_timings = defaultdict(float)

    async def _timed_weight_sync_phase(self, parent_method, phase: str, *args, **kwargs):
        start = time.perf_counter()
        try:
            return await parent_method(*args, **kwargs)
        finally:
            self._sync_phase_timings[phase] += time.perf_counter() - start

    async def _profiled_update_weights(self, raw_parent_update, global_steps):
        self._sync_phase_timings = defaultdict(float)
        start = time.perf_counter()
        sync_metrics = await raw_parent_update(self, global_steps=global_steps)
        total = time.perf_counter() - start
        metrics = dict(sync_metrics or {})
        metrics.update({f"weight_sync/phase/{p}_s": s for p, s in self._sync_phase_timings.items()})
        measured = sum(self._sync_phase_timings.values())
        metrics["weight_sync/phase/transfer_and_finalize_s"] = max(0.0, total - measured)
        metrics["weight_sync/total_s"] = total
        metrics["weight_sync/replicas"] = float(len(getattr(self, "replicas", []) or []))
        return metrics

    def _wrap_phase_method(self, name: str, phase: str, *args, **kwargs):
        parent = getattr(super(), name, None)
        raw = getattr(parent, "__wrapped__", None)
        if raw is None:
            # Parent phase is a plain async def (test doubles): time it by
            # wrapping the coroutine we get from calling it.
            async def _timed_plain():
                start = time.perf_counter()
                try:
                    return await parent(*args, **kwargs)
                finally:
                    self._sync_phase_timings[phase] += time.perf_counter() - start

            return _timed_plain()
        return self._timed_weight_sync_phase(raw, phase, self, *args, **kwargs)

    @auto_await
    async def abort_replicas(self):
        return await self._wrap_phase_method("abort_replicas", "abort")

    @auto_await
    async def release_kv_cache_replicas(self):
        return await self._wrap_phase_method("release_kv_cache_replicas", "kv_release")

    def build_process_group(self, rollout):
        start = time.perf_counter()
        try:
            return super().build_process_group(rollout)
        finally:
            self._sync_phase_timings["topology"] += time.perf_counter() - start

    @auto_await
    async def resume_kv_cache_replicas(self):
        return await self._wrap_phase_method("resume_kv_cache_replicas", "kv_resume")

    @auto_await
    async def resume_generation_replicas(self):
        return await self._wrap_phase_method("resume_generation_replicas", "generation_resume")


class TimedCheckpointEngineManager(WeightSyncPhaseTimerMixin, CheckpointEngineManager):
    """``CheckpointEngineManager`` with per-phase weight-sync metrics.

    Used for the hybrid (colocated) side of separate-async training, where
    weights sync into the trainer-owned replicas at mode switches.
    """

    @auto_await
    async def update_weights(self, global_steps: int = None):
        return await self._profiled_update_weights(CheckpointEngineManager.update_weights.__wrapped__, global_steps)


class TimedOmniCheckpointEngineManager(WeightSyncPhaseTimerMixin, OmniCheckpointEngineManager):
    """``OmniCheckpointEngineManager`` with per-phase weight-sync metrics.

    Used for the standalone rollout side of separate-async training; keeps the
    LoRA ``peft_config`` forwarding and adds the phase timings to the metrics
    the parent returns.
    """

    @auto_await
    async def update_weights(self, global_steps: int = None):
        return await self._profiled_update_weights(OmniCheckpointEngineManager.update_weights.__wrapped__, global_steps)
