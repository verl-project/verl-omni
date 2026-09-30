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
# stream into sentinel-named pairs, so verl's unmodified worker AND verl's
# unmodified vLLM ServerAdapter drive the whole sync -- no worker subclass, no
# gate widening, no adapter subclass. Importing this module performs the
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
        workers before delegating to the parent ``update_weights``.
        """
        if self.backend != "naive":
            peft_config = self._fetch_actor_lora_peft_config()
            self._lora_peft_config = peft_config
            await self._push_lora_peft_config_to_replicas(peft_config)
        await super().update_weights(global_steps=global_steps)

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
