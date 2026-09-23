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
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineWorker
from verl.utils.ray_utils import auto_await


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


class OmniCheckpointEngineWorker(CheckpointEngineWorker):
    """``CheckpointEngineWorker`` that admits ``delta_sharded`` for the vllm_omni rollout.

    verl gates the delta backend to sglang at construction because its sparse apply
    rides sglang's custom-weight-loader hook; verl-omni applies deltas in the
    vllm-omni worker extension (``update_weights_from_ipc``), so the gate is widened
    here. Other backends use the parent construction unchanged.
    """

    def __init__(self, rollout_config, model_config, server_adapter=None, *args, **kwargs):
        backend = rollout_config.checkpoint_engine.backend
        if backend != "delta_sharded" or rollout_config.name == "sglang":
            super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)
            return
        if rollout_config.name != "vllm_omni":
            raise NotImplementedError(
                f"checkpoint_engine.backend='delta_sharded' currently supports only the vllm_omni "
                f"rollout (got rollout.name={rollout_config.name!r}): the sparse apply lives in "
                "the vllm-omni worker extension."
            )
        # Run the parent construction body with a gate-inert backend so Worker init,
        # the rollout construction, and the process group stay verl's single source of
        # truth, then swap in the delta engine. `backend` is the one field
        # CheckpointEngineConfig lists in _mutable_fields.
        from verl.checkpoint_engine import CheckpointEngineRegistry
        from verl.utils.import_utils import import_external_libs

        rollout_config.checkpoint_engine.backend = "naive"
        try:
            super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)
        finally:
            rollout_config.checkpoint_engine.backend = "delta_sharded"
        bucket_size = rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        import_external_libs(rollout_config.checkpoint_engine.custom_backend_module or None)
        self.checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=bucket_size, **engine_kwargs)
