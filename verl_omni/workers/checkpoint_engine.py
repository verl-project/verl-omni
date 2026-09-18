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
        if backend != "delta_sharded":
            super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)
            return
        if rollout_config.name != "vllm_omni":
            raise NotImplementedError(
                f"checkpoint_engine.backend='delta_sharded' currently supports only the vllm_omni "
                f"rollout (got rollout.name={rollout_config.name!r}): the sparse apply lives in "
                "the vllm-omni worker extension."
            )
        # Mirror CheckpointEngineWorker.__init__ with the backend gate widened; see
        # verl/checkpoint_engine/base.py.
        from verl.checkpoint_engine import CheckpointEngineRegistry
        from verl.single_controller.base import Worker
        from verl.utils.distributed import initialize_global_process_group_ray
        from verl.utils.import_utils import import_external_libs
        from verl.workers.rollout import get_rollout_class

        Worker.__init__(self)
        self.rollout_config = rollout_config
        self.model_config = model_config
        self.server_adapter = server_adapter
        bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        import_external_libs(self.rollout_config.checkpoint_engine.custom_backend_module or None)
        self.checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=bucket_size, **engine_kwargs)
        self.extra_rollout_args = args
        self.extra_rollout_kwargs = kwargs
        if self.server_adapter is None:
            self.server_adapter = get_rollout_class(self.rollout_config.name, self.rollout_config.mode)(
                *self.extra_rollout_args,
                config=self.rollout_config,
                model_config=self.model_config,
                device_mesh=None,
                **self.extra_rollout_kwargs,
            )
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")
