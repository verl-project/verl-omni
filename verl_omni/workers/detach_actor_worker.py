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
"""Diffusion-aware actor worker with local CPU parameter snapshots."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name
from verl.workers.config.distillation import DistillationConfig

from verl_omni.workers.engine_workers import ActorRolloutRefWorker


def _fsdp1_sharded_save_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy this rank's local FSDP1 parameter shards to CPU."""
    cpu_sharded_state = {}
    for param_name, param in model.named_parameters():
        cpu_sharded_state[param_name] = param.detach().to("cpu", copy=True)
    return cpu_sharded_state


def _fsdp1_sharded_load_from_cpu(model: torch.nn.Module, cpu_sharded_state: dict[str, torch.Tensor]) -> None:
    """Restore this rank's local FSDP1 parameter shards from CPU."""
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            if param_name not in cpu_sharded_state:
                continue
            param.copy_(cpu_sharded_state[param_name].to(param.device))

    if dist.is_initialized():
        dist.barrier()


class DiffusionDetachActorWorker(ActorRolloutRefWorker):
    """Add sharded CPU save/restore RPCs to the verl-omni hybrid worker."""

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        super().__init__(config, role, distillation_config=distillation_config, **kwargs)
        self._strategy_handlers: tuple[Callable, Callable] | None = None
        self.cpu_saved_models: dict[int, Any] = {}

    def _get_strategy_handlers(self) -> tuple[Callable, Callable]:
        if self._strategy_handlers is not None:
            return self._strategy_handlers

        strategy = self.config.actor.strategy
        if strategy == "fsdp":
            handlers = (_fsdp1_sharded_save_to_cpu, _fsdp1_sharded_load_from_cpu)
        elif strategy in ("fsdp2", "veomni"):
            from verl.utils.fsdp_utils import (
                fsdp2_sharded_load_from_cpu,
                fsdp2_sharded_save_to_cpu,
            )

            handlers = (fsdp2_sharded_save_to_cpu, fsdp2_sharded_load_from_cpu)
        elif strategy == "megatron":
            from verl.utils.megatron_utils import (
                copy_megatron_model_to_cpu,
                restore_megatron_model_from_cpu,
            )

            handlers = (copy_megatron_model_to_cpu, restore_megatron_model_from_cpu)
        else:
            raise NotImplementedError(f"Unsupported strategy: {strategy}")

        self._strategy_handlers = handlers
        return handlers

    @property
    def copy_handler(self) -> Callable:
        return self._get_strategy_handlers()[0]

    @property
    def restore_handler(self) -> Callable:
        return self._get_strategy_handlers()[1]

    @contextmanager
    def _actor_model_for_snapshot(self) -> Iterator[Any]:
        """Materialize offloaded parameters while a snapshot helper accesses them."""
        engine = self.actor.engine
        should_restore_offload = engine.is_param_offload_enabled
        try:
            if should_restore_offload:
                engine.to(get_device_name(), model=True, optimizer=False, grad=False)
            yield engine.module
        finally:
            if should_restore_offload:
                engine.to("cpu", model=True, optimizer=False, grad=False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, snapshot_id: int) -> None:
        """Save this rank's current actor parameter shard to CPU."""
        with self._actor_model_for_snapshot() as module:
            self.cpu_saved_models[snapshot_id] = self.copy_handler(module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, snapshot_id: int) -> None:
        """Restore this rank's actor parameter shard from a CPU snapshot."""
        if snapshot_id not in self.cpu_saved_models:
            raise KeyError(f"Unknown actor CPU snapshot: {snapshot_id}")

        saved_model = self.cpu_saved_models[snapshot_id]
        with self._actor_model_for_snapshot() as module:
            if self.config.actor.strategy in ("fsdp2", "veomni"):
                cpu_sharded_state, global_spec = saved_model
                self.restore_handler(module, cpu_sharded_state, global_spec)
            else:
                self.restore_handler(module, saved_model)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, snapshot_id: int) -> None:
        """Release a CPU actor snapshot."""
        self.cpu_saved_models.pop(snapshot_id, None)
