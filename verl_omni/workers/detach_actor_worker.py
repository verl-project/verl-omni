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

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name
from verl.workers.config.distillation import DistillationConfig

from verl_omni.workers.engine_workers import ActorRolloutRefWorker


def _clone_cpu_tensors(value):
    """Make snapshot tensors independent from CPU-offloaded parameters."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _clone_cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cpu_tensors(item) for item in value)
    if isinstance(value, list):
        return [_clone_cpu_tensors(item) for item in value]
    return value


class DiffusionDetachActorWorker(ActorRolloutRefWorker, DetachActorWorker):
    """Use verl's detach handlers with the verl-omni hybrid actor."""

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        ActorRolloutRefWorker.__init__(self, config, role, distillation_config=distillation_config, **kwargs)
        self._strategy_handlers = None
        self.cpu_saved_models: dict[int, Any] = {}

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
            self.cpu_saved_models[snapshot_id] = _clone_cpu_tensors(self.copy_handler(module))

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
