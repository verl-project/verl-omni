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
from dataclasses import dataclass
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from torch.distributed.tensor import DTensor
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


def _parameter_layout(param: torch.nn.Parameter) -> tuple:
    layout = (param.shape, param.dtype)
    if isinstance(param, DTensor):
        return (*layout, param.device_mesh, param.placements, param.to_local().shape)
    return layout


def _trainable_aliases(module: torch.nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, id(param)) for name, param in module.named_parameters(remove_duplicate=False) if param.requires_grad
    )


@dataclass
class _TrainableSnapshot:
    """Rank-local payload and the parameter mapping required to restore it."""

    module_id: int
    names: tuple[str, ...]
    layouts: tuple[tuple, ...]
    aliases: tuple[tuple[str, int], ...]
    state: Any


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

    def _supports_trainable_snapshot(self, module: torch.nn.Module) -> bool:
        engine = self.actor.engine
        model_config = getattr(engine, "model_config", None)
        engine_config = getattr(engine, "engine_config", None)
        return (
            self.config.actor.strategy == "fsdp2"
            and getattr(model_config, "architecture", None) == "QwenImagePipeline"
            and getattr(model_config, "lora_rank", 0) > 0
            and tuple(getattr(model_config, "policy_state_adapters", ())) == ("default",)
            and getattr(engine_config, "ulysses_sequence_parallel_size", None) == 1
            and not getattr(engine, "_uses_fsdp2_cpu_offload_policy", False)
            and tuple(getattr(module, "peft_config", ())) == ("default",)
        )

    def _restore_trainable_snapshot(self, module: torch.nn.Module, snapshot: _TrainableSnapshot) -> None:
        trainable = {name: param for name, param in module.named_parameters() if param.requires_grad}
        layouts = tuple(_parameter_layout(param) for param in trainable.values())
        compatible = (
            self._supports_trainable_snapshot(module)
            and id(module) == snapshot.module_id
            and tuple(trainable) == snapshot.names
            and layouts == snapshot.layouts
            and _trainable_aliases(module) == snapshot.aliases
        )
        if torch.distributed.is_initialized():
            invalid = torch.tensor(int(not compatible), device=get_device_name())
            torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
            compatible = invalid.item() == 0
        if not compatible:
            raise RuntimeError("Trainable snapshot parameter mapping changed on at least one actor rank")

        # ParameterList retains the live Parameters; the upstream helper owns shard copies and synchronization.
        view = torch.nn.ParameterList(trainable.values())
        cpu_sharded_state, global_spec = snapshot.state
        self.restore_handler(view, cpu_sharded_state, global_spec)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, snapshot_id: int) -> None:
        """Save this rank's current actor parameter shard to CPU."""
        with self._actor_model_for_snapshot() as module:
            trainable = {}
            eligible = self._supports_trainable_snapshot(module)
            if eligible:
                params = dict(module.named_parameters())
                trainable = {name: param for name, param in params.items() if param.requires_grad}
                eligible = len(trainable) < len(params) and any(
                    isinstance(param, DTensor) for param in trainable.values()
                )
            # Actor ranks share a strategy; agree on the representation before restore can dispatch collectives.
            if self.config.actor.strategy == "fsdp2" and torch.distributed.is_initialized():
                flag = torch.tensor(int(eligible), device=get_device_name())
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                eligible = flag.item() == 1
            if eligible:
                view = torch.nn.ParameterList(trainable.values())
                state = _clone_cpu_tensors(self.copy_handler(view))
                self.cpu_saved_models[snapshot_id] = _TrainableSnapshot(
                    module_id=id(module),
                    names=tuple(trainable),
                    layouts=tuple(_parameter_layout(param) for param in trainable.values()),
                    aliases=_trainable_aliases(module),
                    state=state,
                )
                return
            self.cpu_saved_models[snapshot_id] = _clone_cpu_tensors(self.copy_handler(module))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, snapshot_id: int) -> None:
        """Restore this rank's actor parameter shard from a CPU snapshot."""
        if snapshot_id not in self.cpu_saved_models:
            raise KeyError(f"Unknown actor CPU snapshot: {snapshot_id}")

        saved_model = self.cpu_saved_models[snapshot_id]
        with self._actor_model_for_snapshot() as module:
            if isinstance(saved_model, _TrainableSnapshot):
                self._restore_trainable_snapshot(module, saved_model)
            elif self.config.actor.strategy in ("fsdp2", "veomni"):
                cpu_sharded_state, global_spec = saved_model
                self.restore_handler(module, cpu_sharded_state, global_spec)
            else:
                self.restore_handler(module, saved_model)
