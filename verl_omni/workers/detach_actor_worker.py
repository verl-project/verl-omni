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
        mesh = param.device_mesh
        mesh_layout = (
            mesh.device_type,
            tuple(mesh.mesh.shape),
            tuple(mesh.mesh.flatten().tolist()),
            mesh.mesh_dim_names,
        )
        return (*layout, mesh_layout, tuple(param.placements), param.to_local().shape)
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
            error = None
            try:
                if should_restore_offload:
                    engine.to(get_device_name(), model=True, optimizer=False, grad=False)
                module = engine.module
            except Exception as exc:
                error = exc
            if not self._snapshot_agreement(error is None)[0]:
                raise RuntimeError("Actor snapshot materialization failed on at least one actor rank") from error
            yield module
        finally:
            if should_restore_offload:
                engine.to("cpu", model=True, optimizer=False, grad=False)

    def _snapshot_agreement(self, *flags: bool) -> tuple[bool, ...]:
        """Agree before helpers, including full-snapshot backends with internal collectives."""
        if torch.distributed.is_initialized():
            values = torch.tensor(flags, dtype=torch.int32, device=get_device_name())
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MIN)
            return tuple(bool(value) for value in values.tolist())
        return flags

    def _supports_trainable_snapshot(self, module: torch.nn.Module) -> bool:
        engine = self.actor.engine
        model_config = getattr(engine, "model_config", None)
        engine_config = getattr(engine, "engine_config", None)
        return (
            self.config.actor.strategy == "fsdp2"
            and getattr(model_config, "lora_rank", 0) > 0
            and tuple(getattr(model_config, "policy_state_adapters", ())) == ("default",)
            and getattr(engine_config, "ulysses_sequence_parallel_size", None) == 1
            and not getattr(engine, "_uses_fsdp2_cpu_offload_policy", False)
            and tuple(getattr(module, "peft_config", ())) == ("default",)
        )

    def _restore_trainable_snapshot(self, module: torch.nn.Module, snapshot: _TrainableSnapshot) -> None:
        error = None
        compatible = False
        try:
            trainable = {name: param for name, param in module.named_parameters() if param.requires_grad}
            layouts = tuple(_parameter_layout(param) for param in trainable.values())
            compatible = (
                self._supports_trainable_snapshot(module)
                and id(module) == snapshot.module_id
                and tuple(trainable) == snapshot.names
                and layouts == snapshot.layouts
                and _trainable_aliases(module) == snapshot.aliases
            )
            view = torch.nn.ParameterList(trainable.values())
            cpu_sharded_state, global_spec = snapshot.state
        except Exception as exc:
            error = exc
        if not self._snapshot_agreement(error is None and compatible)[0]:
            raise RuntimeError("Trainable snapshot parameter mapping changed on at least one actor rank") from error

        # ParameterList retains the live Parameters; the upstream helper owns shard copies and synchronization.
        self.restore_handler(view, cpu_sharded_state, global_spec)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, snapshot_id: int) -> None:
        """Save this rank's current actor parameter shard to CPU."""
        with self._actor_model_for_snapshot() as module:
            error = None
            eligible = False
            try:
                eligible = self._supports_trainable_snapshot(module)
                if eligible:
                    params = dict(module.named_parameters())
                    trainable = {name: param for name, param in params.items() if param.requires_grad}
                    eligible = len(trainable) < len(params) and any(
                        isinstance(param, DTensor) for param in trainable.values()
                    )
                if eligible:
                    view = torch.nn.ParameterList(trainable.values())
                    layouts = tuple(_parameter_layout(param) for param in trainable.values())
                    aliases = _trainable_aliases(module)
            except Exception as exc:
                error = exc
            # Actor ranks share a strategy; agree on the representation before restore can dispatch collectives.
            prepared, eligible = self._snapshot_agreement(error is None, eligible)
            if not prepared:
                raise RuntimeError("Actor snapshot preparation failed on at least one actor rank") from error
            # The helper owns its collectives. Once it returns, CPU allocation can
            # still fail locally; no rank may publish/overwrite until all clones exist.
            state = self.copy_handler(view if eligible else module)
            error = None
            try:
                saved = _clone_cpu_tensors(state)
                if eligible:
                    saved = _TrainableSnapshot(
                        module_id=id(module),
                        names=tuple(trainable),
                        layouts=layouts,
                        aliases=aliases,
                        state=saved,
                    )
            except Exception as exc:
                error = exc
            if not self._snapshot_agreement(error is None)[0]:
                raise RuntimeError("Actor snapshot CPU capture failed on at least one actor rank") from error
            self.cpu_saved_models[snapshot_id] = saved

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, snapshot_id: int) -> None:
        """Restore this rank's actor parameter shard from a CPU snapshot."""
        present = snapshot_id in self.cpu_saved_models
        saved_model = self.cpu_saved_models.get(snapshot_id)
        trainable = isinstance(saved_model, _TrainableSnapshot)
        present, all_trainable, all_full = self._snapshot_agreement(present, trainable, not trainable)
        if not present:
            raise KeyError(f"Unknown actor CPU snapshot: {snapshot_id}")
        if not (all_trainable or all_full):
            raise RuntimeError("Actor snapshot representation differs across actor ranks")

        with self._actor_model_for_snapshot() as module:
            if isinstance(saved_model, _TrainableSnapshot):
                self._restore_trainable_snapshot(module, saved_model)
            elif self.config.actor.strategy in ("fsdp2", "veomni"):
                error = None
                try:
                    cpu_sharded_state, global_spec = saved_model
                except Exception as exc:
                    error = exc
                if not self._snapshot_agreement(error is None)[0]:
                    raise RuntimeError("Actor snapshot state is invalid on at least one actor rank") from error
                self.restore_handler(module, cpu_sharded_state, global_spec)
            else:
                self.restore_handler(module, saved_model)
