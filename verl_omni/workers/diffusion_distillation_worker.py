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
"""Ray worker and local runtime for multi-role diffusion distillation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from verl.protocol import DataProtoFuture
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.engine import EngineRegistry

from verl_omni.pipelines.model_base import DiffusionModelBase, DistributionMatchingModelAdapter
from verl_omni.trainer.diffusion.distillation.contracts import (
    DistillationPlan,
    PhaseRequest,
    PhaseResult,
    RoleBinding,
)
from verl_omni.utils.fs import resolve_model_local_dir
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.config.diffusion import DiffusionDistributionMatchingConfig
from verl_omni.workers.engine.fsdp.distillation_impl import DistillationRoleGroupEngine

__all__ = [
    "DistillationPhaseComputation",
    "DistillationPhaseRunner",
    "DistillationRoleRuntime",
    "DiffusionDistillationWorker",
    "DiffusionDistillationWorkerGroup",
]


def resolve_profiler_configs(omega_profiler_config):
    """Resolve the selected profiler and its typed tool configuration."""
    profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
    tool = omega_profiler_config.get("tool", None)
    if tool in {"npu", "nsys", "torch", "torch_memory", "precision_debugger"}:
        tool_config = omega_conf_to_dataclass(omega_profiler_config.get("tool_config", {}).get(tool))
    else:
        tool_config = None
    return profiler_config, tool_config


@dataclass
class DistillationPhaseComputation:
    """Scalar role losses and detached metrics produced by an architecture runner."""

    losses: dict[str, torch.Tensor]
    metrics: dict[str, float]


@runtime_checkable
class DistillationPhaseRunner(Protocol):
    """Architecture-owned differentiable computation for one generic phase."""

    def compute_phase(
        self,
        request: PhaseRequest,
        batch: TensorDict,
        runtime: DistillationRoleRuntime,
    ) -> DistillationPhaseComputation:
        """Build scalar losses while the runtime owns role modules and optimization."""
        ...

    def state_dict(self) -> dict:
        """Return architecture-owned RNG and rollout state."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore architecture-owned RNG and rollout state."""
        ...


class DistillationRoleRuntime:
    """Role-to-engine router shared by local tests and the Ray worker."""

    def __init__(
        self,
        plan: DistillationPlan,
        engines: Mapping[str, DistillationRoleGroupEngine],
        *,
        ema_decay: float,
        ema_start_step: int,
        micro_batch_sizes: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.plan = plan
        self.engines = dict(engines)
        self.bindings = {binding.role: binding for binding in plan.role_layout.bindings}
        self.ema_decay = ema_decay
        self.ema_start_step = ema_start_step
        self.micro_batch_sizes = dict(micro_batch_sizes or {"student": 1, "fake_score": 1})
        expected_groups = {group.name for group in plan.role_layout.groups}
        missing_groups = expected_groups - set(self.engines)
        extra_groups = set(self.engines) - expected_groups
        if missing_groups or extra_groups:
            raise ValueError(
                f"Role-group engines must match the plan exactly; missing={sorted(missing_groups)}, "
                f"extra={sorted(extra_groups)}."
            )
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {ema_decay}.")
        if ema_start_step < 0:
            raise ValueError(f"EMA start step must be non-negative, got {ema_start_step}.")
        if set(self.micro_batch_sizes) != {"student", "fake_score"} or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.micro_batch_sizes.values()
        ):
            raise ValueError("micro_batch_sizes must define positive integer student and fake_score sizes.")
        self.initialize_ema()

    def engine_for_role(self, role: str) -> DistillationRoleGroupEngine:
        """Resolve the physical engine backing a semantic role."""
        try:
            binding = self.bindings[role]
        except KeyError:
            raise KeyError(f"Unknown distillation role {role!r}; bound roles: {sorted(self.bindings)}.") from None
        return self.engines[binding.group]

    @contextmanager
    def use_role(self, role: str, *, grad_enabled: Optional[bool] = None):
        """Yield a role model with explicit gradient intent and restore it afterward."""
        engine = self.engine_for_role(role)
        with engine.use_role(role, grad_enabled=grad_enabled) as module:
            yield module

    def scheduler_for_role(self, role: str):
        """Return the diffusion scheduler belonging to a role's physical group."""
        return self.engine_for_role(role).scheduler

    def micro_batch_size(self, phase_kind: str) -> int:
        """Return the per-device micro-batch size for one phase kind."""
        try:
            return self.micro_batch_sizes[phase_kind]
        except KeyError:
            raise ValueError(f"Unknown distillation phase kind {phase_kind!r}.") from None

    def model_config_for_role(self, role: str):
        """Return the resolved model config belonging to a semantic role."""
        return self.engine_for_role(role).model_config

    def export_tensors(self, *, base_sync_done: bool):
        """Export the plan-selected student or EMA role for inference sync."""
        role = self.plan.export.role
        return self.engine_for_role(role).iter_export_tensors(role, base_sync_done)

    def zero_grad(self, roles: tuple[str, ...]) -> None:
        """Clear gradient state for each requested trainable role."""
        for role in roles:
            self.engine_for_role(role).optimizer_zero_grad(role)

    def validate_computation(
        self,
        request: PhaseRequest,
        computation: DistillationPhaseComputation,
    ) -> str:
        """Require one graph-bearing scalar loss owned by the requested role."""
        if not isinstance(computation, DistillationPhaseComputation):
            raise TypeError(f"compute_phase must return DistillationPhaseComputation, got {type(computation)}.")
        expected_roles = set(request.trainable_roles)
        if set(computation.losses) != expected_roles:
            raise ValueError(
                f"Phase computation losses must match requested roles {sorted(expected_roles)}, "
                f"got {sorted(computation.losses)}."
            )
        if len(request.trainable_roles) != 1:
            raise NotImplementedError(
                "Multi-role optimizer phases are not supported by the current distillation runtime."
            )
        role = request.trainable_roles[0]
        loss = computation.losses[role]
        if loss.ndim != 0:
            raise ValueError(f"Role loss must be scalar, got shape {tuple(loss.shape)} for {role!r}.")
        if not loss.requires_grad:
            raise ValueError(f"Role loss for {role!r} must retain an autograd graph.")
        return role

    def backward_micro_batch(
        self,
        request: PhaseRequest,
        computation: DistillationPhaseComputation,
        *,
        weight: float,
    ) -> None:
        """Accumulate one weighted micro-batch loss without stepping."""
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"Micro-batch weight must be in (0, 1], got {weight}.")
        role = self.validate_computation(request, computation)
        self.engine_for_role(role).backward_role(role, computation.losses[role] * weight)

    def step_phase(self, request: PhaseRequest) -> tuple[dict[str, int], dict[str, float]]:
        """Step the phase optimizer once after all micro-batches were accumulated."""
        if len(request.trainable_roles) != 1:
            raise NotImplementedError(
                "Multi-role optimizer phases are not supported by the current distillation runtime."
            )
        role = request.trainable_roles[0]
        for role_engine in self.engines.values():
            if hasattr(role_engine, "assert_gradient_isolation"):
                role_engine.assert_gradient_isolation({role})
        engine = self.engine_for_role(role)
        optimizer_start = time.perf_counter()
        stepped, grad_norm = engine.optimizer_step(role)
        metrics = {
            f"{role}/grad_norm": grad_norm,
            f"perf/{role}_optimizer_s": time.perf_counter() - optimizer_start,
        }
        if stepped and getattr(engine, "lr_schedulers", {}).get(role) is not None:
            metrics[f"{role}/lr"] = float(engine.lr_schedulers[role].get_last_lr()[0])
        if not stepped:
            return {}, metrics
        if request.update_ema and request.global_step + 1 >= self.ema_start_step:
            ema_start = time.perf_counter()
            self.update_ema()
            metrics["ema/decay"] = self.ema_decay
            metrics["perf/ema_update_s"] = time.perf_counter() - ema_start
        return {role: 1}, metrics

    def backward_and_step(
        self,
        request: PhaseRequest,
        computation: DistillationPhaseComputation,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Convenience path for a one-micro-batch phase."""
        self.backward_micro_batch(request, computation, weight=1.0)
        optimizer_steps, metrics = self.step_phase(request)
        role = request.trainable_roles[0]
        metrics.update(computation.metrics)
        metrics[f"{role}/loss"] = float(computation.losses[role].detach().float().item())
        return optimizer_steps, metrics

    def initialize_ema(self) -> None:
        """Initialize the semantic EMA role exactly from the student role."""
        self.update_ema_parameters(decay=0.0)

    def update_ema(self) -> None:
        """Update the semantic student EMA in shared or independent storage."""
        self.update_ema_parameters(decay=self.ema_decay)

    def update_ema_parameters(self, decay: float) -> None:
        """Route EMA updates according to the physical role-group layout."""
        student_engine = self.engine_for_role("student")
        ema_engine = self.engine_for_role("student_ema")
        if student_engine is ema_engine:
            student_engine.update_role_ema("student", "student_ema", decay)
        else:
            ema_engine.update_module_ema_from(student_engine, decay)

    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Average scalar metrics across data-parallel replicas."""
        if not metrics:
            return metrics
        first_engine = next(iter(self.engines.values()))
        group = first_engine.get_data_parallel_group()
        if group is None:
            return metrics
        names = sorted(metrics)
        values = torch.tensor([metrics[name] for name in names], dtype=torch.float32, device=get_device_id())
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.AVG, group=group)
        return {name: value for name, value in zip(names, values.cpu().tolist(), strict=True)}

    def group_metrics(self) -> dict[str, float]:
        """Return stable placement diagnostics for logging."""
        return {
            "memory/role_group_count": float(len(self.engines)),
            "memory/base_model_copies": float(len(self.engines)),
        }


class DiffusionDistillationWorker(Worker, DistProfilerExtension):
    """Own colocated role-group engines and execute one phase per Ray RPC."""

    def __init__(self, config: DictConfig, plan: DistillationPlan):
        Worker.__init__(self)
        if is_npu_available:
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        initialize_global_process_group_ray(timeout_second=None)
        set_numa_affinity()

        self.config = config
        self.plan = plan
        self.device_name = get_device_name()
        profiler_config, tool_config = resolve_profiler_configs(config.actor_rollout_ref.actor.get("profiler", {}))
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )
        self.runtime: Optional[DistillationRoleRuntime] = None
        self.phase_runner: Optional[DistillationPhaseRunner] = None

    def build_optimizer_configs(
        self,
        bindings: tuple[RoleBinding, ...],
        student_optimizer_config: FSDPOptimizerConfig,
        distillation_config: DiffusionDistributionMatchingConfig,
    ) -> dict[str, FSDPOptimizerConfig]:
        """Give each trainable role its own optimizer configuration."""
        configs = {}
        for binding in bindings:
            if not binding.trainable:
                continue
            if binding.role == "student":
                configs[binding.role] = deepcopy(student_optimizer_config)
            elif binding.role == "fake_score":
                configs[binding.role] = deepcopy(distillation_config.fake_score_optim)
            else:
                raise NotImplementedError(f"No optimizer config is defined for role {binding.role!r}.")
        return configs

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        """Allocate every physical group once and bind the architecture phase runner."""
        model_config: DiffusionModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
        # DMD losses live in the phase runner, so instantiate only the actor sub-configs the role engine consumes.
        actor_config = self.config.actor_rollout_ref.actor
        actor_engine_config = omega_conf_to_dataclass(actor_config.fsdp_config)
        object.__setattr__(actor_engine_config, "strategy", actor_config.strategy)
        student_optimizer_config: FSDPOptimizerConfig = omega_conf_to_dataclass(
            actor_config.optim,
            dataclass_type=FSDPOptimizerConfig,
        )
        checkpoint_config = omega_conf_to_dataclass(actor_config.checkpoint)
        distillation_config: DiffusionDistributionMatchingConfig = omega_conf_to_dataclass(
            self.config.distillation.distribution_matching,
            dataclass_type=DiffusionDistributionMatchingConfig,
        )
        adapter_cls = DiffusionModelBase.get_class(model_config)
        if not issubclass(adapter_cls, DistributionMatchingModelAdapter):
            raise TypeError(
                f"{adapter_cls.__name__} must mix in DistributionMatchingModelAdapter for distillation training."
            )

        engines = {}
        resolved_model_paths = {}
        for group in self.plan.role_layout.groups:
            if group.placement != "colocated":
                raise NotImplementedError("The current runtime implements colocated role groups only.")
            bindings = tuple(binding for binding in self.plan.role_layout.bindings if binding.group == group.name)
            trainable_bindings = tuple(binding for binding in bindings if binding.trainable)
            group_model_config = deepcopy(model_config)
            object.__setattr__(group_model_config, "model_type", "diffusion_distillation_model")
            object.__setattr__(group_model_config, "path", group.model_ref)
            if group.model_ref not in resolved_model_paths:
                resolved_model_paths[group.model_ref] = resolve_model_local_dir(
                    group.model_ref, use_shm=group_model_config.use_shm
                )
            object.__setattr__(group_model_config, "local_path", resolved_model_paths[group.model_ref])
            adapters = tuple(binding.adapter for binding in bindings if binding.adapter is not None)
            if group.storage == "shared_base_adapters" and group_model_config.lora_rank <= 0:
                raise ValueError("shared_base_adapters requires actor_rollout_ref.model.lora_rank > 0.")
            if group.storage == "independent_module" and not adapters:
                object.__setattr__(group_model_config, "lora_rank", 0)
                object.__setattr__(group_model_config, "lora_adapter_path", None)
            if group_model_config.lora_rank > 0 and adapters:
                object.__setattr__(
                    group_model_config,
                    "policy_state_adapters",
                    tuple(dict.fromkeys(("default", *adapters, "reference"))),
                )

            group_engine_config = deepcopy(actor_engine_config)
            object.__setattr__(group_engine_config, "forward_only", not trainable_bindings)
            role_optimizer_configs = self.build_optimizer_configs(
                bindings, student_optimizer_config, distillation_config
            )
            primary_optimizer_config = next(iter(role_optimizer_configs.values()), deepcopy(student_optimizer_config))
            engine = EngineRegistry.new(
                model_type="diffusion_distillation_model",
                backend=group_engine_config.strategy,
                model_config=group_model_config,
                engine_config=group_engine_config,
                optimizer_config=primary_optimizer_config,
                checkpoint_config=deepcopy(checkpoint_config),
                role_group=group,
                role_bindings=bindings,
                optimizer_configs=role_optimizer_configs,
            )
            engine.initialize()
            engines[group.name] = engine

        self.runtime = DistillationRoleRuntime(
            self.plan,
            engines,
            ema_decay=distillation_config.ema_decay,
            ema_start_step=distillation_config.ema_start_step,
            micro_batch_sizes={
                "student": distillation_config.student_micro_batch_size_per_gpu,
                "fake_score": distillation_config.fake_score_micro_batch_size_per_gpu,
            },
        )
        self.phase_runner = adapter_cls.build_distillation_phase_runner(model_config, self.plan)
        if not isinstance(self.phase_runner, DistillationPhaseRunner):
            raise TypeError(
                "build_distillation_phase_runner() must return an object implementing "
                "compute_phase(), state_dict(), and load_state_dict()."
            )
        first_engine = next(iter(engines.values()))
        self._register_dispatch_collect_info(
            mesh_name="distillation",
            dp_rank=first_engine.get_data_parallel_rank(),
            is_collect=first_engine.is_mp_src_rank_with_outputs(),
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="distillation"), blocking=True)
    @DistProfiler.annotate(color="red", role="distillation_phase")
    def execute_phase(self, data: TensorDict) -> TensorDict:
        """Resolve rank failures before lazy collect metadata RPCs can wait behind peer collectives."""
        if self.runtime is None or self.phase_runner is None:
            raise RuntimeError("init_model() must be called before execute_phase().")
        request = tu.pop(data, key="phase_request")
        if not isinstance(request, PhaseRequest):
            raise TypeError(f"phase_request must be PhaseRequest, got {type(request)}.")
        self.runtime.zero_grad(request.trainable_roles)
        device_module = get_torch_device()
        device_module.reset_peak_memory_stats()
        start = time.perf_counter()
        if not data.batch_size:
            raise ValueError("A distillation phase requires a TensorDict with a leading batch dimension.")
        total_samples = data.batch_size[0]
        if total_samples <= 0:
            raise ValueError("A distillation phase cannot execute an empty batch.")
        micro_batch_size = self.runtime.micro_batch_size(request.kind)
        accumulated_metrics: dict[str, float] = {}
        accumulated_losses: dict[str, float] = {}
        with ExitStack() as stack:
            for engine in self.runtime.engines.values():
                context = engine.train_mode() if engine.optimizers else engine.eval_mode()
                stack.enter_context(context)
            forward_duration = 0.0
            backward_duration = 0.0
            for micro_batch in data.split(micro_batch_size, dim=0):
                weight = micro_batch.batch_size[0] / total_samples
                micro_batch = micro_batch.to(get_device_id())
                forward_start = time.perf_counter()
                computation = self.phase_runner.compute_phase(request, micro_batch, self.runtime)
                forward_duration += time.perf_counter() - forward_start
                backward_start = time.perf_counter()
                self.runtime.backward_micro_batch(request, computation, weight=weight)
                backward_duration += time.perf_counter() - backward_start
                for name, value in computation.metrics.items():
                    accumulated_metrics[name] = accumulated_metrics.get(name, 0.0) + float(value) * weight
                for role, loss in computation.losses.items():
                    accumulated_losses[role] = accumulated_losses.get(role, 0.0) + float(loss.detach().float()) * weight
            optimizer_steps, step_metrics = self.runtime.step_phase(request)
            metrics = {**accumulated_metrics, **step_metrics}
            metrics.update({f"{role}/loss": loss for role, loss in accumulated_losses.items()})
            metrics[f"perf/{request.kind}_forward_s"] = forward_duration
            metrics[f"perf/{request.kind}_backward_s"] = backward_duration
            metrics[f"perf/{request.kind}_s"] = time.perf_counter() - start
            metrics["memory/max_allocated_gb"] = device_module.max_memory_allocated() / (1024**3)
            metrics["memory/max_reserved_gb"] = device_module.max_memory_reserved() / (1024**3)
            metrics.update(self.runtime.group_metrics())
            metrics = self.runtime.reduce_metrics(metrics)
        return tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": metrics, "optimizer_steps": optimizer_steps},
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path: str, global_step: int) -> None:
        """Save every physical role group under one checkpoint root."""
        if self.runtime is None:
            raise RuntimeError("init_model() must be called before save_checkpoint().")
        os.makedirs(local_path, exist_ok=True)
        for group_name, engine in self.runtime.engines.items():
            engine.save_role_group_checkpoint(os.path.join(local_path, "role_groups", group_name), global_step)
        runner_state = self.phase_runner.state_dict() if hasattr(self.phase_runner, "state_dict") else {}
        torch.save(runner_state, os.path.join(local_path, f"phase_runner_rank_{self.rank}.pt"))
        if self.rank == 0:
            with open(os.path.join(local_path, "worker_manifest.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "plan_name": self.plan.name,
                        "plan_version": self.plan.version,
                        "groups": [asdict(group) for group in self.plan.role_layout.groups],
                        "bindings": [asdict(binding) for binding in self.plan.role_layout.bindings],
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )
        torch.distributed.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path: str) -> None:
        """Restore every role group after validating the worker manifest."""
        if self.runtime is None:
            raise RuntimeError("init_model() must be called before load_checkpoint().")
        manifest_path = os.path.join(local_path, "worker_manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Missing distillation worker manifest: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("plan_name") != self.plan.name or manifest.get("plan_version") != self.plan.version:
            raise ValueError("Checkpoint recipe identity does not match the active distillation plan.")
        expected_groups = [asdict(group) for group in self.plan.role_layout.groups]
        expected_bindings = [asdict(binding) for binding in self.plan.role_layout.bindings]
        if manifest.get("groups") != expected_groups or manifest.get("bindings") != expected_bindings:
            raise ValueError("Checkpoint role layout does not match the active distillation plan.")
        for group_name, engine in self.runtime.engines.items():
            engine.load_role_group_checkpoint(os.path.join(local_path, "role_groups", group_name))
        runner_state_path = os.path.join(local_path, f"phase_runner_rank_{self.rank}.pt")
        if not os.path.isfile(runner_state_path):
            raise FileNotFoundError(f"Missing phase-runner state: {runner_state_path}")
        runner_state = torch.load(runner_state_path, map_location="cpu", weights_only=False)
        if runner_state:
            if not hasattr(self.phase_runner, "load_state_dict"):
                raise ValueError("Checkpoint contains phase-runner state, but the active runner cannot restore it.")
            self.phase_runner.load_state_dict(runner_state)


class DiffusionDistillationWorkerGroup:
    """Driver-side executor facade over a Ray worker group."""

    def __init__(self, worker_group) -> None:
        self.worker_group = worker_group

    def execute_phase(self, request: PhaseRequest, batch: TensorDict) -> PhaseResult:
        """Run one distributed phase and convert the collected TensorDict result."""
        batch = batch.copy()
        tu.assign_non_tensor(batch, phase_request=request)
        output = self.worker_group.execute_phase(batch)
        if isinstance(output, DataProtoFuture):
            output = output.get()
        metrics = dict(tu.get(output, "metrics"))
        optimizer_steps = dict(tu.get(output, "optimizer_steps"))
        return PhaseResult(metrics=metrics, optimizer_steps=optimizer_steps)

    def save_checkpoint(self, local_path: str, global_step: int) -> None:
        """Save all worker-local role groups."""
        self.worker_group.save_checkpoint(local_path, global_step)

    def load_checkpoint(self, local_path: str) -> None:
        """Restore all worker-local role groups."""
        self.worker_group.load_checkpoint(local_path)
