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
"""Multi-role FSDP runtime for distribution-matching distillation."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from typing import Any, Optional

import torch
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine.base import EngineRegistry

from verl_omni.trainer.diffusion.distillation.contracts import RoleBinding, RoleGroupSpec
from verl_omni.workers.config import DiffusionModelConfig

from .diffusers_impl import DiffusersFSDPEngine

__all__ = ["DistillationRoleGroupEngine"]


@EngineRegistry.register(
    model_type="diffusion_distillation_model",
    backend=["fsdp", "fsdp2"],
    device=["cuda", "npu"],
)
class DistillationRoleGroupEngine(DiffusersFSDPEngine):
    """One physical FSDP model with logical distillation-role bindings."""

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        *,
        role_group: RoleGroupSpec,
        role_bindings: tuple[RoleBinding, ...],
        optimizer_configs: Mapping[str, FSDPOptimizerConfig],
    ) -> None:
        self.role_group = role_group
        self.role_bindings = {binding.role: binding for binding in role_bindings}
        self.optimizer_configs = dict(optimizer_configs)
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.lr_schedulers: dict[str, Any] = {}
        self._role_parameters: dict[str, tuple[torch.nn.Parameter, ...]] = {}
        self._active_role: Optional[str] = None
        self._primary_role: Optional[str] = None
        self.validate_constructor_inputs(engine_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def validate_constructor_inputs(self, engine_config: FSDPEngineConfig) -> None:
        """Validate role ownership and supported FSDP adapter layouts."""
        if not self.role_bindings:
            raise ValueError(f"Role group {self.role_group.name!r} must contain at least one binding.")
        if any(binding.group != self.role_group.name for binding in self.role_bindings.values()):
            raise ValueError(f"Every binding passed to {self.role_group.name!r} must reference that group.")
        trainable = {role for role, binding in self.role_bindings.items() if binding.trainable}
        if set(self.optimizer_configs) != trainable:
            raise ValueError(
                f"Optimizer configs for group {self.role_group.name!r} must match trainable roles "
                f"{sorted(trainable)}, got {sorted(self.optimizer_configs)}."
            )
        if self.role_group.storage == "shared_base_adapters" and engine_config.strategy == "fsdp":
            if not engine_config.use_orig_params:
                raise ValueError("shared_base_adapters with FSDP1 requires engine.use_orig_params=true.")

    def initialize(self) -> None:
        """Build the shared/independent module and all role optimizer states."""
        super().initialize()
        if self.has_adapters():
            available_adapters = set(getattr(self.peft_model(), "peft_config", {}))
            required_adapters = {
                binding.adapter for binding in self.role_bindings.values() if binding.adapter is not None
            }
            missing_adapters = required_adapters - available_adapters
            if missing_adapters:
                raise ValueError(
                    f"Role group {self.role_group.name!r} is missing configured adapters {sorted(missing_adapters)}."
                )
            if "default" in available_adapters:
                for binding in self.role_bindings.values():
                    if binding.trainable and binding.adapter not in {None, "default"}:
                        self.copy_adapter(source="default", target=binding.adapter)
        if "student" in self.role_bindings and "student_ema" in self.role_bindings:
            student = self.role_bindings["student"]
            ema = self.role_bindings["student_ema"]
            if student.adapter and ema.adapter:
                self.copy_adapter(source=student.adapter, target=ema.adapter)
        self.activate_role(self._primary_role or next(iter(self.role_bindings)))

    def _build_model_optimizer(self) -> None:
        super()._build_model_optimizer()
        self.optimizers = {}
        self.lr_schedulers = {}
        self._role_parameters = {}

        if not any(binding.trainable for binding in self.role_bindings.values()):
            self.module.requires_grad_(False)

        owned_parameters: dict[int, str] = {}
        for role, binding in self.role_bindings.items():
            if not binding.trainable:
                continue
            with self.use_role(role):
                parameters = tuple(parameter for parameter in self.module.parameters() if parameter.requires_grad)
            if not parameters:
                raise ValueError(f"Trainable role {role!r} resolved no trainable parameters.")
            overlap = {owned_parameters[id(parameter)] for parameter in parameters if id(parameter) in owned_parameters}
            if overlap:
                raise ValueError(f"Trainable role {role!r} shares optimizer parameters with {sorted(overlap)}.")
            for parameter in parameters:
                owned_parameters[id(parameter)] = role

            optimizer_config = self.optimizer_configs[role]
            optimizer = build_optimizer(parameters, optimizer_config)
            previous_config = self.optimizer_config
            try:
                self.optimizer_config = optimizer_config
                lr_scheduler = self._build_lr_scheduler(optimizer)
            finally:
                self.optimizer_config = previous_config
            self._role_parameters[role] = parameters
            self.optimizers[role] = optimizer
            self.lr_schedulers[role] = lr_scheduler

        if self.optimizers:
            self._primary_role = next(iter(self.optimizers))
            self.optimizer = self.optimizers[self._primary_role]
            self.lr_scheduler = self.lr_schedulers[self._primary_role]
            self.optimizer_config = self.optimizer_configs[self._primary_role]
        else:
            self.optimizer = None
            self.lr_scheduler = None

    def peft_model(self):
        """Unwrap the FSDP1 root to reach the adapter interface."""
        return getattr(self.module, "_fsdp_wrapped_module", self.module)

    def has_adapters(self) -> bool:
        """Whether the physical model exposes named adapters."""
        return hasattr(self.peft_model(), "set_adapter")

    def set_adapters_enabled(self, enabled: bool) -> None:
        """Toggle adapters through the Diffusers or PEFT interface."""
        peft_model = self.peft_model()
        method_name = "enable_adapters" if enabled else "disable_adapters"
        method = getattr(peft_model, method_name, None)
        if method is None:
            fallback_name = "enable_adapter_layers" if enabled else "disable_adapter_layers"
            method = getattr(getattr(peft_model, "base_model", peft_model), fallback_name, None)
        if method is None:
            raise AttributeError(f"PEFT model does not implement {method_name}().")
        method()

    def activate_role(self, role: str) -> None:
        """Select one logical role and its optimizer without entering a context."""
        try:
            binding = self.role_bindings[role]
        except KeyError:
            raise KeyError(
                f"Role {role!r} is not bound to group {self.role_group.name!r}; "
                f"bound roles: {sorted(self.role_bindings)}."
            ) from None

        if self.has_adapters():
            peft_model = self.peft_model()
            if binding.adapter is None:
                self.set_adapters_enabled(False)
            else:
                self.set_adapters_enabled(True)
                peft_model.set_adapter(binding.adapter)
        elif binding.adapter is not None and self.role_group.storage == "shared_base_adapters":
            raise ValueError(
                f"Shared-base role {role!r} requires adapter {binding.adapter!r}, but the model has no PEFT adapters."
            )

        self._active_role = role
        if role in self.optimizers:
            self.optimizer = self.optimizers[role]
            self.lr_scheduler = self.lr_schedulers[role]
            self.optimizer_config = self.optimizer_configs[role]

    @contextmanager
    def use_role(self, role: str, *, grad_enabled: Optional[bool] = None) -> Iterator[torch.nn.Module]:
        """Activate one role with explicit train/eval and autograd state."""
        previous_role = self._active_role
        previous_training = self.module.training
        binding = self.role_bindings[role]
        effective_grad = binding.trainable if grad_enabled is None else binding.trainable and grad_enabled
        self.activate_role(role)
        self.module.train(effective_grad)
        grad_context = nullcontext() if effective_grad else torch.no_grad()
        try:
            with grad_context:
                yield self.module
        finally:
            self.module.train(previous_training)
            if previous_role is not None:
                self.activate_role(previous_role)
            elif self._primary_role is not None:
                self.activate_role(self._primary_role)

    def parameters_for_role(self, role: str) -> tuple[torch.nn.Parameter, ...]:
        """Return the exact optimizer-owned parameters for a trainable role."""
        try:
            return self._role_parameters[role]
        except KeyError:
            raise ValueError(f"Role {role!r} has no optimizer-owned parameters.") from None

    def optimizer_zero_grad(self, role: Optional[str] = None) -> None:
        """Clear one role optimizer or every optimizer in the group."""
        optimizers = self.optimizers.values() if role is None else (self.optimizers[role],)
        for optimizer in optimizers:
            optimizer.zero_grad()

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True) -> None:
        """Move one physical model and every role optimizer across the offload boundary."""
        from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer

        super().to(device=device, model=model, optimizer=False, grad=grad)
        if not optimizer:
            return
        if device == "cpu":
            for role_optimizer in self.optimizers.values():
                offload_fsdp_optimizer(role_optimizer)
        else:
            for role_optimizer in self.optimizers.values():
                load_fsdp_optimizer(role_optimizer, device)

    def backward_role(self, role: str, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        """Backpropagate a role loss with the matching adapter active."""
        if loss.ndim != 0:
            raise ValueError(f"Role loss must be scalar, got shape {tuple(loss.shape)} for {role!r}.")
        with self.use_role(role):
            loss.backward(retain_graph=retain_graph)

    def assert_gradient_isolation(self, active_roles: set[str]) -> None:
        """Reject gradients on optimizer-owned parameters outside the active phase."""
        leaked_roles = []
        for role, parameters in self._role_parameters.items():
            if role in active_roles:
                continue
            if any(parameter.grad is not None for parameter in parameters):
                leaked_roles.append(role)
        if leaked_roles:
            raise RuntimeError(f"Gradient leaked into inactive distillation roles: {sorted(leaked_roles)}.")

    def optimizer_step(self, role: Optional[str] = None) -> tuple[bool, float]:
        """Clip and step one role, advancing its scheduler only on finite gradients."""
        role = role or self._active_role
        if role is None or role not in self.optimizers:
            raise ValueError(f"No trainable active role for optimizer step: {role!r}.")
        self.activate_role(role)
        grad_norm = super().optimizer_step()
        stepped = math.isfinite(grad_norm)
        if stepped:
            self.lr_schedulers[role].step()
        self.optimizers[role].zero_grad()
        return stepped, grad_norm

    def lr_scheduler_step(self, role: Optional[str] = None) -> float:
        """Advance one role scheduler explicitly."""
        role = role or self._active_role
        if role is None or role not in self.lr_schedulers:
            raise ValueError(f"No scheduler for role {role!r}.")
        self.lr_schedulers[role].step()
        return self.lr_schedulers[role].get_last_lr()[0]

    def update_role_ema(self, source_role: str, target_role: str, decay: float) -> None:
        """EMA-update two adapter roles in the same physical group."""
        source = self.role_bindings[source_role]
        target = self.role_bindings[target_role]
        if not source.adapter or not target.adapter:
            raise ValueError("In-group EMA requires named source and target adapters.")
        self.ema_update_adapter(source=source.adapter, target=target.adapter, decay=decay)

    def update_module_ema_from(self, source: DistillationRoleGroupEngine, decay: float) -> None:
        """EMA-update this independent module from an identically sharded source."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}.")
        source_adapter = (
            next((binding.adapter for binding in source.role_bindings.values() if binding.adapter is not None), None)
            if source.has_adapters()
            else None
        )
        target_adapter = (
            next((binding.adapter for binding in self.role_bindings.values() if binding.adapter is not None), None)
            if self.has_adapters()
            else None
        )
        if (source_adapter is None) != (target_adapter is None):
            raise ValueError("Independent EMA source and target must both use LoRA adapters or both use full modules.")

        if source_adapter is not None:
            with source._adapter_state_context(), self._adapter_state_context(), torch.no_grad():
                source_parameters = source._active_adapter_trainable_params(source_adapter)
                target_parameters = self._active_adapter_trainable_params(target_adapter)
                self.ema_parameter_lists(source_parameters, target_parameters, decay)
            return

        source_parameters = tuple(source.module.named_parameters())
        target_parameters = tuple(self.module.named_parameters())
        if len(source_parameters) != len(target_parameters):
            raise ValueError("Independent EMA source and target parameter counts do not match.")
        with torch.no_grad():
            for (source_name, source_parameter), (target_name, target_parameter) in zip(
                source_parameters, target_parameters, strict=True
            ):
                if source_name != target_name:
                    raise ValueError(
                        f"Independent EMA parameter names do not match: {source_name!r} and {target_name!r}."
                    )
                if source_parameter.shape != target_parameter.shape:
                    raise ValueError("Independent EMA source and target parameter shapes do not match.")
                target_parameter.lerp_(source_parameter, 1.0 - decay)

    @staticmethod
    def ema_parameter_lists(source_parameters, target_parameters, decay: float) -> None:
        """Blend corresponding independent-module adapter parameters in place."""
        if len(source_parameters) != len(target_parameters) or not source_parameters:
            raise ValueError("Independent EMA source and target adapter parameter counts must match and be non-empty.")
        for source_parameter, target_parameter in zip(source_parameters, target_parameters, strict=True):
            if source_parameter.shape != target_parameter.shape:
                raise ValueError("Independent EMA source and target parameter shapes do not match.")
            target_parameter.lerp_(source_parameter, 1.0 - decay)

    def iter_export_tensors(self, role: str, base_sync_done: bool):
        """Return the selected student role's parameter iterator and matching PEFT config."""
        if role not in {"student", "student_ema"}:
            raise ValueError(f"Only student or student_ema can be exported, got {role!r}.")
        binding = self.role_bindings[role]
        return self.get_per_tensor_param(
            base_sync_done=base_sync_done,
            adapter_name=binding.adapter,
        )

    def additional_state_path(self, local_path: str) -> str:
        """Locate this rank's secondary optimizer and scheduler state."""
        return os.path.join(local_path, f"role_state_rank_{self.rank}.pt")

    def save_role_group_checkpoint(self, local_path: str, global_step: int) -> None:
        """Save one physical model once plus every role optimizer/scheduler."""
        immutable_teacher_only = set(self.role_bindings) == {"teacher_score"}
        if not immutable_teacher_only:
            if self._primary_role is not None:
                self.activate_role(self._primary_role)
            super().save_checkpoint(local_path=local_path, global_step=global_step)

        os.makedirs(local_path, exist_ok=True)
        additional_roles = [role for role in self.optimizers if role != self._primary_role]
        torch.save(
            {
                "optimizers": {role: self.optimizers[role].state_dict() for role in additional_roles},
                "schedulers": {role: self.lr_schedulers[role].state_dict() for role in additional_roles},
                "primary_role": self._primary_role,
            },
            self.additional_state_path(local_path),
        )
        if self.rank == 0:
            with open(os.path.join(local_path, "role_group.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "group": asdict(self.role_group),
                        "bindings": [asdict(binding) for binding in self.role_bindings.values()],
                        "immutable_teacher_only": immutable_teacher_only,
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )
        torch.distributed.barrier()

    def load_role_group_checkpoint(self, local_path: str) -> None:
        """Restore the model and every role optimizer/scheduler."""
        manifest_path = os.path.join(local_path, "role_group.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Missing role-group manifest: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        expected_bindings = [asdict(binding) for binding in self.role_bindings.values()]
        if manifest.get("group") != asdict(self.role_group) or manifest.get("bindings") != expected_bindings:
            raise ValueError(f"Checkpoint role layout does not match runtime group {self.role_group.name!r}.")

        if not manifest.get("immutable_teacher_only", False):
            if self._primary_role is not None:
                self.activate_role(self._primary_role)
            super().load_checkpoint(local_path=local_path, del_local_after_load=False)

        role_state = torch.load(self.additional_state_path(local_path), map_location="cpu", weights_only=False)
        if role_state.get("primary_role") != self._primary_role:
            raise ValueError(
                f"Checkpoint primary role {role_state.get('primary_role')!r} does not match {self._primary_role!r}."
            )
        expected_additional_roles = set(self.optimizers) - ({self._primary_role} if self._primary_role else set())
        if set(role_state.get("optimizers", {})) != expected_additional_roles:
            raise ValueError("Checkpoint secondary optimizer roles do not match the active role group.")
        if set(role_state.get("schedulers", {})) != expected_additional_roles:
            raise ValueError("Checkpoint secondary scheduler roles do not match the active role group.")
        for role, state in role_state["optimizers"].items():
            self.optimizers[role].load_state_dict(state)
        for role, state in role_state["schedulers"].items():
            self.lr_schedulers[role].load_state_dict(state)
        torch.distributed.barrier()

    def forward_backward_batch(self, data: TensorDict, loss_function, forward_only: bool = False):
        """Reject PPO-shaped execution; distillation phases use the phase runner."""
        raise NotImplementedError("DistillationRoleGroupEngine is driven through DiffusionDistillationWorker phases.")

    def prepare_model_inputs(self, micro_batch: TensorDict, step: int):
        """Keep model-specific preparation in the architecture phase runner."""
        raise NotImplementedError("Architecture-owned distillation phase runners prepare model inputs.")

    def prepare_model_outputs(self, output, micro_batch: TensorDict):
        """Keep model-specific output conversion in the phase runner."""
        raise NotImplementedError("Architecture-owned distillation phase runners prepare model outputs.")

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only, step):
        """Reject the PPO step interface for multi-role computation."""
        raise NotImplementedError("Architecture-owned distillation phase runners execute forwards.")
