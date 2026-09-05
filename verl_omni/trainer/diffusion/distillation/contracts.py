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
"""Immutable execution contracts for distribution-matching distillation.

These architecture-neutral types make up a validated
:class:`~verl_omni.trainer.diffusion.distillation.recipes.DistillationPlan`.
They carry no Ray, model-pipeline, or FSDP dependency. The generic equations in
``equations.py`` operate on unpacked tensors; :class:`LatentBundle` is only a
transport container used across role boundaries.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from torch import Tensor

__all__ = [
    "FrozenDict",
    "LatentBundle",
    "StudentRollout",
    "ScoreBatch",
    "ConditionBundle",
    "CanonicalPrediction",
    "RoleGroupSpec",
    "RoleBinding",
    "ScoreTransportSpec",
    "ExportSpec",
    "RoleLayoutSpec",
    "DataRequirements",
    "ObjectiveSpec",
    "RolloutSpec",
    "InitializationSpec",
    "UpdatePhaseSpec",
    "PhaseRequest",
    "PhaseResult",
    "UpdateCycle",
    "UpdateSchedule",
    "TrainerCounters",
    "DistillationPlan",
    "validate_role_layout",
    "validate_export_role",
    "validate_distillation_plan",
    "describe_role_groups",
    "resolve_export_role",
    "EXPORTABLE_ROLES",
    "TeacherScoreProvider",
    "RoleCheckpointManifest",
    "DistillationCheckpointState",
]


class FrozenDict(Mapping[str, Any]):
    """Small recursively immutable, pickle-friendly mapping for plan specs."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        values = values or {}
        self._data = {key: freeze_value(value) for key, value in values.items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._data.items())))

    def __reduce__(self):
        return FrozenDict, (self._data,)


def freeze_value(value: Any) -> Any:
    """Recursively freeze plan mappings and collections."""
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, list | tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(freeze_value(item) for item in value)
    return value


@dataclass
class LatentBundle:
    """Transport container for architecture-native per-modality tensors."""

    tensors: dict[str, Tensor]

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ValueError("LatentBundle must contain at least one modality tensor.")
        for key, value in self.tensors.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"LatentBundle[{key!r}] must be a torch.Tensor, got {type(value)}.")

    def __len__(self) -> int:
        return len(self.tensors)

    def get(self, modality: str) -> Tensor:
        return self.tensors[modality]

    @property
    def single(self) -> Tensor:
        """Return the single tensor of a one-modality bundle."""
        if len(self.tensors) != 1:
            raise ValueError(f"single only valid for a one-modality bundle, got {sorted(self.tensors)}")
        return next(iter(self.tensors.values()))

    def map(self, fn):
        """Apply ``fn`` to every tensor and return a new bundle."""
        return LatentBundle({key: fn(value) for key, value in self.tensors.items()})


@dataclass
class StudentRollout:
    """Output of a student rollout pass, consumed by score-model phases."""

    generated_x0: LatentBundle
    initial_noise: LatentBundle
    selected_step_indices: Tensor
    denoised_sigma_from: Optional[Tensor] = None
    denoised_sigma_to: Optional[Tensor] = None
    gradient_mask: Optional[LatentBundle] = None
    committed_context_length: Optional[Tensor] = None


@dataclass
class ScoreBatch:
    """Generated samples and conditioning scored by real/fake score models."""

    generated_x0: LatentBundle
    generated_x0_detached: LatentBundle
    noisy_latents: LatentBundle
    noise: LatentBundle
    sigma: Tensor
    condition: ConditionBundle
    negative_condition: Optional[ConditionBundle] = None


@dataclass
class CanonicalPrediction:
    """Teacher or fake-score output converted to canonical fp32 ``x0``."""

    x0: LatentBundle
    raw: Optional[LatentBundle] = None


@dataclass
class ConditionBundle:
    """Conditioning tensors, masks, and metadata shared by all roles."""

    tensors: dict[str, Tensor]
    masks: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleGroupSpec:
    """A physical model group owning one wrapped model."""

    name: str
    model_ref: str = ""
    storage: Literal["independent_module", "shared_base_adapters"] = "shared_base_adapters"
    placement: Literal["colocated", "standalone"] = "colocated"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RoleGroupSpec.name must not be empty.")
        valid_storage = {"independent_module", "shared_base_adapters"}
        if self.storage not in valid_storage:
            raise ValueError(f"Invalid role-group storage {self.storage!r}; expected one of {sorted(valid_storage)}.")
        valid_placement = {"colocated", "standalone"}
        if self.placement not in valid_placement:
            raise ValueError(
                f"Invalid role-group placement {self.placement!r}; expected one of {sorted(valid_placement)}."
            )


@dataclass(frozen=True)
class RoleBinding:
    """A logical algorithm role bound to a group and optional named adapter."""

    role: str
    group: str
    adapter: Optional[str] = None
    trainable: bool = False
    optimizer_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.role or not self.group:
            raise ValueError("RoleBinding.role and RoleBinding.group must not be empty.")


@dataclass(frozen=True)
class ScoreTransportSpec:
    """How teacher/fake scores are transported."""

    provider: Literal["colocated", "ray"] = "colocated"
    tensor_backend: Literal["local", "ray_nixl", "mooncake"] = "local"

    def __post_init__(self) -> None:
        valid_providers = {"colocated", "ray"}
        if self.provider not in valid_providers:
            raise ValueError(f"Invalid score provider {self.provider!r}; expected one of {sorted(valid_providers)}.")
        valid_backends = {"local", "ray_nixl", "mooncake"}
        if self.tensor_backend not in valid_backends:
            raise ValueError(
                f"Invalid score tensor backend {self.tensor_backend!r}; expected one of {sorted(valid_backends)}."
            )
        if self.provider == "colocated" and self.tensor_backend != "local":
            raise ValueError("A colocated score provider requires tensor_backend='local'.")
        if self.provider == "ray" and self.tensor_backend == "local":
            raise ValueError("A Ray score provider requires tensor_backend='ray_nixl' or 'mooncake'.")


@dataclass(frozen=True)
class ExportSpec:
    """Which semantic role is exported and through which backend."""

    role: Literal["student", "student_ema"] = "student_ema"
    checkpoint_engine_backend: str = "naive"

    def __post_init__(self) -> None:
        if self.role not in EXPORTABLE_ROLES:
            raise ValueError(f"Export role must be one of {EXPORTABLE_ROLES}, got {self.role!r}.")
        if not self.checkpoint_engine_backend:
            raise ValueError("checkpoint_engine_backend must not be empty.")


@dataclass(frozen=True)
class RoleLayoutSpec:
    """Validated physical groups, logical bindings, and score transport."""

    groups: tuple[RoleGroupSpec, ...]
    bindings: tuple[RoleBinding, ...]
    score_transport: ScoreTransportSpec = ScoreTransportSpec()


DataRequirements = Mapping[str, Any]
ObjectiveSpec = Mapping[str, Any]
RolloutSpec = Mapping[str, Any]
InitializationSpec = Mapping[str, Any]


@dataclass(frozen=True)
class UpdatePhaseSpec:
    """A static phase specification expanded by :class:`UpdateSchedule`."""

    kind: Literal["student", "fake_score"]
    repeats: int = 1
    batch_policy: Literal["fresh", "reuse_student"] = "fresh"
    trainable_roles: tuple[str, ...] = ()
    update_ema: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"student", "fake_score"}:
            raise ValueError(f"Invalid phase kind {self.kind!r}; expected 'student' or 'fake_score'.")
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int) or self.repeats <= 0:
            raise ValueError(f"Phase repeats must be an integer greater than zero, got {self.repeats}.")
        if self.batch_policy not in {"fresh", "reuse_student"}:
            raise ValueError(f"Invalid batch_policy {self.batch_policy!r}; expected 'fresh' or 'reuse_student'.")
        if not self.trainable_roles:
            raise ValueError(f"{self.kind!r} phase must declare at least one trainable role.")
        if len(set(self.trainable_roles)) != len(self.trainable_roles):
            raise ValueError(f"{self.kind!r} phase contains duplicate trainable roles.")
        if self.kind == "student" and self.trainable_roles != ("student",):
            raise ValueError("A student phase must train exactly the 'student' role.")
        if self.kind == "fake_score" and "student" in self.trainable_roles:
            raise ValueError("A fake_score phase must not train the student role.")
        if self.update_ema and self.kind != "student":
            raise ValueError("EMA updates are only valid on student phases.")


@dataclass(frozen=True)
class PhaseRequest:
    """A concrete phase request emitted by :meth:`UpdateSchedule.next_cycle`."""

    kind: Literal["student", "fake_score"]
    global_step: int
    repeat_index: int
    batch_policy: Literal["fresh", "reuse_student"]
    trainable_roles: tuple[str, ...]
    update_ema: bool = False


@dataclass
class PhaseResult:
    """The phase-specific state returned from an executor to the driver."""

    metrics: dict[str, float] = field(default_factory=dict)
    optimizer_steps: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateCycle:
    """A sequence of phase requests produced for one cycle."""

    requests: tuple[PhaseRequest, ...]
    requires_student_update: bool
    is_warmup: bool = False


@dataclass
class TrainerCounters:
    """Driver counters; ``global_step`` counts completed student updates only."""

    global_step: int = 0
    optimizer_steps: dict[str, int] = field(default_factory=dict)
    completed_cycles: int = 0

    def increment_global(self) -> None:
        self.global_step += 1

    def record_step(self, role: str) -> None:
        self.optimizer_steps[role] = self.optimizer_steps.get(role, 0) + 1


@dataclass(frozen=True)
class UpdateSchedule:
    """Normal phases plus an optional finite fake/discriminator warmup."""

    phases: tuple[UpdatePhaseSpec, ...]
    warmup_phases: tuple[UpdatePhaseSpec, ...] = ()
    warmup_cycles: int = 0

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("UpdateSchedule must contain normal-cycle phases.")
        student_indices = [index for index, phase in enumerate(self.phases) if phase.kind == "student"]
        if student_indices != [0] or self.phases[0].repeats != 1:
            raise ValueError(
                "A normal cycle must contain exactly one student phase with repeats=1 and it must be first."
            )
        if isinstance(self.warmup_cycles, bool) or not isinstance(self.warmup_cycles, int) or self.warmup_cycles < 0:
            raise ValueError(f"warmup_cycles must be a non-negative integer, got {self.warmup_cycles}.")
        if self.warmup_cycles > 0 and not self.warmup_phases:
            raise ValueError("warmup_cycles > 0 requires at least one warmup phase.")
        if self.warmup_cycles == 0 and self.warmup_phases:
            raise ValueError("warmup_phases require warmup_cycles > 0.")
        if any(phase.kind == "student" for phase in self.warmup_phases):
            raise ValueError("Warmup phases must not contain a student phase.")
        if any(phase.batch_policy == "reuse_student" for phase in self.warmup_phases):
            raise ValueError("Warmup phases cannot reuse a student batch before any student phase has run.")

    def next_cycle(self, counters: TrainerCounters) -> UpdateCycle:
        """Expand either the next warmup cycle or the normal static phases."""
        is_warmup = counters.completed_cycles < self.warmup_cycles
        phases = self.warmup_phases if is_warmup else self.phases
        requests = tuple(
            PhaseRequest(
                kind=phase.kind,
                global_step=counters.global_step,
                repeat_index=repeat_index,
                batch_policy=phase.batch_policy,
                trainable_roles=phase.trainable_roles,
                update_ema=phase.update_ema,
            )
            for phase in phases
            for repeat_index in range(phase.repeats)
        )
        return UpdateCycle(
            requests=requests,
            requires_student_update=not is_warmup,
            is_warmup=is_warmup,
        )


@dataclass(frozen=True)
class DistillationPlan:
    """An immutable, validated plan describing one named recipe."""

    name: str
    version: int
    role_layout: RoleLayoutSpec
    data_requirements: DataRequirements
    objective: ObjectiveSpec
    rollout: RolloutSpec
    initialization: InitializationSpec
    update_schedule: UpdateSchedule
    export: ExportSpec
    required_capabilities: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_requirements", FrozenDict(self.data_requirements))
        object.__setattr__(self, "objective", FrozenDict(self.objective))
        object.__setattr__(self, "rollout", FrozenDict(self.rollout))
        object.__setattr__(self, "initialization", FrozenDict(self.initialization))
        object.__setattr__(self, "required_capabilities", frozenset(self.required_capabilities))
        validate_distillation_plan(self)


EXPORTABLE_ROLES = ("student", "student_ema")


def validate_role_layout(layout: RoleLayoutSpec) -> None:
    """Fail closed on invalid group/binding ownership before model allocation."""
    group_names = [group.name for group in layout.groups]
    if not group_names:
        raise ValueError("RoleLayoutSpec must contain at least one role group.")
    duplicate_groups = {name for name in group_names if group_names.count(name) > 1}
    if duplicate_groups:
        raise ValueError(f"Duplicate role-group names: {sorted(duplicate_groups)}.")
    if not layout.bindings:
        raise ValueError("RoleLayoutSpec must contain at least one role binding.")

    binding_roles: set[str] = set()
    optimizer_keys: set[str] = set()
    groups = {group.name: group for group in layout.groups}
    for binding in layout.bindings:
        if binding.group not in groups:
            raise ValueError(f"RoleBinding {binding.role!r} references unknown group {binding.group!r}.")
        if binding.role in binding_roles:
            raise ValueError(f"Duplicate role binding for {binding.role!r}.")
        binding_roles.add(binding.role)
        if binding.trainable and not binding.optimizer_key:
            raise ValueError(f"Trainable role {binding.role!r} must set an optimizer_key.")
        if not binding.trainable and binding.optimizer_key is not None:
            raise ValueError(f"Frozen role {binding.role!r} must not set an optimizer_key.")
        if binding.trainable and binding.optimizer_key in optimizer_keys:
            raise ValueError(f"Duplicate optimizer_key {binding.optimizer_key!r}.")
        if binding.optimizer_key is not None:
            optimizer_keys.add(binding.optimizer_key)
        if groups[binding.group].storage == "shared_base_adapters" and binding.trainable and not binding.adapter:
            raise ValueError(f"Trainable role {binding.role!r} in a shared-base group must name an adapter.")

    for group in layout.groups:
        if group.storage != "shared_base_adapters":
            continue
        adapters = [binding.adapter for binding in layout.bindings if binding.group == group.name and binding.adapter]
        duplicates = {adapter for adapter in adapters if adapters.count(adapter) > 1}
        if duplicates:
            raise ValueError(f"Shared-base group {group.name!r} has duplicate adapter names {sorted(duplicates)}.")


def validate_export_role(export: ExportSpec, binding_roles: set[str]) -> None:
    """Ensure the export role is semantic, exportable, and bound."""
    if export.role not in EXPORTABLE_ROLES:
        raise ValueError(f"Export role must be one of {EXPORTABLE_ROLES}, got {export.role!r}.")
    if export.role not in binding_roles:
        raise ValueError(f"Export role {export.role!r} is not bound. Bound roles: {sorted(binding_roles)}.")


def validate_distillation_plan(plan: DistillationPlan) -> None:
    """Validate architecture-neutral plan invariants before execution."""
    if not plan.name:
        raise ValueError("DistillationPlan.name must not be empty.")
    if plan.version <= 0:
        raise ValueError(f"DistillationPlan.version must be positive, got {plan.version}.")
    validate_role_layout(plan.role_layout)
    missing_model_refs = [group.name for group in plan.role_layout.groups if not group.model_ref]
    if missing_model_refs:
        raise ValueError(f"Every role group must define model_ref; missing for {sorted(missing_model_refs)}.")

    binding_roles = {binding.role for binding in plan.role_layout.bindings}
    required_roles = {"student", "student_ema", "teacher_score", "fake_score"}
    missing_roles = required_roles - binding_roles
    if missing_roles:
        raise ValueError(f"DistillationPlan is missing required role bindings: {sorted(missing_roles)}.")
    validate_export_role(plan.export, binding_roles)

    trainable_roles = {binding.role for binding in plan.role_layout.bindings if binding.trainable}
    all_phases = plan.update_schedule.phases + plan.update_schedule.warmup_phases
    for phase in all_phases:
        unknown_roles = set(phase.trainable_roles) - trainable_roles
        if unknown_roles:
            raise ValueError(
                f"Phase {phase.kind!r} references roles that are not trainable bindings: {sorted(unknown_roles)}."
            )
        if phase.update_ema and "student_ema" not in binding_roles:
            raise ValueError("A phase requesting EMA requires a bound student_ema role.")

    required_spec_keys = (
        ("data_requirements", plan.data_requirements, "mode"),
        ("objective", plan.objective, "name"),
        ("rollout", plan.rollout, "strategy"),
        ("initialization", plan.initialization, "stage"),
    )
    for spec_name, spec, required_key in required_spec_keys:
        if not spec.get(required_key):
            raise ValueError(f"DistillationPlan.{spec_name} must define non-empty {required_key!r}.")


def describe_role_groups(layout: RoleLayoutSpec) -> dict[str, str]:
    """Return concise role-group descriptions for logging."""
    return {group.name: f"({group.storage}, {group.placement})" for group in layout.groups}


def resolve_export_role(export: ExportSpec) -> str:
    """Return the semantic export role after validation."""
    if export.role not in EXPORTABLE_ROLES:
        raise ValueError(f"Export role must be one of {EXPORTABLE_ROLES}, got {export.role!r}.")
    return export.role


@runtime_checkable
class TeacherScoreProvider(Protocol):
    """Provides canonical ``x0`` teacher predictions for a score batch."""

    def predict_x0(self, score_batch: ScoreBatch) -> CanonicalPrediction:
        """Return the teacher's canonical fp32 ``x0`` for ``score_batch``."""
        ...


@dataclass
class RoleCheckpointManifest:
    """Metadata describing one role's stored state."""

    role: str
    model_path: str = ""
    model_revision: str = ""
    config_hash: str = ""
    optimizer_key: str = ""


@dataclass
class DistillationCheckpointState:
    """Composite multi-role checkpoint state, restored atomically."""

    global_step: int = 0
    completed_cycles: int = 0
    role_manifests: list[RoleCheckpointManifest] = field(default_factory=list)
    rng: dict[str, Any] = field(default_factory=dict)
