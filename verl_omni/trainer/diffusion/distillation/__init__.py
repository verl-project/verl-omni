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
"""Distribution-matching distillation runtime (DMD, DMD2, CausVid, Self-Forcing).

PR 1 provides the architecture-neutral trainer control plane, the immutable
execution contracts, the recipe/objective/rollout registries, and pure DMD
equations. It defines no model pipeline, Ray worker, FSDP model, or GPU runtime.
"""

from verl_omni.trainer.diffusion.distillation import contracts, control_plane, equations, ray_trainer, recipes
from verl_omni.trainer.diffusion.distillation.contracts import (
    CanonicalPrediction,
    ConditionBundle,
    DistillationCheckpointState,
    DistillationPlan,
    ExportSpec,
    FrozenDict,
    LatentBundle,
    PhaseRequest,
    PhaseResult,
    RoleBinding,
    RoleCheckpointManifest,
    RoleGroupSpec,
    RoleLayoutSpec,
    ScoreBatch,
    ScoreTransportSpec,
    StudentRollout,
    TeacherScoreProvider,
    TrainerCounters,
    UpdateCycle,
    UpdatePhaseSpec,
    UpdateSchedule,
    describe_role_groups,
    resolve_export_role,
    validate_distillation_plan,
    validate_export_role,
    validate_role_layout,
)
from verl_omni.trainer.diffusion.distillation.control_plane import (
    BatchProvider,
    DistillationPhaseExecutor,
    DistillationTrainerControlPlane,
    DistillationTrainerHooks,
    FakeBatchProvider,
    FakeDistillationHooks,
    FakePhaseExecutor,
)
from verl_omni.trainer.diffusion.distillation.ray_trainer import DistillationRayTrainer
from verl_omni.trainer.diffusion.distillation.recipes import build_plan, build_plan_from_config, recipe_registry

__all__ = [
    # submodules
    "contracts",
    "equations",
    "recipes",
    "control_plane",
    "ray_trainer",
    # contracts
    "FrozenDict",
    "LatentBundle",
    "StudentRollout",
    "ScoreBatch",
    "ConditionBundle",
    "CanonicalPrediction",
    "RoleGroupSpec",
    "RoleBinding",
    "RoleLayoutSpec",
    "ScoreTransportSpec",
    "ExportSpec",
    "UpdatePhaseSpec",
    "UpdateSchedule",
    "UpdateCycle",
    "PhaseRequest",
    "PhaseResult",
    "TrainerCounters",
    "DistillationPlan",
    # role layout validation / export / provider / checkpoint
    "validate_role_layout",
    "validate_export_role",
    "validate_distillation_plan",
    "describe_role_groups",
    "resolve_export_role",
    "TeacherScoreProvider",
    "RoleCheckpointManifest",
    "DistillationCheckpointState",
    # control plane / executor
    "DistillationTrainerControlPlane",
    "BatchProvider",
    "DistillationTrainerHooks",
    "DistillationPhaseExecutor",
    "FakePhaseExecutor",
    "FakeBatchProvider",
    "FakeDistillationHooks",
    # driver
    "DistillationRayTrainer",
    # recipes
    "build_plan",
    "build_plan_from_config",
    "recipe_registry",
]
