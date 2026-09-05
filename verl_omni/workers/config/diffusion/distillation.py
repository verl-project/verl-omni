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

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.workers.config import FSDPOptimizerConfig


def default_fake_score_optimizer() -> FSDPOptimizerConfig:
    return FSDPOptimizerConfig(lr=2e-5, weight_decay=0.01, clip_grad=1.0, lr_scheduler_type="constant")


__all__ = [
    "DiffusionDistillationTeacherModelConfig",
    "DiffusionDistributionMatchingConfig",
    "DiffusionDistillationConfig",
]


@dataclass
class DiffusionDistillationTeacherModelConfig(BaseConfig):
    """Frozen diffusion distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Local path to the teacher checkpoint, a full pipeline checkpoint from the
        same pipeline family as the student.
    world_size (int):
        Number of GPUs this teacher occupies in the distillation resource pool.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"key", "world_size"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    world_size: int = 0

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")


@dataclass
class DiffusionDistributionMatchingConfig(BaseConfig):
    """Architecture-neutral DMD-family recipe selection.

    This config is active only when ``algorithm.trainer_type=distillation``.
    The existing parent ``enabled`` flag remains exclusively owned by on-policy
    distillation and must stay false for DMD-family training.
    """

    # Registered recipe name.
    recipe: str = "dmd2"
    # Optional recipe profile; null selects the recipe default.
    profile: Optional[str] = None
    # Optional fake-score phase count; null selects the recipe default.
    fake_update_ratio: Optional[int] = None
    # Number of fake/discriminator-only cycles before student updates begin.
    fake_warmup_cycles: int = 0
    # Optional registered rollout override; null selects the recipe default.
    rollout_strategy: Optional[str] = None
    # Optional data-mode override; null selects the recipe default.
    data_mode: Optional[str] = None
    # Semantic role exported to inference replicas.
    export_role: str = "student_ema"
    # Physical storage used by the initial colocated runtime.
    role_storage: str = "shared_base_adapters"
    # Per-device student phase micro-batch size.
    student_micro_batch_size_per_gpu: int = 1
    # Per-device fake-score phase micro-batch size.
    fake_score_micro_batch_size_per_gpu: int = 1
    # Independent fake-score optimizer and scheduler configuration.
    fake_score_optim: FSDPOptimizerConfig = field(default_factory=default_fake_score_optimizer)
    # EMA decay applied after successful student optimizer steps.
    ema_decay: float = 0.999
    # First completed student step that updates EMA.
    ema_start_step: int = 0

    def __post_init__(self):
        valid_recipes = {"dmd", "dmd2", "causvid", "self_forcing"}
        if self.recipe not in valid_recipes:
            raise ValueError(f"Invalid recipe: {self.recipe}. Must be one of {sorted(valid_recipes)}")
        valid_profiles = {"distribution_only", "paper"}
        if self.profile is not None and self.profile not in valid_profiles:
            raise ValueError(f"Invalid profile: {self.profile}. Must be one of {sorted(valid_profiles)}")
        if self.fake_update_ratio is not None and self.fake_update_ratio <= 0:
            raise ValueError(f"fake_update_ratio must be greater than 0, got {self.fake_update_ratio}")
        if self.fake_warmup_cycles < 0:
            raise ValueError(f"fake_warmup_cycles must be non-negative, got {self.fake_warmup_cycles}")
        valid_rollout_strategies = {
            "backward_simulated",
            "consistency_renoise",
            "ode_euler",
            "one_step",
            "self_forced",
            "teacher_forced_causal",
        }
        if self.rollout_strategy is not None and self.rollout_strategy not in valid_rollout_strategies:
            raise ValueError(
                f"Invalid rollout_strategy: {self.rollout_strategy}. Must be one of {sorted(valid_rollout_strategies)}"
            )
        valid_data_modes = {"prompts", "prompt_and_real_latent", "regression_pairs"}
        if self.data_mode is not None and self.data_mode not in valid_data_modes:
            raise ValueError(f"Invalid data_mode: {self.data_mode}. Must be one of {sorted(valid_data_modes)}")
        valid_export_roles = {"student", "student_ema"}
        if self.export_role not in valid_export_roles:
            raise ValueError(f"Invalid export_role: {self.export_role}. Must be one of {sorted(valid_export_roles)}")
        valid_role_storage = {"shared_base_adapters", "colocated_independent"}
        if self.role_storage not in valid_role_storage:
            raise ValueError(f"Invalid role_storage: {self.role_storage}. Must be one of {sorted(valid_role_storage)}")
        if self.student_micro_batch_size_per_gpu <= 0:
            raise ValueError(
                f"student_micro_batch_size_per_gpu must be greater than 0, got {self.student_micro_batch_size_per_gpu}"
            )
        if self.fake_score_micro_batch_size_per_gpu <= 0:
            raise ValueError(
                "fake_score_micro_batch_size_per_gpu must be greater than 0, "
                f"got {self.fake_score_micro_batch_size_per_gpu}"
            )
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {self.ema_decay}")
        if self.ema_start_step < 0:
            raise ValueError(f"ema_start_step must be non-negative, got {self.ema_start_step}")


@dataclass
class DiffusionDistillationConfig(BaseConfig):
    """Diffusion distillation settings shared by OPD and DMD-family routing.

    ``enabled`` and the teacher-pool fields remain exclusive to OPD. DMD-family
    training is selected by ``algorithm.trainer_type=distillation`` and reads the
    nested ``distribution_matching`` config while keeping ``enabled=false``.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool. 0 colocates the teachers with the actor.
    teacher_models (dict[str, DiffusionDistillationTeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=ocr
    distillation.teacher_models.teacher_model.model_path=/ckpt/ocr_teacher
    +distillation.teacher_models.teacher_model2.key=aesthetic
    +distillation.teacher_models.teacher_model2.model_path=/ckpt/aesthetic_teacher
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=ocr
    +distillation.teacher_models.teacher_model1.model_path=/ckpt/ocr_teacher
    +distillation.teacher_models.teacher_model2.key=aesthetic
    +distillation.teacher_models.teacher_model2.model_path=/ckpt/aesthetic_teacher
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "distribution_matching"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DiffusionDistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    # DMD-family recipe settings; selected by algorithm.trainer_type rather than enabled.
    distribution_matching: DiffusionDistributionMatchingConfig = field(
        default_factory=DiffusionDistributionMatchingConfig
    )

    def __post_init__(self):
        if not self.enabled:
            return

        self.teacher_models = self._resolve_teacher_models()
        if self.nnodes > 0:
            teacher_world_size_sum = sum(teacher_model.world_size for teacher_model in self.teacher_models.values())
            total_pool_size = self.n_gpus_per_node * self.nnodes
            if teacher_world_size_sum != total_pool_size:
                raise ValueError(
                    f"Sum of teacher world_size ({teacher_world_size_sum}) must match "
                    f"the distillation resource pool size "
                    f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
                )

    def _resolve_teacher_models(self) -> dict[str, DiffusionDistillationTeacherModelConfig]:
        from verl.utils.config import omega_conf_to_dataclass

        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            teacher_model.world_size = self.n_gpus_per_node * self.nnodes
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(
                teacher_config, dataclass_type=DiffusionDistillationTeacherModelConfig
            )
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
