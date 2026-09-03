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

__all__ = ["DiffusionDistillationTeacherModelConfig", "DiffusionDistillationConfig"]


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
class DiffusionDistillationConfig(BaseConfig):
    """Diffusion on-policy distillation.

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

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DiffusionDistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"

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
