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
    """Frozen diffusion distillation teacher."""

    # Local path to the teacher checkpoint, a full pipeline checkpoint from the
    # same pipeline family as the student.
    model_path: Optional[str] = None


@dataclass
class DiffusionDistillationConfig(BaseConfig):
    """Diffusion on-policy distillation."""

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models"}

    # Whether on-policy distillation is enabled.
    enabled: bool = False

    # Teacher entries. Only the single `teacher_model` entry is supported.
    teacher_models: dict[str, DiffusionDistillationTeacherModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.enabled:
            return
        if set(self.teacher_models) != {"teacher_model"}:
            raise NotImplementedError(
                f"Diffusion distillation supports a single teacher configured under "
                f"distillation.teacher_models.teacher_model, but got entries {sorted(self.teacher_models)}."
            )
        from verl.utils.config import omega_conf_to_dataclass

        teacher_model = omega_conf_to_dataclass(
            self.teacher_models["teacher_model"], dataclass_type=DiffusionDistillationTeacherModelConfig
        )
        if teacher_model.model_path is None:
            raise ValueError("model_path must be specified for the distillation teacher model.")
        self.teacher_models["teacher_model"] = teacher_model
