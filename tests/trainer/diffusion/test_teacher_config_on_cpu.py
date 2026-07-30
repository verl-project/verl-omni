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
"""CPU tests for the diffusion OPD teacher config group (RFC #293)."""

import os

import pytest
from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.workers.config.diffusion import DiffusionTeacherConfig, TeacherModelEntry

CONFIG_DIR = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config")

ENABLED = "actor_rollout_ref.teacher.enabled=true"
ONE_TEACHER = "+actor_rollout_ref.teacher.models.default.model.path=/ckpt/teacher"


def compose_teacher(*overrides: str) -> DiffusionTeacherConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="diffusion_trainer", overrides=list(overrides))
    return omega_conf_to_dataclass(cfg.actor_rollout_ref.teacher, DiffusionTeacherConfig)


class TestTeacherConfigGroup:
    def test_default_disabled_composes(self):
        teacher = compose_teacher()

        assert isinstance(teacher, DiffusionTeacherConfig)
        assert teacher.enabled is False
        assert teacher.models == {}
        assert teacher.placement.mode == "colocated"
        assert teacher.placement.n_gpus_per_node is None
        assert teacher.placement.nnodes is None

    def test_disabled_teacher_skips_validation(self):
        """Default-off must be inert: a stale placement never breaks a non-teacher run."""
        teacher = compose_teacher("actor_rollout_ref.teacher.placement.mode=standalone")

        assert teacher.enabled is False

    def test_single_teacher_entry_is_typed(self):
        teacher = compose_teacher(ENABLED, ONE_TEACHER)

        assert list(teacher.models) == ["default"]
        entry = teacher.models["default"]
        assert isinstance(entry, TeacherModelEntry)
        assert entry.model.path == "/ckpt/teacher"
        assert entry.model.transformer_subfolder == "transformer"
        assert entry.model.trust_remote_code is None
        assert entry.model.use_shm is None
        # backend is teacher-owned and pinned, never inherited from the actor
        assert entry.engine.strategy == "fsdp"
        assert entry.engine.model_dtype is None
        assert entry.engine.micro_batch_size_per_gpu is None

    def test_two_teachers_rejected(self):
        with pytest.raises(NotImplementedError, match="exactly one teacher"):
            compose_teacher(ENABLED, ONE_TEACHER, "+actor_rollout_ref.teacher.models.expert.model.path=/ckpt/other")

    def test_enabled_without_models_rejected(self):
        with pytest.raises(ValueError, match="teacher.models"):
            compose_teacher(ENABLED)

    @pytest.mark.parametrize("resource_field", ["n_gpus_per_node", "nnodes"])
    def test_resource_fields_under_colocated_raise(self, resource_field):
        with pytest.raises(ValueError, match="colocated"):
            compose_teacher(ENABLED, ONE_TEACHER, f"actor_rollout_ref.teacher.placement.{resource_field}=1")

    def test_standalone_mode_rejected(self):
        with pytest.raises(NotImplementedError, match="next runtime PR"):
            compose_teacher(ENABLED, ONE_TEACHER, "actor_rollout_ref.teacher.placement.mode=standalone")

    def test_unknown_placement_mode_rejected(self):
        with pytest.raises(ValueError, match="placement.mode"):
            compose_teacher(ENABLED, ONE_TEACHER, "actor_rollout_ref.teacher.placement.mode=hybrid")

    def test_untyped_conversion_fails_loudly(self):
        """The group carries no `_target_`, anywhere, and that must stay true.

        Hydra only instantiates `_target_`-bearing nodes, and `models` keys are
        user-named, so no entry can pre-declare one. Adding `_target_` back would
        turn every user-added teacher into a bare dict -- entry defaults dropped,
        `__post_init__` never run -- instead of the loud failure below.
        """
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="diffusion_trainer", overrides=[ENABLED, ONE_TEACHER])
        teacher_node = cfg.actor_rollout_ref.teacher

        assert "_target_" not in teacher_node
        assert "_target_" not in teacher_node.models.default
        with pytest.raises(AssertionError, match="_target_"):
            omega_conf_to_dataclass(teacher_node)
