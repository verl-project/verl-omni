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

import json
import os
from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.model import MtpConfig

import verl_omni
from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config.diffusion import (
    DiffusionModelConfig,
    DiffusionTeacherConfig,
    TeacherCheckpointConfig,
    TeacherModelEntry,
    resolve_teacher_model_config,
)

CONFIG_DIR = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config")

ENABLED = "actor_rollout_ref.teacher.enabled=true"
ONE_TEACHER = "+actor_rollout_ref.teacher.models.default.model.path=/ckpt/teacher"


def compose_teacher(*overrides: str) -> DiffusionTeacherConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="diffusion_trainer", overrides=list(overrides))
    return omega_conf_to_dataclass(cfg.actor_rollout_ref.teacher, DiffusionTeacherConfig)


def make_fake_sd3_checkpoint(tmp_path, name, class_name="StableDiffusion3Pipeline") -> str:
    """A diffusers layout with just what DiffusionModelConfig.__post_init__ reads."""
    ckpt = tmp_path / name
    (ckpt / "scheduler").mkdir(parents=True)
    (ckpt / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    FlowMatchSDEDiscreteScheduler().save_pretrained(ckpt / "scheduler")
    return str(ckpt)


def make_teacher_entry(path, **model_overrides) -> TeacherModelEntry:
    return TeacherModelEntry(model=TeacherCheckpointConfig(path=path, **model_overrides))


def make_actor_model_config(path, **overrides) -> DiffusionModelConfig:
    kwargs = {
        "path": path,
        "algorithm": "flow_grpo",
        "attn_backend": "native",
        "load_tokenizer": False,
    }
    kwargs.update(overrides)
    return DiffusionModelConfig(**kwargs)


@pytest.fixture
def stub_tokenizer_loading(monkeypatch):
    """Let an actor config carry load_tokenizer=True without a real tokenizer on disk."""
    sentinel = object()
    monkeypatch.setattr("verl_omni.workers.config.diffusion.model.copy_to_local", lambda path, **kw: path)
    monkeypatch.setattr("verl_omni.workers.config.diffusion.model.hf_tokenizer", lambda *a, **kw: sentinel)
    return sentinel


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


class TestResolveTeacherModelConfig:
    """§5.2 provenance: teacher-owned / inherited / checkpoint-derived / forced."""

    @pytest.fixture
    def actor(self, tmp_path):
        return make_actor_model_config(
            make_fake_sd3_checkpoint(tmp_path, "student"),
            external_lib=None,
            fsdp_layer_prefixes=["transformer_blocks.", "single_blocks."],
        )

    @pytest.fixture
    def teacher_path(self, tmp_path):
        return make_fake_sd3_checkpoint(tmp_path, "teacher")

    def test_teacher_owned_path(self, actor, teacher_path):
        teacher = resolve_teacher_model_config(actor, make_teacher_entry(teacher_path))

        assert teacher.path == teacher_path
        assert teacher.path != actor.path
        # local_path is re-derived from the teacher path, never carried over
        assert teacher.local_path == teacher_path
        assert teacher.local_path != actor.local_path
        assert teacher.transformer_subfolder == "transformer"

    def test_disk_fields_default_to_actor_unless_set(self, tmp_path, teacher_path):
        actor = make_actor_model_config(
            make_fake_sd3_checkpoint(tmp_path, "student"), trust_remote_code=True, use_shm=False
        )

        inherited = resolve_teacher_model_config(actor, make_teacher_entry(teacher_path))
        assert inherited.trust_remote_code is True
        assert inherited.use_shm is False

        overridden = resolve_teacher_model_config(
            actor,
            make_teacher_entry(teacher_path, trust_remote_code=False, transformer_subfolder="dit"),
        )
        assert overridden.trust_remote_code is False
        assert overridden.transformer_subfolder == "dit"

    def test_inherited_fields(self, actor, teacher_path):
        teacher = resolve_teacher_model_config(actor, make_teacher_entry(teacher_path))

        assert teacher.model_type == actor.model_type
        assert teacher.algorithm == actor.algorithm
        assert teacher.attn_backend == actor.attn_backend
        assert teacher.external_lib == actor.external_lib
        assert teacher.fsdp_layer_prefixes == actor.fsdp_layer_prefixes
        # trajectory semantics: a teacher-owned sde_type/noise_level would stop
        # the distill_kl quantity from being a KL at all
        assert teacher.algo.sde_type == actor.algo.sde_type
        assert teacher.algo.noise_level == actor.algo.noise_level
        assert teacher.pipeline == actor.pipeline

        # copied, not aliased
        assert teacher.pipeline is not actor.pipeline
        assert teacher.algo is not actor.algo
        assert teacher.fsdp_layer_prefixes is not actor.fsdp_layer_prefixes

    def test_checkpoint_derived_fields(self, actor, teacher_path):
        """__post_init__ owns these; the resolver must leave them unset."""
        teacher = resolve_teacher_model_config(actor, make_teacher_entry(teacher_path))

        assert teacher.architecture == "StableDiffusion3Pipeline"

    def test_forced_frozen_defaults(self, tmp_path, teacher_path, stub_tokenizer_loading):
        """A maximally LoRA-flavoured actor must leak none of it into the teacher."""
        actor = make_actor_model_config(
            make_fake_sd3_checkpoint(tmp_path, "student"),
            load_tokenizer=True,
            enable_gradient_checkpointing=True,
            lora_rank=32,
            lora_adapter_path="/adapters/student",
            lora={"rank": 32},
            lora_dtype="fp32",
            policy_state_adapters=("default", "reference"),
            mtp=MtpConfig(enable=True),
            config_path="/student/transformer",
        )
        assert actor.tokenizer is stub_tokenizer_loading  # the actor really did load one

        teacher = resolve_teacher_model_config(actor, make_teacher_entry(teacher_path))

        assert teacher.load_tokenizer is False
        assert teacher.tokenizer is None
        assert teacher.local_tokenizer_path is None
        assert teacher.enable_gradient_checkpointing is False
        assert teacher.lora_rank == 0
        assert teacher.lora_adapter_path is None
        assert teacher.lora == {}
        assert teacher.lora_dtype is None
        assert teacher.policy_state_adapters == ()
        assert teacher.mtp.enable is False
        # the VeOmni engine resolves `config_path or weights_path`, so an inherited
        # config_path would define the teacher from the student's directory
        assert teacher.config_path is None

    def test_actor_config_not_mutated(self, actor, teacher_path):
        before = deepcopy(actor)

        resolve_teacher_model_config(actor, make_teacher_entry(teacher_path))

        assert actor == before
