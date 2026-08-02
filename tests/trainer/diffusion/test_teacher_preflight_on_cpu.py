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
"""CPU tests for the diffusion teacher static preflight."""

import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl_omni
from verl_omni.trainer.diffusion.teacher_preflight import validate_teacher_preflight

CONFIG_DIR = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config")

TEACHER_ON = "actor_rollout_ref.teacher.enabled=true"
TEACHER_CKPT = "+actor_rollout_ref.teacher.models.default.model.path=/ckpt/teacher"
DISTILL_KL = "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"
TEACHER = (TEACHER_ON, TEACHER_CKPT, DISTILL_KL)


def compose_config(*overrides: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=list(overrides))


# (id, overrides, stack, expected exception, message fragment)
REJECTIONS = [
    (
        "distill_loss_without_teacher",
        (DISTILL_KL,),
        "v0",
        ValueError,
        "actor_rollout_ref.teacher.enabled",
    ),
    (
        "auxiliary_distill_loss_without_teacher",
        ("actor_rollout_ref.actor.use_distill_loss=true",),
        "v0",
        ValueError,
        "actor_rollout_ref.teacher.enabled",
    ),
    (
        "teacher_without_distill_loss",
        (TEACHER_ON, TEACHER_CKPT),
        "v0",
        ValueError,
        "no distillation loss consumes",
    ),
    (
        "top_level_distillation_armed",
        ("+distillation.enabled=true",),
        "v0",
        ValueError,
        "distillation",
    ),
    (
        "v1_stack_argument",
        TEACHER,
        "v1",
        ValueError,
        "v1",
    ),
    (
        "use_v1_flag",
        TEACHER + ("trainer.use_v1=true",),
        "v0",
        ValueError,
        "trainer.use_v1",
    ),
    (
        "direct_preference_trainer",
        TEACHER + ("algorithm.trainer_type=direct_preference",),
        "v0",
        ValueError,
        "policy_gradient",
    ),
    (
        "offline_sample_source",
        TEACHER + ("algorithm.sample_source=offline",),
        "v0",
        ValueError,
        "sample_source",
    ),
    (
        "non_fsdp_teacher_backend",
        TEACHER + ("+actor_rollout_ref.teacher.models.default.engine.strategy=veomni",),
        "v0",
        ValueError,
        "engine.strategy",
    ),
    (
        "non_fsdp_actor_backend",
        TEACHER + ("actor_rollout_ref.actor.strategy=veomni",),
        "v0",
        ValueError,
        "actor_rollout_ref.actor.strategy",
    ),
    (
        "standalone_placement",
        TEACHER + ("actor_rollout_ref.teacher.placement.mode=standalone",),
        "v0",
        NotImplementedError,
        "next runtime PR",
    ),
    (
        "fm_mse_has_no_producer",
        (TEACHER_ON, TEACHER_CKPT, "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_fm_mse"),
        "v0",
        ValueError,
        "distill_fm_mse",
    ),
]


class TestTeacherPreflight:
    @pytest.mark.parametrize("stack", ["v0", "v1"])
    def test_default_config_passes(self, stack):
        """Default-off: the validator is a no-op on every existing recipe."""
        validate_teacher_preflight(compose_config(), stack=stack)

    def test_valid_teacher_config_passes(self):
        validate_teacher_preflight(compose_config(*TEACHER), stack="v0")

    def test_auxiliary_distill_loss_with_teacher_passes(self):
        """use_distill_loss next to a policy-gradient loss is the other valid pairing."""
        validate_teacher_preflight(
            compose_config(TEACHER_ON, TEACHER_CKPT, "actor_rollout_ref.actor.use_distill_loss=true"),
            stack="v0",
        )

    @pytest.mark.parametrize(
        "overrides,stack,expected,fragment",
        [case[1:] for case in REJECTIONS],
        ids=[case[0] for case in REJECTIONS],
    )
    def test_rejections(self, overrides, stack, expected, fragment):
        with pytest.raises(expected, match=fragment):
            validate_teacher_preflight(compose_config(*overrides), stack=stack)


class TestPreflightIsBound:
    """Sentinels: both run functions must call the validator, before ray.init.

    Routing every rejection through a Hydra-decorated main() would test Hydra more
    than the contract, so the suite above drives the validator directly and these
    two only prove the binding.
    """

    @staticmethod
    def _arm(monkeypatch, module, calls):
        def fake_validate(config, stack):
            calls.append(stack)
            raise RuntimeError("preflight sentinel")

        def exploding_init(*args, **kwargs):
            raise AssertionError("ray.init reached before the preflight")

        monkeypatch.setattr(module, "validate_teacher_preflight", fake_validate)
        monkeypatch.setattr(module.ray, "is_initialized", lambda: False)
        monkeypatch.setattr(module.ray, "init", exploding_init)

    def test_run_diffusion_passes_v0(self, monkeypatch):
        from verl_omni.trainer import main_diffusion

        calls = []
        self._arm(monkeypatch, main_diffusion, calls)

        with pytest.raises(RuntimeError, match="preflight sentinel"):
            main_diffusion.run_diffusion(OmegaConf.create({}))

        assert calls == ["v0"]

    def test_run_diffusion_v1_passes_v1(self, monkeypatch):
        from verl_omni.trainer import main_diffusion_v1

        calls = []
        self._arm(monkeypatch, main_diffusion_v1, calls)

        with pytest.raises(RuntimeError, match="preflight sentinel"):
            main_diffusion_v1.run_diffusion_v1(OmegaConf.create({}))

        assert calls == ["v1"]
