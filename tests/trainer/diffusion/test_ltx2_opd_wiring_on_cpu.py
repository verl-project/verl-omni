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
"""CPU tests for the LTX-2.3 text-to-audio-video on-policy distillation recipe."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import verl_omni
from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.config.diffusion import DiffusionDistillationTeacherModelConfig, DiffusionPipelineConfig
from verl_omni.workers.engine_workers import build_teacher_training_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")

# The LTX-2.3 T2AV geometry and rollout wiring shared by every recipe in
# examples/flowgrpo_trainer/ltx2 and examples/diffusionopd_trainer/ltx2.
LTX2_T2AV = [
    "actor_rollout_ref.model.algorithm=flow_grpo",
    "actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent",
    "actor_rollout_ref.rollout.pipeline.height=256",
    "actor_rollout_ref.rollout.pipeline.width=384",
    "actor_rollout_ref.rollout.pipeline.num_frames=81",
    "actor_rollout_ref.rollout.pipeline.frame_rate=24.0",
    "actor_rollout_ref.rollout.pipeline.num_inference_steps=24",
    "actor_rollout_ref.rollout.pipeline.guidance_scale=4.0",
    "actor_rollout_ref.rollout.algo.sde_type=cps",
    "actor_rollout_ref.rollout.algo.noise_level=0.8",
    "trainer.use_v1=true",
    "trainer.v1.trainer_mode=sync",
]

# Pure distillation: the CLAP and ImageBind rewards stay monitored only.
OPD = [
    "distillation.enabled=true",
    "distillation.teacher_models.teacher_model.model_path=/ckpt/ltx2-teacher",
    "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
    "actor_rollout_ref.actor.use_kl_loss=False",
]


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


def make_trainer(overrides):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    return PolicyGradientDiffusionTrainerV1Sync(compose_cfg(LTX2_T2AV + overrides))


class TestLTX2OPDComposition:
    def test_recipe_overrides_enable_a_single_teacher(self):
        trainer = make_trainer(OPD)
        assert trainer.use_teacher_policy
        teachers = trainer.distillation_config.teacher_models
        assert list(teachers) == ["default"]
        assert teachers["default"].model_path == "/ckpt/ltx2-teacher"
        assert trainer.distillation_config.scheduler == "inline"

    def test_teacher_colocates_with_the_actor_by_default(self):
        from verl.trainer.ppo.utils import Role

        trainer = make_trainer(OPD)
        trainer._init_resource_pool_mgr()
        assert "teacher_pool" not in trainer.resource_pool_manager.resource_pool_spec
        assert Role.TeacherModel not in trainer.role_worker_mapping

    def test_standalone_pool_gives_the_sole_teacher_the_whole_pool(self):
        from verl.trainer.ppo.utils import Role

        trainer = make_trainer(OPD + ["distillation.nnodes=1", "distillation.n_gpus_per_node=4"])
        trainer._init_resource_pool_mgr()
        assert trainer.resource_pool_manager.resource_pool_spec["teacher_pool"] == [4]
        assert trainer.mapping[Role.TeacherModel] == "teacher_pool"
        assert trainer.distillation_config.teacher_models["default"].world_size == 4

    def test_one_step_off_is_rejected_on_sync(self):
        # the recipe runs sync, so the overlapped schedule has no async step to hide behind
        with pytest.raises(ValueError, match="one_step_off"):
            make_trainer(OPD + ["distillation.scheduler=one_step_off"])

    def test_teacher_without_a_distillation_loss_raises(self):
        with pytest.raises(ValueError, match="no distillation loss is active"):
            make_trainer([override for override in OPD if "loss_mode" not in override])

    def test_distill_loss_without_a_teacher_raises(self):
        with pytest.raises(ValueError, match="no teacher is configured"):
            make_trainer(["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"])


class TestLTX2TeacherDerivation:
    """The teacher resolves the student's LTX adapter and inherits its video geometry."""

    @staticmethod
    def _student_config(path, **overrides):
        pipeline = DiffusionPipelineConfig(
            height=256, width=384, num_frames=81, frame_rate=24.0, num_inference_steps=24, guidance_scale=4.0
        )
        return DiffusionModelConfig(
            path=str(path),
            architecture="LTX2Pipeline",
            algorithm="flow_grpo",
            attn_backend="native",
            load_tokenizer=False,
            transformer_config={},
            pipeline=pipeline,
            lora_rank=64,
            **overrides,
        )

    def test_architecture_and_algorithm_resolve_the_ltx_adapter(self):
        assert DiffusionModelBase.get_class_by_name("LTX2Pipeline", "flow_grpo") is LTX23FlowGRPO

    def test_teacher_inherits_the_student_video_geometry(self, tmp_path):
        student_dir, teacher_dir = tmp_path / "student", tmp_path / "teacher"
        student_dir.mkdir()
        teacher_dir.mkdir()
        cfg = compose_cfg(LTX2_T2AV + OPD + ["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"])

        teacher_training_config = build_teacher_training_config(
            config=cfg.actor_rollout_ref,
            model_config=self._student_config(student_dir),
            teacher_model_config=DiffusionDistillationTeacherModelConfig(model_path=str(teacher_dir)),
        )

        teacher_model_config = teacher_training_config.model_config
        assert teacher_model_config.path == str(teacher_dir)
        assert teacher_model_config.architecture == "LTX2Pipeline"
        assert teacher_model_config.algorithm == "flow_grpo"
        # prepare_model_inputs derives the latent grid from these, so a teacher that did not
        # inherit them would score a different trajectory than the student sampled
        assert teacher_model_config.pipeline.num_frames == 81
        assert teacher_model_config.pipeline.frame_rate == 24.0
        assert teacher_model_config.pipeline.height == 256
        assert teacher_model_config.pipeline.width == 384
        assert teacher_model_config.pipeline.num_inference_steps == 24
        # the teacher never loads adapters, so an unmerged LoRA teacher is not silently accepted
        assert teacher_model_config.lora_rank == 0
        assert teacher_model_config.lora_adapter_path is None
        assert teacher_training_config.engine_config.forward_only is True
        assert teacher_training_config.engine_config.infer_micro_batch_size_per_gpu == 1


RECIPE = Path(__file__).resolve().parents[3] / "examples/diffusionopd_trainer/ltx2/run_ltx2_3_t2av_opd_npu.sh"


def run_recipe(tmp_path, device, **env):
    """Run the launcher with a stub interpreter and stub device tooling, capturing its overrides."""
    args_path = tmp_path / "args.txt"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()

    python = stub_bin / "python3"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$LTX_TEST_ARGS"\n')
    npu_smi = stub_bin / "npu-smi"
    npu_smi.write_text("#!/bin/sh\nexit 0\n" if device == "npu" else "#!/bin/sh\nexit 1\n")
    nvidia_smi = stub_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho stub-gpu\necho stub-gpu\n")
    for stub in (python, npu_smi, nvidia_smi):
        stub.chmod(0o755)

    ascend_home = tmp_path / "ascend" / "ascend-toolkit"
    (ascend_home).mkdir(parents=True)
    (ascend_home / "set_env.sh").write_text("")
    atb = tmp_path / "ascend" / "nnal" / "atb"
    atb.mkdir(parents=True)
    (atb / "set_env.sh").write_text("")

    subprocess.run(
        ["bash", str(RECIPE)],
        env={
            **os.environ,
            "PATH": f"{stub_bin}:{os.environ['PATH']}",
            "OUTPUT_DIR": str(tmp_path / "output"),
            "LTX_TEST_ARGS": str(args_path),
            "TEACHER_PATH": "/ckpt/ltx2-teacher",
            "ASCEND_HOME_PATH": str(ascend_home),
            **env,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    args = args_path.read_text().splitlines()
    return dict(arg.split("=", 1) for arg in args if "=" in arg), args


class TestLTX2OPDRecipeLauncher:
    def test_recipe_requires_a_teacher_checkpoint(self, tmp_path):
        with pytest.raises(subprocess.CalledProcessError):
            run_recipe(tmp_path, "npu", TEACHER_PATH="")

    @pytest.mark.parametrize("device", ["npu", "gpu"])
    def test_recipe_overrides_compose_into_the_intended_opd_config(self, tmp_path, device):
        overrides, args = run_recipe(tmp_path, device)

        # composing catches a key the schema does not have, the usual new-recipe typo
        cfg = compose_cfg([arg for arg in args if "=" in arg])

        assert args[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]
        assert cfg.trainer.device == device
        assert cfg.trainer.use_v1 is True
        assert cfg.trainer.v1.trainer_mode == "sync"
        assert cfg.distillation.enabled is True
        assert cfg.distillation.teacher_models.teacher_model.model_path == "/ckpt/ltx2-teacher"
        # colocated by default; a standalone pool is opt-in
        assert cfg.distillation.nnodes == 0
        assert cfg.distillation.n_gpus_per_node == 0
        assert cfg.actor_rollout_ref.actor.diffusion_loss.loss_mode == "distill_kl"
        assert cfg.actor_rollout_ref.actor.use_kl_loss is False
        # OPD is a config composition on top of the registered (LTX2Pipeline, flow_grpo) pair
        assert cfg.actor_rollout_ref.model.algorithm == "flow_grpo"
        assert cfg.actor_rollout_ref.rollout.agent.default_agent_loop == "ltx2_diffusion_single_turn_agent"
        assert overrides["actor_rollout_ref.rollout.pipeline.num_frames"] == "81"

    def test_standalone_pool_is_opt_in_through_the_environment(self, tmp_path):
        overrides, _ = run_recipe(tmp_path, "npu", TEACHER_NNODES="1", TEACHER_NPUS="4")

        assert overrides["distillation.nnodes"] == "1"
        assert overrides["distillation.n_gpus_per_node"] == "4"

    def test_npu_branch_sets_ascend_only_settings(self, tmp_path):
        overrides, _ = run_recipe(tmp_path, "npu")

        assert overrides["trainer.resume_mode"] == "disable"
        assert overrides["actor_rollout_ref.actor.fsdp_config.param_offload"] == "True"
        assert overrides["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] == "True"
        assert overrides["+reward.reward_functions.clap.device"] == "npu:0"
