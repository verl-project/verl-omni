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
"""Compose the real CRD recipe and reject unsupported group/policy configurations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.diffusion_algos import CRDLoss, DGPOLoss

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def recipe_args(capture_launcher):
    def run(*overrides, code=0, comparison=False):
        recipe = "run_dgpo_comparison.sh" if comparison else "run_sd35_medium_ocr_lora_v1.sh"
        result, calls = capture_launcher(f"examples/crd_trainer/sd35/{recipe}", *overrides, exit_code=code)
        assert len(calls) == 1
        return result, calls[0]

    return run


def compose_recipe(args):
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        return compose(config_name="diffusion_trainer", overrides=args[2:])


def test_recipe_composes_and_preserves_caller_overrides(recipe_args):
    result, args = recipe_args(
        "actor_rollout_ref.actor.optim.lr=0.0002",
        "trainer.total_training_steps=3",
        "trainer.resume_mode=auto",
        "trainer.max_actor_ckpt_to_keep=5",
    )
    assert result.returncode == 0, result.stderr
    assert args[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]
    config = compose_recipe(args)
    CRDLoss.validate_trainer_config(config)
    assert config.actor_rollout_ref.model.algorithm == "crd"
    assert config.actor_rollout_ref.actor.diffusion_loss.loss_mode == "crd"
    assert config.actor_rollout_ref.actor.optim.lr == 0.0002
    assert config.trainer.total_training_steps == 3
    assert config.trainer.resume_mode == "auto"
    assert config.trainer.max_actor_ckpt_to_keep == 5
    assert config.actor_rollout_ref.rollout.n == 2
    assert "workspace with spaces" in config.data.train_files
    assert "output with spaces" in config.trainer.default_local_dir


def test_recipe_preserves_trainer_exit_status(recipe_args):
    result, _ = recipe_args(code=17)
    assert result.returncode == 17


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("actor_rollout_ref.model.policy_state_adapters", ["default", "old"], "moving_reference"),
        ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1, "multiple of rollout.n"),
        ("trainer.v1.trainer_mode", "separate_async", "V1 sync"),
        ("trainer.use_v1", False, "V1 sync"),
        ("algorithm.paired_preference", True, "paired_preference"),
        ("actor_rollout_ref.rollout.n", 0, "rollout.n"),
        ("actor_rollout_ref.model.lora.merge", True, "lora.merge"),
        ("actor_rollout_ref.model.pipeline.reference_guidance_scale", float("nan"), "reference_guidance_scale"),
    ],
)
def test_unsupported_recipe_configs_fail_early(recipe_args, key, value, match):
    _, args = recipe_args()
    config = compose_recipe(args)
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match=match):
        CRDLoss.validate_trainer_config(config)


def test_dgpo_comparison_matches_experiment_settings(recipe_args):
    _, crd_args = recipe_args()
    result, dgpo_args = recipe_args(comparison=True)
    assert result.returncode == 0, result.stderr
    crd, dgpo = compose_recipe(crd_args), compose_recipe(dgpo_args)
    DGPOLoss.validate_trainer_config(dgpo)
    assert dgpo.actor_rollout_ref.actor.diffusion_loss.loss_mode == "dgpo"
    for config in (crd, dgpo):
        assert config.trainer.resume_mode == "disable"
        assert config.trainer.max_actor_ckpt_to_keep == 2
    for key in (
        "data",
        "reward",
        "actor_rollout_ref.rollout",
        "actor_rollout_ref.actor.optim",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.model.lora_rank",
        "actor_rollout_ref.model.lora_alpha",
        "trainer.total_training_steps",
        "trainer.test_freq",
    ):
        assert OmegaConf.select(crd, key) == OmegaConf.select(dgpo, key), key


@pytest.fixture
def capture_launcher(tmp_path):
    """Record every subprocess while stubbing only the Python executable."""
    stub = tmp_path / "python3"
    stub.write_text(
        f"#!{sys.executable} -S\n"
        "import json, os, sys\n"
        "with open(os.environ['CRD_CAPTURE'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "is_trainer = sys.argv[1:3] == ['-m', 'verl_omni.trainer.main_diffusion_v1']\n"
        "raise SystemExit(int(os.environ.get('CRD_TRAINER_EXIT', '0')) if is_trainer else 0)\n"
    )
    stub.chmod(0o755)
    captured = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CRD_CAPTURE": str(captured),
        "MODEL_PATH": str(tmp_path / "tiny model"),
        "OUTPUT_DIR": str(tmp_path / "output with spaces"),
        "OCR_WORKSPACE": str(tmp_path / "workspace with spaces"),
        "NUM_GPUS": "2",
        "NUM_GPUS_ACTOR_ROLLOUT": "1",
        "NUM_GPUS_REWARD": "1",
        "GROUP_SIZE": "2",
    }

    def run(script, *overrides, exit_code=0):
        captured.write_text("")
        result = subprocess.run(
            ["bash", str(ROOT / script), *overrides],
            env={**env, "CRD_TRAINER_EXIT": str(exit_code)},
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return result, [json.loads(line) for line in captured.read_text().splitlines()]

    return run


def test_tiny_gpu_smoke_composes_crd_and_runs_checkpoint_regression(capture_launcher):
    result, calls = capture_launcher("tests/special_e2e/run_crd_sd35.sh")
    assert result.returncode == 0, result.stderr
    trainer_args = next(args for args in calls if args[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"])
    config = compose_recipe(trainer_args)
    CRDLoss.validate_trainer_config(config)
    assert config.trainer.n_gpus_per_node == 2
    assert config.reward.models == {}
    assert config.reward.custom_reward_function.path.endswith("crd_smoke_reward.py")
    assert set(config.actor_rollout_ref.model.target_modules) == {
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    }
    assert calls[-1][:2] == ["-m", "torch.distributed.run"]
    assert "tests/workers/crd_fsdp_checkpoint.py" in calls[-1]


@pytest.mark.parametrize(
    "script", ["tests/special_e2e/run_crd_sd35.sh", "tests/special_e2e/run_crd_sd35_ocr_resume.sh"]
)
def test_validation_launchers_stop_on_failed_training(capture_launcher, script):
    result, calls = capture_launcher(script, exit_code=17)
    assert result.returncode == 17
    assert len([args for args in calls if args[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]]) == 1
    assert not any("tests/workers/crd_fsdp_checkpoint.py" in args for args in calls)


def test_real_model_validation_launches_fresh_then_resume(capture_launcher):
    result, calls = capture_launcher("tests/special_e2e/run_crd_sd35_ocr_resume.sh")
    assert result.returncode == 0, result.stderr
    trainers = [compose_recipe(args) for args in calls if args[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]]
    assert [cfg.trainer.total_training_steps for cfg in trainers] == [2, 4]
    assert [cfg.trainer.resume_mode for cfg in trainers] == ["disable", "resume_path"]
    assert trainers[1].trainer.resume_from_path.endswith("global_step_2")
    checkers = [args for args in calls if args[0] == "tests/special_e2e/check_crd_run.py"]
    assert checkers[0][-3:] == ["--steps", "1", "2"]
    assert checkers[1][-3:] == ["--steps", "3", "4"]
