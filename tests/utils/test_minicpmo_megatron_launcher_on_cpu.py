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
"""Exercise the public MiniCPM Megatron launcher's process and config boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
LAUNCHER = ROOT / "examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_megatron_avqa.sh"


def _environment(tmp_path):
    return dict(
        os.environ,
        MODEL_PATH="/model with spaces",
        TRAIN_FILE="/data/train file.parquet",
        VAL_FILE="/data/validation file.parquet",
        OUTPUT_DIR=str(tmp_path / "run with spaces"),
    )


def test_paths_overrides_and_failure_status_are_preserved(tmp_path):
    capture = tmp_path / "argv.json"
    interpreter = tmp_path / "python-recorder"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + "import json, os, sys\n"
        + "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
        + "sys.exit(17)\n"
    )
    interpreter.chmod(0o755)
    env = dict(_environment(tmp_path), PYTHON=str(interpreter), CAPTURE=str(capture))
    result = subprocess.run(
        ["bash", str(LAUNCHER), "trainer.total_training_steps=100"],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 17
    args = json.loads(capture.read_text())
    assert args[:4] == ["-m", "verl_omni.trainer.main_omni", "--config-name", "omni_megatron_trainer"]
    assert "actor_rollout_ref.model.path=/model with spaces" in args
    assert "data.train_files=/data/train file.parquet" in args
    assert "data.val_files=/data/validation file.parquet" in args
    assert args[-1] == "trainer.total_training_steps=100"
    assert args.index("trainer.total_training_steps=1") < len(args) - 1


@pytest.mark.parametrize("missing", ["MODEL_PATH", "TRAIN_FILE", "VAL_FILE"])
def test_missing_required_paths_fail_before_python(tmp_path, missing):
    env = dict(_environment(tmp_path), PYTHON="/does/not/exist")
    env.pop(missing)
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert missing in result.stderr
    assert "/does/not/exist" not in result.stderr


def test_shared_config_resolves_and_accepts_long_run_overrides(tmp_path):
    pytest.importorskip("verl", reason="Full config composition needs the training dependencies")
    from omegaconf import OmegaConf

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "trainer.total_training_steps=100",
            "trainer.total_epochs=2",
            "trainer.test_freq=10",
            "data.train_max_samples=-1",
            "data.val_max_samples=-1",
            "--cfg",
            "job",
            "--resolve",
        ],
        env=dict(_environment(tmp_path), PYTHON=sys.executable, CUDA_VISIBLE_DEVICES=""),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    config = OmegaConf.create(result.stdout)
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    assert actor.strategy == "megatron" and actor.megatron.tensor_model_parallel_size == 4
    assert actor.megatron.pipeline_model_parallel_size == actor.megatron.context_parallel_size == 1
    assert not actor.megatron.sequence_parallel and not actor.megatron.use_remove_padding
    assert not config.actor_rollout_ref.model.use_fused_kernels
    assert rollout.n_gpus_per_node == 4 and rollout.tensor_model_parallel_size == 2
    assert config.trainer.n_gpus_per_node == 4 and config.ray_kwargs.ray_init.num_gpus == 8
    assert actor.megatron.override_transformer_config.freeze_vision_model
    assert actor.megatron.override_transformer_config.freeze_audio_model
    assert not actor.megatron.override_transformer_config.freeze_language_model
    assert actor.policy_loss.loss_mode == "gspo" and config.algorithm.adv_estimator == "grpo"
    assert actor.clip_ratio_low == 0.0003 and actor.clip_ratio_high == 0.0004
    assert config.trainer.total_training_steps == 100 and config.trainer.test_freq == 10
    assert config.data.train_max_samples == config.data.val_max_samples == -1
    assert rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb == 0
    assert config.trainer.v1.trainer_mode == "omni_separate_async"
    assert config.reward.custom_reward_function.path.endswith("choice_reward.py")
