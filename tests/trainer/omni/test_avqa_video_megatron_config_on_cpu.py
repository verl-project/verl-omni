# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compose the shared AVQA video dataset on the Megatron separate-async route."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _compose(tmp_path, overrides, *arguments):
    root = Path(__file__).parents[3]
    launcher = root / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_avqa_video_separate_async.sh"
    env = dict(
        os.environ,
        AVQA_VIDEO_CONFIG_ONLY="1",
        MODEL_PATH="/tmp/model",
        TRAIN_FILE="/tmp/avqa_video_train.parquet",
        VAL_FILE="/tmp/avqa_video_validation.parquet",
        OUTPUT_DIR=str(tmp_path),
        PYTHON=sys.executable,
    )
    for key in ("NUM_GPUS", "ROLLOUT_GPUS", "ROLLOUT_TP"):
        env.pop(key, None)
    env.update(overrides)
    process = subprocess.Popen(
        ["bash", str(launcher), *arguments],
        cwd=root,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        pytest.fail("AVQA video config composition timed out")
    return process.returncode, stderr


@pytest.mark.parametrize(
    "overrides,num_gpus,rollout_gpus,rollout_tp",
    [
        ({}, 8, 4, 4),
        ({"NUM_GPUS": "6", "ROLLOUT_GPUS": "2", "ROLLOUT_TP": "2"}, 6, 2, 2),
        ({"NUM_GPUS": "8", "ROLLOUT_GPUS": "4", "ROLLOUT_TP": "4"}, 8, 4, 4),
    ],
)
def test_avqa_video_recipe(tmp_path, overrides, num_gpus, rollout_gpus, rollout_tp):
    returncode, stderr = _compose(tmp_path, overrides)
    assert returncode == 0, stderr[-3000:]

    [config_path] = tmp_path.glob("run.*/config.yaml")
    config = OmegaConf.load(config_path)
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    assert config.data.train_files == "/tmp/avqa_video_train.parquet"
    assert config.data.val_files == "/tmp/avqa_video_validation.parquet"
    assert config.data.custom_cls.name == "NextQARLHFDataset"
    assert config.data.mm_processor_kwargs.use_audio_in_video is False
    assert config.data.max_prompt_length == 8192 and config.data.max_response_length == 1024
    assert dict(rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt) == {"image": 0, "video": 1, "audio": 1}
    assert rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb == 0
    assert rollout.max_model_len == 9216 and rollout.n == 4
    assert rollout.max_num_batched_tokens >= rollout.max_model_len
    assert config.trainer.n_gpus_per_node == 4
    assert rollout.n_gpus_per_node == rollout_gpus and rollout.tensor_model_parallel_size == rollout_tp
    assert config.trainer.n_gpus_per_node + rollout.n_gpus_per_node == num_gpus
    assert actor.policy_loss.loss_mode == "gspo" and actor.loss_agg_mode == "seq-mean-token-mean"
    assert actor.optim.use_precision_aware_optimizer
    assert actor.optim.override_optimizer_config.optimizer_cpu_offload
    assert actor.optim.override_optimizer_config.optimizer_offload_fraction == 1.0
    assert actor.megatron.override_transformer_config.freeze_vision_model
    assert actor.megatron.override_transformer_config.freeze_audio_model
    assert not actor.megatron.override_transformer_config.freeze_language_model
    assert not actor.use_kl_loss and config.algorithm.adv_estimator == "grpo"
    assert config.reward.custom_reward_function.path == "pkg://verl_omni.utils.reward_score.choice_reward"
    assert config.trainer.total_training_steps == 20 and config.trainer.test_freq == 10
    assert config.data.val_max_samples == -1 and config.trainer.val_before_train
    assert config.trainer.save_freq == -1
    assert config.trainer.rollout_data_dir and config.trainer.validation_data_dir


def test_video_launcher_preserves_caller_overrides(tmp_path):
    returncode, stderr = _compose(
        tmp_path, {}, "trainer.total_training_steps=3", "trainer.test_freq=1", "ray_kwargs.ray_init.num_cpus=2"
    )
    assert returncode == 0, stderr[-3000:]
    [path] = tmp_path.glob("run.*/config.yaml")
    config = OmegaConf.load(path)
    assert config.trainer.total_training_steps == 3 and config.trainer.test_freq == 1
    assert config.ray_kwargs.ray_init.num_cpus == 2


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"ROLLOUT_GPUS": "3", "ROLLOUT_TP": "2"}, "divisible by TP"),
        ({"NUM_GPUS": "8", "ROLLOUT_GPUS": "2", "ROLLOUT_TP": "2"}, "four actor GPUs"),
        ({"NUM_GPUS": "invalid"}, "positive GPU count"),
    ],
)
def test_video_launcher_rejects_invalid_topology_before_artifacts(tmp_path, overrides, reason):
    returncode, stderr = _compose(tmp_path, overrides)
    assert returncode == 2 and reason in stderr
    assert not list(tmp_path.glob("run.*"))
