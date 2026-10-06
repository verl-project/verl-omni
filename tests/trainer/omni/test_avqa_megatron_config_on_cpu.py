# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compose the public AVQA launcher without allocating a GPU."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "num_gpus,rollout_gpus,rollout_tp,resource_overrides",
    [(6, 2, 2, False), (8, 4, 4, False), (8, 4, 4, True)],
)
def test_avqa_full_model_recipe(tmp_path, num_gpus, rollout_gpus, rollout_tp, resource_overrides):
    root = Path(__file__).parents[3]
    launcher = root / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_avqa_separate_async.sh"
    env = dict(
        os.environ,
        AVQA_CONFIG_ONLY="1",
        MODEL_PATH="/tmp/model",
        TRAIN_FILE="/tmp/train_strict.parquet",
        VAL_FILE="/tmp/validation.parquet",
        OUTPUT_DIR=str(tmp_path),
        NUM_GPUS=str(num_gpus),
        ROLLOUT_GPUS=str(rollout_gpus),
        ROLLOUT_TP=str(rollout_tp),
        PYTHON=sys.executable,
    )
    command = ["bash", str(launcher)]
    if resource_overrides:
        command.extend(["ray_kwargs.ray_init.num_cpus=12", "+ray_kwargs.ray_init.object_store_memory=2147483648"])
    process = subprocess.Popen(
        command,
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
        process.communicate(timeout=10)
        pytest.fail("AVQA config composition timed out")
    assert process.returncode == 0, stderr[-3000:]
    [config_path] = tmp_path.glob("run.*/config.yaml")
    config = OmegaConf.load(config_path)
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    assert config.data.train_files == "/tmp/train_strict.parquet"
    assert config.data.val_files == "/tmp/validation.parquet"
    assert config.data.max_prompt_length == 4096 and config.data.max_response_length == 2048
    assert config.data.mm_processor_kwargs.sampling_rate == 16000
    assert rollout.max_model_len == 6144 and dict(rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt) == {
        "image": 1,
        "audio": 1,
        "video": 0,
    }
    assert actor.strategy == "megatron"
    assert actor.megatron.tensor_model_parallel_size == 4
    assert actor.megatron.expert_model_parallel_size == 4
    assert actor.megatron.pipeline_model_parallel_size == 1
    assert actor.megatron.context_parallel_size == 1
    towers = actor.megatron.override_transformer_config
    assert not towers.freeze_language_model
    assert towers.freeze_vision_model and towers.freeze_audio_model
    assert not actor.use_dynamic_bsz and not rollout.log_prob_use_dynamic_bsz
    assert not config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz
    assert not config.algorithm.rollout_correction.bypass_mode
    assert config.trainer.v1.separate_async.parameter_sync_step == 1
    assert config.trainer.n_gpus_per_node == 4
    assert rollout.n_gpus_per_node == rollout_gpus and rollout.tensor_model_parallel_size == rollout_tp
    assert config.ray_kwargs.ray_init.num_gpus == num_gpus
    if resource_overrides:
        assert config.ray_kwargs.ray_init.num_cpus == 12
        assert config.ray_kwargs.ray_init.object_store_memory == 2147483648
    else:
        assert config.ray_kwargs.ray_init.num_cpus is None
        assert "object_store_memory" not in config.ray_kwargs.ray_init
    runtime_env = config.ray_kwargs.ray_init.runtime_env.env_vars
    assert "NCCL_NVLS_ENABLE" not in runtime_env
    assert "VLLM_ALLREDUCE_USE_SYMM_MEM" not in runtime_env
    assert actor.policy_loss.loss_mode == "gspo" and actor.loss_agg_mode == "seq-mean-token-mean"
    assert actor.optim.use_precision_aware_optimizer
    assert actor.optim.override_optimizer_config.optimizer_cpu_offload
    assert actor.optim.override_optimizer_config.optimizer_offload_fraction == 1.0
    assert not actor.use_kl_loss and config.algorithm.adv_estimator == "grpo"
    assert config.reward.custom_reward_function.path.endswith("choice_reward.py")
    assert config.trainer.total_training_steps == 150 and config.trainer.test_freq == 30
    assert config.data.val_max_samples == -1 and config.trainer.val_before_train
    assert config.trainer.save_freq == -1
    assert config.trainer.rollout_data_dir and config.trainer.validation_data_dir
