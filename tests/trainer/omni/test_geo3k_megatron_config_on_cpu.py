# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compose the public Geo3K launcher and check the full-model V1 formulation."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def test_geo3k_full_model_recipe(tmp_path):
    root = Path(__file__).parents[3]
    launcher = root / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_geo3k_separate_async.sh"
    env = dict(
        os.environ,
        GEO3K_CONFIG_ONLY="1",
        MODEL_PATH="/tmp/model",
        TRAIN_FILE="/tmp/train.parquet",
        VAL_FILE="/tmp/test.parquet",
        OUTPUT_DIR=str(tmp_path),
        PYTHON=sys.executable,
    )
    process = subprocess.Popen(
        ["bash", str(launcher)],
        cwd=root,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        pytest.fail(f"Geo3K config timed out: {stderr[-2000:]}")
    assert process.returncode == 0, stderr[-3000:]
    [path] = tmp_path.glob("run.*/config.yaml")
    config = OmegaConf.load(path)
    model, actor, rollout = (config.actor_rollout_ref[k] for k in ("model", "actor", "rollout"))
    assert model.model_type == "omni_model" and model.lora.rank == 0
    assert actor.strategy == "megatron"
    for engine in (actor.megatron, config.actor_rollout_ref.ref.megatron):
        assert engine.override_transformer_config.attention_dropout == 0.0
        assert engine.override_transformer_config.hidden_dropout == 0.0
    assert actor.megatron.override_transformer_config.moe_router_load_balancing_type == "none"
    assert actor.megatron.override_transformer_config.moe_aux_loss_coeff == 0.0
    assert actor.optim.override_optimizer_config.optimizer_cpu_offload
    assert actor.optim.override_optimizer_config.optimizer_offload_fraction == 0.5
    assert actor.megatron.pipeline_model_parallel_size == actor.megatron.context_parallel_size == 1
    assert actor.megatron.tensor_model_parallel_size == actor.megatron.expert_model_parallel_size == 4
    assert actor.megatron.override_transformer_config.freeze_vision_model
    assert actor.megatron.override_transformer_config.freeze_audio_model
    assert not actor.megatron.override_transformer_config.freeze_language_model
    assert not model.use_remove_padding and not actor.megatron.use_remove_padding
    assert not actor.use_dynamic_bsz and not rollout.log_prob_use_dynamic_bsz
    assert actor.policy_loss.loss_mode == "vanilla" and actor.loss_agg_mode == "token-mean"
    assert config.algorithm.adv_estimator == "grpo" and actor.use_kl_loss
    assert actor.kl_loss_coef == 0.001 and actor.kl_loss_type == "low_var_kl"
    assert not config.algorithm.rollout_correction.bypass_mode
    assert config.trainer.v1.trainer_mode == "omni_separate_async"
    assert (
        config.data.train_batch_size == actor.ppo_mini_batch_size * config.trainer.v1.separate_async.parameter_sync_step
    )
    assert config.trainer.nnodes == rollout.nnodes == 1
    assert config.trainer.n_gpus_per_node == rollout.n_gpus_per_node == 4
    assert rollout.tensor_model_parallel_size == 4 and rollout.n == 8
    assert rollout.checkpoint_engine.backend == "nccl" and rollout.calculate_log_probs
    assert rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt == {"image": 1, "audio": 0, "video": 0}
    assert config.reward.custom_reward_function.path is None
    assert config.trainer.total_training_steps >= 30 and config.trainer.test_freq == 10
