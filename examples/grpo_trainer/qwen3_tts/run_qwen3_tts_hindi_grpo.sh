#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}" \
TEST_FREQ="${TEST_FREQ:-20}" \
SAVE_FREQ="${SAVE_FREQ:-20}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_tts_hindi_grpo}" \
bash "${SCRIPT_DIR}/run_qwen3_tts_grpo.sh" \
    data.train_batch_size=4 \
    data.train_max_samples=863 \
    data.val_max_samples=100 \
    data.max_prompt_length=256 \
    data.max_response_length=240 \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']" \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.08 \
    actor_rollout_ref.actor.kl_loss_type=k3 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    trainer.experiment_name=qwen3_tts_0_6b_hindi_sft_lora_grpo \
    trainer.max_actor_ckpt_to_keep=21 \
    "$@"
