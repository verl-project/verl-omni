#!/usr/bin/env bash
# =============================================================================
# Qwen3-Omni Thinker GSPO Training with LoRA (FSDP + vLLM-Omni)
#
# This script trains the Thinker stage of Qwen3-Omni using GSPO algorithm
# with LoRA adapters. It uses:
#   - Actor (training): transformers + FSDP + LoRA
#   - Rollout (inference): vLLM-Omni with LoRA hot-loading
#   - Algorithm: GRPO advantage + GSPO loss (matching Qwen3-Omni paper §4.1)
#   - Reward: rule-based math verification (GSM8K)
#
# Reference:
#   - Qwen3-Omni paper §4.1: Thinker post-training uses GSPO
#   - VeRL example: run_qwen2_5-3b_gsm8k_grpo_lora.sh
#   - Relax config: run-qwen3-30B-A3B-omni-16xgpu.sh
#
# Hardware: 2× H100 80GB (minimum for 30B-A3B MoE model)
# =============================================================================
set -x

# ─── NCCL Diagnostics ───────────────────────────────────────────────────
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

# ─── Model ───────────────────────────────────────────────────────────────
# Qwen3-Omni-30B-A3B is an MoE model (30B total, 3B active per token).
# The actor loads the FULL model via transformers (Thinker+Talker+Code2Wav)
# but LoRA only targets Thinker layers, so Talker/Code2Wav are frozen.
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}

# ─── Data ────────────────────────────────────────────────────────────────
# Start with GSM8K (text-only math) for simplest e2e validation.
# Switch to AVQA later for multimodal (audio+image) training.
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/gsm8k_nothink/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/gsm8k_nothink/test.parquet"}

# ─── Algorithm ───────────────────────────────────────────────────────────
# GSPO = GRPO advantage estimation + sequence-level policy loss.
# Matching Qwen3-Omni paper and Relax's Qwen3-Omni training config.
ADV_ESTIMATOR="grpo"       # Advantage: group-relative (no critic needed)
LOSS_MODE="gspo"           # Loss: sequence-level importance ratio
LOSS_AGG="seq-mean-token-mean"  # Aggregation: sequence mean first

# GSPO clipping: very tight, matching paper recommendation.
# Standard PPO uses ~0.2; GSPO uses ~0.0003-0.0004 for MoE stability.
CLIP_RATIO_LOW="3e-4"
CLIP_RATIO_HIGH="4e-4"

# ─── LoRA ────────────────────────────────────────────────────────────────
# LoRA rank 64 with alpha 32 (standard for MoE models).
# exclude_modules ensures LoRA is ONLY added to Thinker layers.
# Without this, PEFT would also add LoRA to Talker/Code2Wav (same layer names).
# freeze_vision_tower freezes audio_tower and visual encoder (no LoRA, no gradient).
LORA_RANK=64
LORA_ALPHA=32
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

# ─── GRPO Sampling ───────────────────────────────────────────────────────
# Generate N responses per prompt, compute group-relative advantage.
N_RESP=8                  # 8 responses per prompt (matches Relax config)
TEMPERATURE=0.8           # Exploration temperature (matches Relax)
TRAIN_BATCH_SIZE=8        # Prompts per batch (small: response_length=8192 uses ~8× more memory)

# ─── Rollout Engine ──────────────────────────────────────────────────────
# Use vLLM-Omni for inference. The rollout engine will:
# 1. Load base model weights
# 2. Hot-load LoRA adapter from actor after each training step
# 3. Generate responses with logprobs=1 for RL training
#
# NOTE: For Thinker-only mode, we use the thinker-only stage config.
# This avoids loading Talker/Code2Wav on the inference GPU.
ROLLOUT_NAME="vllm_omni_ar"  # AR server (token-in, token-out)
ROLLOUT_TP=4              # Tensor parallel for inference (4 GPU)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_omni_thinker_only.yaml"

python3 -m verl_omni.trainer.omni.main_ppo \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules="all-linear" \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.actor.freeze_vision_tower=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG} \
    \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=100000000 \
    \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    \
    reward.reward_manager.name=dapo \
    \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='qwen3_omni_thinker_rl' \
    trainer.experiment_name='gspo_lora_gsm8k' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    "$@"

# =============================================================================
# Notes:
#
# 1. To switch to AVQA (multimodal), change:
#    - TRAIN_FILE to AVQA dataset path
#    - reward.reward_manager.name to 'multiple_choice' (need to implement)
#    - Add: data.image_key=image data.audio_key=audio
#
# 2. To use GRPO instead of GSPO (simpler):
#    - Remove: actor_rollout_ref.actor.policy_loss.loss_mode
#    - Remove: actor_rollout_ref.actor.clip_ratio_low/high
#    - Remove: actor_rollout_ref.actor.loss_agg_mode
#
# 3. To scale up (8 GPU):
#    - trainer.n_gpus_per_node=8
#    - actor_rollout_ref.rollout.tensor_model_parallel_size=4
#    - data.train_batch_size=256
# =============================================================================
