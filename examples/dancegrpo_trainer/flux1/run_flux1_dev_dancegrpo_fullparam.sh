#!/usr/bin/env bash
# FLUX.1-dev full-parameter DanceGRPO recipe with HPSv2 reward.
set -euo pipefail

if ! npu-smi info >/dev/null 2>&1; then
    echo "Error: this recipe requires Ascend NPUs." >&2
    exit 1
fi

WORKSPACE=${WORKSPACE:-$HOME}
TRAIN_FILES_PATH=${TRAIN_FILES_PATH:-$WORKSPACE/data/hpsv2/train.parquet}
VAL_FILES_PATH=${VAL_FILES_PATH:-$WORKSPACE/data/hpsv2/test.parquet}

: "${MODEL_NAME:?Set MODEL_NAME to a local FLUX.1-dev diffusers checkpoint}"
: "${CUSTOM_REWARD_MODEL_PATH:?Set CUSTOM_REWARD_MODEL_PATH to HPS_v2.1_compressed.pt}"
: "${HPSV2_PRETRAINED_PATH:?Set HPSV2_PRETRAINED_PATH to open_clip_pytorch_model.bin}"

REWARD_FUNCTION_PATH=verl_omni/utils/reward_score/hpsv2_reward.py
REWARD_FUNCTION_NAME=compute_score_hpsv2
# RewardLoopWorkers do not reserve Ray accelerator resources. Keep the HPSv2
# scorers on CPU while using two workers for scoring throughput. Accelerator
# use must be an explicit override.
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-2}
REWARD_DEVICE=${REWARD_DEVICE:-cpu}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-fp32}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-flux1_dev_fullparam_hpsv2}

export CUSTOM_REWARD_MODEL_PATH HPSV2_PRETRAINED_PATH REWARD_DEVICE

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-1.0}

NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_N=${ROLLOUT_N:-12}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
SAVE_FREQ=${SAVE_FREQ:-1000000000}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}
PROJECT_NAME=${PROJECT_NAME:-dance_grpo}
CUSTOM_CHAT_TEMPLATE='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

if (( NUM_GPUS % ROLLOUT_TP != 0 )); then
    echo "Error: NUM_GPUS ($NUM_GPUS) must be divisible by ROLLOUT_TP ($ROLLOUT_TP)." >&2
    exit 1
fi

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=npu \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    algorithm.adv_estimator=dance_grpo \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.model.custom_chat_template="\"$CUSTOM_CHAT_TEMPLATE\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_2, max_length: 256}}' \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.optim.optimizer=AdamW \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$NUM_GPUS \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$ACTOR_MODEL_DTYPE \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.layered_summon=true \
    +actor_rollout_ref.rollout.enable_sleep_mode=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.pipeline.height=720 \
    actor_rollout_ref.rollout.pipeline.width=720 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=16 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=3.5 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.max_prompt_embed_length=256 \
    actor_rollout_ref.rollout.algo.sde_type=dance_sde \
    actor_rollout_ref.rollout.algo.noise_level=0.3 \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.algo.sde_window_range=null \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    data.train_files=$TRAIN_FILES_PATH \
    data.val_files=$VAL_FILES_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=256 \
    data.shuffle=true \
    data.seed=1223627 \
    data.dataloader_num_workers=4 \
    reward.num_workers=$REWARD_NUM_WORKERS \
    reward.reward_model.enable=false \
    reward.custom_reward_function.path=$REWARD_FUNCTION_PATH \
    reward.custom_reward_function.name=$REWARD_FUNCTION_NAME \
    "$@"
