#!/usr/bin/env bash
# SD3.5-Medium LoRA OCR recipe aligned with mm_grpo's fast SD3 OCR script.
#
# mm_grpo uses a diffusers rollout with raw train.txt/test.txt prompts and a
# PaddleOCR reward. This repo uses vllm_omni rollout, parquet data, and a
# GenRM-OCR reward model, so this script maps only the comparable knobs into the
# local config structure.
#
# Reference:
# https://github.com/chenyingshu/mm_grpo/blob/main/examples/flowgrpo_trainer/experimental/run_sd3_fast_2p_a1_r1.sh
set -x

# Set OCR_WORKSPACE or WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${OCR_WORKSPACE:-${WORKSPACE:-$HOME}}

ocr_train_path=$WORKSPACE/data/ocr/sd3/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/sd3/test.parquet

model_name=stabilityai/stable-diffusion-3.5-medium
reward_model_name=Qwen/Qwen2.5-VL-3B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

NUM_GPUS_ACTOR_ROLLOUT=2
NUM_GPUS_REWARD=1
ROLLOUT_TP=1
REWARD_TP=1
IMAGE_RESOLUTION=384
TOTAL_TRAINING_STEPS=100
ATTN_BACKEND=native

MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

if [ "${FA3:-0}" = "1" ]; then
    ATTN_BACKEND="_flash_3_varlen_hub"
    ROLLOUT_ATTN_BACKEND=FLASH_ATTN_3_HUB
fi

ENGINE=vllm_omni
REWARD_ENGINE=vllm

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=8 \
    data.val_max_samples=32 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    actor_rollout_ref.model.attn_backend=$ATTN_BACKEND \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type="cps" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=${MAX_NUM_SEQS} \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=${REQUEST_BATCH_MAX_WAIT_MS} \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward.num_workers=$((NUM_GPUS_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=$NUM_GPUS_REWARD \
    reward.reward_model.rollout.gpu_memory_utilization=0.9 \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.enforce_eager=False \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=sd35_medium_ocr_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    "$@"
