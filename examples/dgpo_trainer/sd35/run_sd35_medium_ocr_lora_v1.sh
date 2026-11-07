#!/usr/bin/env bash
# SD3.5-Medium LoRA OCR recipe for DGPO (V1 trainer, sync mode).
#
# Same data, reward, LoRA and rollout resolution/steps as
# examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh; the learning
# rate (3e-4) and rollout guidance scale (4.5) follow the DGPO reference
# implementation. DGPO rolls out deterministically,
# trains from the final latents with a group-level preference weight, and
# needs every rollout group whole inside one micro batch on one rank
# (ppo_micro_batch_size_per_gpu a multiple of rollout.n, ppo_mini_batch_size
# divisible by the number of actor GPUs).
set -x

# Set OCR_WORKSPACE or WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${OCR_WORKSPACE:-${WORKSPACE:-$HOME}}

ocr_train_path=$WORKSPACE/data/ocr/sd3/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/sd3/test.parquet

model_name=stabilityai/stable-diffusion-3.5-medium
reward_model_name=Qwen/Qwen2.5-VL-3B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

NUM_GPUS_ACTOR_ROLLOUT=${NUM_GPUS_ACTOR_ROLLOUT:-2}
NUM_GPUS_REWARD=${NUM_GPUS_REWARD:-1}
ROLLOUT_TP=1
REWARD_TP=1
IMAGE_RESOLUTION=384
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA
MAX_NUM_SEQS=256
N_RESP_PER_PROMPT=8

if [ "${FA3:-0}" = "1" ]; then
    ATTN_BACKEND="_flash_3_varlen_hub"
    ROLLOUT_ATTN_BACKEND=FLASH_ATTN_3_HUB
fi

ENGINE=vllm_omni
REWARD_ENGINE=vllm

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="$ocr_train_path" \
    data.val_files="$ocr_test_path" \
    data.train_batch_size=8 \
    data.val_max_samples=32 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.global_std=True \
    algorithm.train_timestep_range="[0,7]" \
    algorithm.train_timestep_count=4 \
    algorithm.old_policy_decay_schedule=linear_to_0_3 \
    algorithm.old_policy_update_interval=1 \
    actor_rollout_ref.model.algorithm=dgpo \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 256}}' \
    actor_rollout_ref.model.attn_backend=$ATTN_BACKEND \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.dgpo_beta=100.0 \
    actor_rollout_ref.actor.diffusion_loss.dgpo_clip_range=0.01 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.02 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$N_RESP_PER_PROMPT \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=true \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.5 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.max_prompt_embed_length=333 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=$((NUM_GPUS_REWARD / REWARD_TP)) \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=$NUM_GPUS_REWARD \
    reward.reward_manager.name=MultiVisualRewardManager \
    +reward.models.ocr.backend=engine \
    +reward.models.ocr.offload=False \
    +reward.models.ocr.model_path=$reward_model_name \
    +reward.models.ocr.n_gpus_per_node=$NUM_GPUS_REWARD \
    +reward.models.ocr.nnodes=1 \
    +reward.models.ocr.rollout.name=$REWARD_ENGINE \
    +reward.models.ocr.rollout.gpu_memory_utilization=0.9 \
    +reward.models.ocr.rollout.tensor_model_parallel_size=$REWARD_TP \
    +reward.models.ocr.rollout.enforce_eager=False \
    +reward.reward_functions.ocr.path=$reward_function_path \
    +reward.reward_functions.ocr.name=compute_score_ocr \
    +reward.reward_functions.ocr.weight=1.0 \
    +reward.reward_functions.ocr.required=True \
    +reward.reward_functions.ocr.use_rollout_sampling_params=True \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=dgpo \
    trainer.experiment_name=sd35_medium_ocr_lora_v1 \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    trainer.v1.sampler.drop_incomplete_groups=True "$@"
