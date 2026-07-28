#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA training on MMK12 with omni V1 trainer.
# Data preparation (run once):
#   pip install math-verify
#   python examples/gspo_trainer/data_process/mmk12.py \
#       --local_dataset_path <path_to_raw_mmk12> \
#       --local_save_dir ~/data/mmk12
#
# Runtime dependencies (all Ray worker nodes):
#   pip install math-verify    # required by mmk12_reward.py
#   pip install qwen-vl-utils  # required for multimodal data processing

set -x

# Make verl_omni available to Ray workers
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/mmk12/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/mmk12/test.parquet"}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=gspo \
    trainer.experiment_name=qwen3_omni_thinker_lora_mmk12 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@"
