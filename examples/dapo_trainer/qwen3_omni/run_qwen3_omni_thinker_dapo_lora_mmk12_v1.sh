#!/usr/bin/env bash
# Qwen3-Omni Thinker DAPO + LoRA training on MMK12 with dynamic sampling.
#
# Phase 4 (#446): the AVQA DAPO recipes (Phase 1/2/3) applied to a second
# modality/dataset, MMK12, reusing the existing GSPO MMK12 data pipeline and
# reward scorer. Same token-level DAPO policy contract as the AVQA dynamic
# sampling launcher: vanilla clip-higher, token-mean, GRPO advantages, no KL,
# algorithm.filter_groups enabled on the streaming `dapo` reward manager, and
# the overlong response-length buffer from Phase 2.
#
# Data preparation (run once):
#   pip install math-verify
#   python examples/gspo_trainer/data_process/mmk12.py \
#       --local_dataset_path <path_to_raw_mmk12> \
#       --local_save_dir ~/data/mmk12
#
# Runtime dependencies (all Ray worker nodes):
#   pip install math-verify    # required by mmk12_reward.py
#   pip install qwen-vl-utils  # required for multimodal data processing

set -xeuo pipefail

# Make verl_omni available to Ray workers.
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
    data.shuffle=true \
    data.seed=42 \
    data.val_max_samples=-1 \
    data.validation_shuffle=false \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_inflight_gen_batches=2 \
    reward.reward_model.enable=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12_reward.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_kwargs.max_resp_len=12288 \
    reward.reward_kwargs.overlong_buffer_cfg.enable=true \
    reward.reward_kwargs.overlong_buffer_cfg.len=1024 \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    reward.reward_kwargs.overlong_buffer_cfg.log=true \
    trainer.val_before_train=true \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=dapo \
    trainer.experiment_name=qwen3_omni_thinker_lora_mmk12_dynamic_sampling \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@"
