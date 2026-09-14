#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO on MMK12: VeOmni FSDP2/EP actor + vLLM-Omni rollout.
# Install VeOmni 0.1.12 using docs/start/install.md and prepare MMK12 as in
# examples/gspo_trainer/README.md. All Ray nodes need the same environment.
set -euo pipefail

export VERL_USE_EXTERNAL_MODULES=verl_omni
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/mmk12/train.parquet}
VAL_FILE=${VAL_FILE:-${HOME}/data/mmk12/test.parquet}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-verl_omni/utils/reward_score/mmk12_reward.py}
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-2}
ACTOR_EP=${ACTOR_EP:-8}
# The 30B model's MoE intermediate size (768) permits rollout TP=1 or 2.
ROLLOUT_TP=${ROLLOUT_TP:-2}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
MOE_IMPL=${MOE_IMPL:-fused}

python3 -m verl_omni.trainer.main_omni \
    model_engine=veomni \
    actor_rollout_ref.actor._target_=verl_omni.workers.config.omni.OmniVeOmniActorConfig \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.veomni.expert_parallel_size="${ACTOR_EP}" \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.veomni.attn_implementation="${ATTN_IMPL}" \
    actor_rollout_ref.actor.veomni.moe_implementation="${MOE_IMPL}" \
    actor_rollout_ref.actor.veomni.param_offload=true \
    actor_rollout_ref.actor.veomni.optimizer_offload=true \
    actor_rollout_ref.ref.veomni.expert_parallel_size="${ACTOR_EP}" \
    actor_rollout_ref.ref.veomni.ulysses_parallel_size=1 \
    actor_rollout_ref.ref.veomni.attn_implementation="${ATTN_IMPL}" \
    actor_rollout_ref.ref.veomni.moe_implementation="${MOE_IMPL}" \
    actor_rollout_ref.ref.veomni.param_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=true \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=gspo \
    trainer.experiment_name=qwen3_omni_thinker_veomni_mmk12 \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl_omni \
    "$@"
