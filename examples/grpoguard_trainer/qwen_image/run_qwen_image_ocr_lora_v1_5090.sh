#!/usr/bin/env bash
# Three-GPU candidate for Qwen-Image GRPO-Guard with CPU offload.
set -e

export NUM_GPUS_ACTOR_ROLLOUT_REWARD=2
export ROLLOUT_TP=2
export REWARD_TP=1

bash "$(dirname "$0")/run_qwen_image_ocr_lora_v1.sh" \
    data.train_batch_size=4 \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.height=512 \
    actor_rollout_ref.rollout.pipeline.width=512 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.step_execution=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_layerwise_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.num_workers=1 \
    reward.reward_model.enable_resource_pool=true \
    reward.reward_model.n_gpus_per_node=1 \
    reward.reward_model.nnodes=1 \
    reward.reward_model.rollout.max_model_len=8192 \
    reward.reward_model.rollout.gpu_memory_utilization=0.85 \
    reward.reward_model.rollout.free_cache_engine=false \
    trainer.experiment_name=qwen_image_ocr_lora_v1_5090 \
    trainer.val_before_train=true \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_training_steps=100 "$@"
