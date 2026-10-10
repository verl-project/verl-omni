#!/usr/bin/env bash
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

bash "$recipe_dir/run_bagel_dvreward_lora.sh" \
    actor_rollout_ref.model.algorithm=alphagrpo \
    actor_rollout_ref.model.model_type=diffusion_alphagrpo_model \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.trainer_type=policy_gradient \
    algorithm.adv_estimator=flow_grpo \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj','q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=2048 \
    actor_rollout_ref.rollout.pipeline.max_think_tokens=512 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=16 \
    actor_rollout_ref.rollout.algo.sde_window_size=10 \
    actor_rollout_ref.rollout.algo.sde_window_range='[0,11]' \
    reward.custom_reward_function.name=compute_score_alphagrpo \
    trainer.experiment_name=bagel_alphagrpo_lora \
    "$@"
