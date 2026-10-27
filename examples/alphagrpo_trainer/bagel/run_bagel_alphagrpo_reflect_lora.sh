#!/usr/bin/env bash
set -euo pipefail

: "${REWARD_GPUS:?Set REWARD_GPUS for the separate judge pool; these GPUs are additional to actor/rollout GPUs}"
if ! [[ "$REWARD_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REWARD_GPUS must be a positive integer" >&2
    exit 1
fi
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

bash "$recipe_dir/run_bagel_alphagrpo_lora.sh" \
    trainer.use_v1=false \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.pipelines.bagel_alphagrpo.reflection_agent.BagelReflectionAgentLoopManager \
    "+actor_rollout_ref.rollout.agent.agent_loop_config_path=$recipe_dir/reflection_agent.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=bagel_reflection_agent \
    actor_rollout_ref.rollout.n=14 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=8192 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.algo.sde_window_size=5 \
    actor_rollout_ref.rollout.algo.sde_window_range='[0,15]' \
    algorithm.global_std=True \
    actor_rollout_ref.actor.diffusion_loss.text_loss_weight=0.2 \
    reward.custom_reward_function.name=compute_score_dvreward \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.nnodes=1 \
    "reward.reward_model.n_gpus_per_node=$REWARD_GPUS" \
    "reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_GPUS" \
    reward.num_workers=1 \
    trainer.experiment_name=bagel_alphagrpo_reflect_lora \
    "$@"
