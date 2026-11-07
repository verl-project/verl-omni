#!/usr/bin/env bash
# Run from the repository root; requires the prepared AVQA dataset and four GPUs.
set -euo pipefail

PROFILE_DIR=${PROFILE_DIR:-./outputs/profile_qwen3_omni_avqa_v1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$SCRIPT_DIR/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh" \
    data.train_batch_size=2 \
    data.max_response_length=2048 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    trainer.experiment_name=qwen3_omni_avqa_v1_profile \
    trainer.total_training_steps=4 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    'trainer.logger=[console]' \
    global_profiler.tool=torch \
    'global_profiler.steps=[3]' \
    global_profiler.profile_continuous_steps=False \
    "global_profiler.save_path=$PROFILE_DIR" \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=False \
    'actor_rollout_ref.actor.profiler.ranks=[0]' \
    actor_rollout_ref.actor.profiler.tool=torch \
    'actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda]' \
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.all_ranks=False \
    'actor_rollout_ref.rollout.profiler.ranks=[0]' \
    actor_rollout_ref.rollout.profiler.tool=torch \
    'actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cpu,cuda]' \
    actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True \
    "$@"
