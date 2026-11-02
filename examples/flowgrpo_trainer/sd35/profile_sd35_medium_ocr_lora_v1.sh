#!/usr/bin/env bash
# Run from the repository root; requires the SD3.5 OCR dataset and three CUDA GPUs.
set -euo pipefail

PROFILE_DIR=${PROFILE_DIR:-./outputs/profile_sd35_v1}
MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-3.5-medium}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$SCRIPT_DIR/run_sd35_medium_ocr_lora_v1.sh" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,4]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    trainer.experiment_name=sd35_medium_ocr_lora_v1_profile \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    'trainer.logger=[console]' \
    global_profiler.tool=torch \
    'global_profiler.steps=[1]' \
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
