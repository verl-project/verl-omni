#!/usr/bin/env bash
# Two training steps with the real Lance video and ImageBind checkpoints on 4 GPUs.
# Keeps the video-edit recipe's schedule and log-prob comparison; shrinks the
# batch and skips validation and saving so it finishes in minutes.
set -euo pipefail

export PYTHONUNBUFFERED=1

# Denoising schedule for the smoke; the generated video's shape follows the
# reference's bucketed shape, so the declared shape only mirrors the recipe.
export VIDEO_HEIGHT=${VIDEO_HEIGHT:-480}
export VIDEO_WIDTH=${VIDEO_WIDTH:-768}
export NUM_FRAMES=${NUM_FRAMES:-25}

exec bash "$(dirname "$0")/run_lance_video_edit_imagebind_lora.sh" \
    data.train_batch_size=4 \
    data.val_max_samples=4 \
    data.val_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN \
    trainer.logger=console \
    trainer.experiment_name=lance_video_edit_imagebind_smoke \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=2 \
    "$@"
