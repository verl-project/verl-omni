#!/usr/bin/env bash
# Two training steps with the real Lance and PickScore checkpoints on 4 GPUs.
# Keep the training recipe's resolution, denoising schedule and log-prob checks.
set -euo pipefail

export PYTHONUNBUFFERED=1

exec bash "$(dirname "$0")/run_lance_image_edit_pickscore_lora.sh" \
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
    trainer.experiment_name=lance_pickscore_lora_smoke \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=2 \
    "$@"
