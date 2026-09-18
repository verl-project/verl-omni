#!/bin/bash
# Qwen-Image full-weight RL throughput benchmark for 8 x 80 GB GPUs.
#
# This is an incremental override of run_qwen_image_ocr.sh. FSDP2 is required
# because FSDP1 runs out of memory on this configuration. Rollout step execution
# and reward-model CUDA graphs/batching are enabled to improve throughput. This
# benchmark disables checkpoint saving and periodic validation.
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
REWARD_TP=1

NUM_GPUS=$NUM_GPUS NUM_NODES=$NUM_NODES bash "$SCRIPT_DIR/run_qwen_image_ocr.sh" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.step_execution=True \
    reward.num_workers=$((NUM_GPUS / REWARD_TP)) \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.enforce_eager=False \
    reward.reward_model.rollout.max_num_seqs=128 \
    reward.reward_model.rollout.max_model_len=8192 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.experiment_name=qwen_image_ocr_8x80g_fsdp2_benchmark \
    trainer.resume_mode=disable \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    "$@"
