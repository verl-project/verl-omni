#!/usr/bin/env bash
# Wan2.2 5B full-weight DanceGRPO throughput benchmark for 8 x 180 GB GPUs.
#
# This is an incremental override of run_wan22_5b_t2v_hpsv3_v1.sh. It uses
# FSDP2 for actor training and removes checkpoint and validation work from the
# measured run.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

# The base recipe tees Python output to a log. Propagate pipefail into its shell
# so the profiling wrapper can preserve a real training failure status.
bash -o pipefail "$SCRIPT_DIR/run_wan22_5b_t2v_hpsv3_v1.sh" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    trainer.experiment_name=wan22_5b_t2v_hpsv3_8x180g_v1_fsdp2_benchmark \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    reward.num_workers=8 \
    reward.custom_reward_function.use_accelerator=True \
    "$@"
