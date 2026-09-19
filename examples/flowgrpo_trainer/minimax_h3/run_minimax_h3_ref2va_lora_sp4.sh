#!/usr/bin/env bash
# Four-GPU MiniMax H3 Ref2VA LoRA FlowGRPO with Actor SP=4 and 1,000 training steps.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(readlink -f "$script_dir/../../..")

export N_GPUS=4
export ACTOR_SP=4
export ROLLOUT_TP=4
export TEXT_ENCODER_TP=4
export TOTAL_TRAINING_STEPS=1000
export OUTPUT_DIR=${OUTPUT_DIR:-$repo_root/outputs/minimax_h3_ref2va_lora_sp4}

exec "$script_dir/run_minimax_h3_ref2va_lora.sh" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}" \
    "$@"
