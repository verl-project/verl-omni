#!/usr/bin/env bash
set -euo pipefail

# Fixed-sequence direct-old scorer probe with Megatron MoE router replay enabled
# in R2 mode. Purpose: test whether constraining MoE routing restores old
# run1/run2 self-consistency under the 32-GPU TP2/PP2/EP4/SP topology.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ID=${RUN_ID:-full32_fixed_sequence_router_replay_r2_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_router_replay_r2_${RUN_ID}}

export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=1

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_score_direct_old_probe.sh" \
  actor_rollout_ref.actor.router_replay.mode=R2 \
  actor_rollout_ref.actor.megatron.router_replay.mode=R2 \
  "$@"
