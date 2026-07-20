#!/usr/bin/env bash
set -euo pipefail

# 32-GPU Megatron-only R2 audit on the same rows as the 20260716 model-position
# control. Run1 records MoE routes; run2 must replay them or fail loudly.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ID=${RUN_ID:-full32_fixed_sequence_latest_model_pos_r2_audit_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_latest_model_pos_r2_audit_${RUN_ID}}

export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-${LOG_DIR}/rollout_corr_debug/full32_vllm_omni_multi_probe_20260716_123444.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-4}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1
export VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=1
export VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS=model
export VERL_OMNI_QWEN3_OMNI_SP_SCATTER_POSITION_IDS=0

# Capture enough events to compare the first score, R2 replay score, and ref.
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-4}
export VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT:-4}

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_score_probe.sh" \
  actor_rollout_ref.actor.router_replay.mode=R2 \
  actor_rollout_ref.actor.megatron.router_replay.mode=R2 \
  "$@"
