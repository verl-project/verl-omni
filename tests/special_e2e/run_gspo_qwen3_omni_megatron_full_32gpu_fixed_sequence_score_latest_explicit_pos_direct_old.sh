#!/usr/bin/env bash
set -euo pipefail

# Replay the latest NeMo-style gate dump through Megatron actor/ref scoring only.
# This is paired with latest_model_pos_direct_old and changes only the Qwen3-Omni
# BSHD position-id contract: Megatron consumes verl-provided position_ids instead
# of letting the model build text-only mRoPE ids internally.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ID=${RUN_ID:-full32_fixed_sequence_latest_explicit_pos_direct_old_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_latest_explicit_pos_direct_old_${RUN_ID}}

export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-${LOG_DIR}/rollout_corr_debug/full32_vllm_omni_multi_probe_20260716_123444.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-4}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1
export VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS=explicit
export VERL_OMNI_QWEN3_OMNI_SP_SCATTER_POSITION_IDS=0

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_score_probe.sh" "$@"
