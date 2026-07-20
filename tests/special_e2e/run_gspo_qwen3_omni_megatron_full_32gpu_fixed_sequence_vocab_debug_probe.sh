#!/usr/bin/env bash
set -euo pipefail

# Fixed-sequence direct-old scorer probe with Megatron vocab debug enabled.
# Purpose: compare Megatron vocab_parallel_cross_entropy output against the
# existing manual TP-gather logprob calculation in the same forward.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ID=${RUN_ID:-full32_fixed_sequence_vocab_debug_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_vocab_debug_${RUN_ID}}

export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL=${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-${LOG_DIR}/megatron_bshd_debug/${RUN_ID}}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-2}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS:-2}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS:-16}
export VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT:-2}

mkdir -p "$(dirname -- "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}")"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_score_direct_old_probe.sh" "$@"
