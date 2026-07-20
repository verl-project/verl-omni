#!/usr/bin/env bash
set -xeuo pipefail

# Replay fixed rollout-corr JSONL rows through the 4-node Megatron actor/ref
# scorer only. This keeps the multi-node Megatron topology while skipping
# vLLM-Omni rollout and weight sync, so it isolates scoring self-consistency.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ID=${RUN_ID:-full32_fixed_sequence_score_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_score_${RUN_ID}}

export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-${LOG_DIR}/rollout_corr_debug/full32_raw_cold_no_update_20260714_112553.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-8}
mkdir -p "$(dirname "${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT}")"

if [[ ! -f "${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL}" ]]; then
  echo "[error] VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL not found: ${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL}" >&2
  exit 1
fi

export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT=${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL=${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-}
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS=${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-0}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL=${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.bshd_debug}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-2}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS:-2}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS:-16}

# The fixed-sequence probe only needs Megatron trainer workers. Keep Ray worker
# ports out of the high ephemeral range and require a completely free block.
# Ray assigns CoreWorker gRPC listeners from this range, so a single occupied
# port can make the driver fail before the probe reaches Megatron.
export RAY_WORKER_PORT_POOL_START=${RAY_WORKER_PORT_POOL_START:-30000}
export RAY_WORKER_PORT_POOL_END=${RAY_WORKER_PORT_POOL_END:-49999}
export RAY_WORKER_PORT_SPAN=${RAY_WORKER_PORT_SPAN:-256}
export RAY_WORKER_PORT_MIN_FREE=${RAY_WORKER_PORT_MIN_FREE:-${RAY_WORKER_PORT_SPAN}}
export VERL_OMNI_SKIP_WEIGHT_UPDATE=${VERL_OMNI_SKIP_WEIGHT_UPDATE:-1}
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS=${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-1}

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh" "$@"
