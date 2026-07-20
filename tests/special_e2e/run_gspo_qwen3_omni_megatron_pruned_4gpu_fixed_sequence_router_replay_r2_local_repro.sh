#!/usr/bin/env bash
# One-shell 4-GPU pruned local repro for fixed-sequence R2 router replay.
#
# Phase 1: run pruned Megatron + vLLM-Omni parity and dump rollout-corr rows.
# Phase 2: replay the dumped rows through Megatron scorer with R2 routing
# replay, checking old run1/run2 self-consistency before spending 32 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_ROOT="${PROBE_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${PROBE_ROOT}" "${LOG_DIR}/fixed_sequence_score"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
REPRO_ID="${REPRO_ID:-pruned4_fixed_sequence_r2_local_repro_${RUN_STAMP}}"
DUMP_RUN_ID="${DUMP_RUN_ID:-${REPRO_ID}_dump}"
SCORE_RUN_ID="${SCORE_RUN_ID:-${REPRO_ID}_score_r2}"
FIXED_SEQUENCE_JSONL="${FIXED_SEQUENCE_JSONL:-${PROBE_ROOT}/${REPRO_ID}.rollout_corr_samples.jsonl}"
FIXED_SEQUENCE_SCORE_OUTPUT="${FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${SCORE_RUN_ID}.jsonl}"

echo "[info] local repro id: ${REPRO_ID}"
echo "[info] phase1 dump JSONL: ${FIXED_SEQUENCE_JSONL}"
echo "[info] phase2 score JSONL: ${FIXED_SEQUENCE_SCORE_OUTPUT}"

RUN_ID="${DUMP_RUN_ID}" \
VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${FIXED_SEQUENCE_JSONL}" \
"${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_4gpu_fixed_sequence_dump_probe.sh" "$@"

if [[ ! -s "${FIXED_SEQUENCE_JSONL}" ]]; then
  echo "[error] fixed-sequence dump is empty or missing: ${FIXED_SEQUENCE_JSONL}" >&2
  exit 1
fi

RUN_ID="${SCORE_RUN_ID}" \
VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL="${FIXED_SEQUENCE_JSONL}" \
VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT="${FIXED_SEQUENCE_SCORE_OUTPUT}" \
"${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_4gpu_fixed_sequence_router_replay_r2_probe.sh" "$@"
