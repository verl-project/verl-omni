#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

BASE_ID="${BASE_ID:-pruned_vllm_omni_replica_compare_${RUN_STAMP}}"

echo "[info] phase=replica1 BASE_ID=${BASE_ID}"
RUN_ID="${BASE_ID}_replica1" \
  "${SCRIPT_DIR}/run_qwen3_omni_vllm_omni_pruned_replica1_standalone_logprob_probe.sh" "$@"

echo "[info] phase=replica4 BASE_ID=${BASE_ID}"
RUN_ID="${BASE_ID}_replica4" \
  "${SCRIPT_DIR}/run_qwen3_omni_vllm_omni_pruned_replica4_standalone_logprob_probe.sh" "$@"
