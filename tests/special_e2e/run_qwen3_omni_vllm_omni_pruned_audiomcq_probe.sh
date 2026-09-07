#!/usr/bin/env bash
# Fixed real-audio vLLM-Omni Thinker smoke. This intentionally refuses to run
# until the owner-approved AudioMCQ media has been preprocessed into parquet.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_OMNI_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_PATH="${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-smoke-chattemplate}"
export DATA_FILE="${DATA_FILE:-/nfs/ml-training-ssd/users/liuwei/data/audiomcq_verl_v2/train.parquet}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp1_pruned_audiomcq.yaml}"
export ROW_INDICES="${ROW_INDICES:-0}"
export MAX_TOKENS="${MAX_TOKENS:-32}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
export CONCURRENCY="${CONCURRENCY:-1}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-24}"
export PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
export RUN_ID="${RUN_ID:-pruned_audiomcq_$(date +%Y%m%d_%H%M%S)}"
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-verl_omni,verl_omni.models.transformers.qwen3_omni_thinker}"
export PYTHONPATH="${VERL_OMNI_REPO}/tests/special_e2e/runtime_compat:${PYTHONPATH:-}"

export LOG_DIR="${LOG_DIR:-${VERL_OMNI_REPO}/outputs/audiomcq_probe/logs}"
export OUT_DIR="${OUT_DIR:-${VERL_OMNI_REPO}/outputs/audiomcq_probe/results}"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "[blocked] Processed AudioMCQ parquet is absent: ${DATA_FILE}" >&2
  echo "Run examples/data_preprocess/audio_mcq.py against the approved Harland snapshot, then retry." >&2
  exit 2
fi

exec "${SCRIPT_DIR}/run_qwen3_omni_vllm_omni_full_tp4_standalone_logprob_probe.sh" "$@"
