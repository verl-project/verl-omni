#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_OMNI_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STACK_ROOT="$(cd "${VERL_OMNI_REPO}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async}"
source "${CONDA_ENV}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${VERL_OMNI_REPO}:${STACK_ROOT}/verl:${STACK_ROOT}/vllm-omni:${STACK_ROOT}/megatron-bridge/src:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

CACHE_ROOT="${CACHE_ROOT:-/nfs/ml-training-ssd/users/liuwei/verl_omni_probe_cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

if [[ $# -lt 1 && -z "${DUMP_JSONL:-}" ]]; then
  echo "usage: $0 /path/to/*.rollout_corr_samples.jsonl" >&2
  echo "   or: DUMP_JSONL=/path/to/dump.jsonl $0" >&2
  exit 2
fi

if [[ -z "${DUMP_JSONL:-}" ]]; then
  DUMP_JSONL="$1"
  shift
fi
MODEL_PATH="${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate}"
RECORD_LIMIT="${RECORD_LIMIT:-4}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-32}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
POSITION_IDS_MODE="${POSITION_IDS_MODE:-none}"
RUN_ID="${RUN_ID:-hf_score_rollout_corr_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes}"
LOG_DIR="${LOG_DIR:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUT_DIR}/${RUN_ID}.hf_score.jsonl}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"

python3 "${SCRIPT_DIR}/qwen3_omni_hf_score_rollout_corr_dump.py" \
  --dump-jsonl "${DUMP_JSONL}" \
  --model-path "${MODEL_PATH}" \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  --position-ids-mode "${POSITION_IDS_MODE}" \
  --record-limit "${RECORD_LIMIT}" \
  --sample-limit "${SAMPLE_LIMIT}" \
  --output-file "${OUTPUT_FILE}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

echo "LOG_FILE=${LOG_FILE}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
