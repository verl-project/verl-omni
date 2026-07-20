#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_OMNI_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STACK_ROOT="$(cd "${VERL_OMNI_REPO}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async}"
if [[ ! -f "${CONDA_ENV}/bin/activate" ]]; then
  echo "[error] CONDA_ENV not found: ${CONDA_ENV}" >&2
  exit 1
fi
source "${CONDA_ENV}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${VERL_OMNI_REPO}:${STACK_ROOT}/verl:${STACK_ROOT}/vllm-omni:${STACK_ROOT}/megatron-bridge/src:${PYTHONPATH:-}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VERL_OMNI_SKIP_PIPELINES="${VERL_OMNI_SKIP_PIPELINES:-1}"
export VERL_OMNI_SKIP_REWARD_LOOP="${VERL_OMNI_SKIP_REWARD_LOOP:-1}"
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-16}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

CACHE_ROOT="${CACHE_ROOT:-/nfs/ml-training-ssd/users/liuwei/verl_omni_probe_cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

MODEL_PATH="${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate}"
DATA_FILE="${DATA_FILE:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/train.parquet}"
STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp4_full_async_no_sleep_raw_logprobs.yaml}"
ROW_INDICES="${ROW_INDICES:-0}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
STAGE_INIT_TIMEOUT="${STAGE_INIT_TIMEOUT:-1800}"
INIT_TIMEOUT="${INIT_TIMEOUT:-1800}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
LOGPROBS="${LOGPROBS:-1}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-32}"
IGNORE_EOS="${IGNORE_EOS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

RUN_ID="${RUN_ID:-full_tp4_standalone_logprob_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs}"
OUT_DIR="${OUT_DIR:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
OUTPUT_FILE="${OUT_DIR}/${RUN_ID}.jsonl"

cmd=(
  python3 "${SCRIPT_DIR}/qwen3_omni_vllm_omni_standalone_logprob_probe.py"
  --model-path "${MODEL_PATH}"
  --stage-config "${STAGE_CONFIG}"
  --data-file "${DATA_FILE}"
  --row-indices "${ROW_INDICES}"
  --max-model-len "${MAX_MODEL_LEN}"
  --stage-init-timeout "${STAGE_INIT_TIMEOUT}"
  --init-timeout "${INIT_TIMEOUT}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --logprobs "${LOGPROBS}"
  --prompt-logprobs "${PROMPT_LOGPROBS}"
  --concurrency "${CONCURRENCY}"
  --sample-limit "${SAMPLE_LIMIT}"
  --output-file "${OUTPUT_FILE}"
)
if [[ "${IGNORE_EOS}" == "1" || "${IGNORE_EOS}" == "true" || "${IGNORE_EOS}" == "True" ]]; then
  cmd+=(--ignore-eos)
fi
cmd+=("$@")

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"

echo "LOG_FILE=${LOG_FILE}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
