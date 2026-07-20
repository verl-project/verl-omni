#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs"

export RUN_ID="${RUN_ID:-pruned_vllm_omni_replica4_standalone_logprob_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_PATH="${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-smoke-chattemplate}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp1_dp4_pruned_async_raw_logprobs.yaml}"
export ROW_INDICES="${ROW_INDICES:-0,1,2,3,4,5,6,7}"
export MAX_TOKENS="${MAX_TOKENS:-64}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-384}"
export CONCURRENCY="${CONCURRENCY:-4}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-24}"

export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-32}"
export VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT:-16}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS="${VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS:-1}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL="${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL:-${LOG_ROOT}/vllm_omni_logprob_debug/${RUN_ID}.jsonl}"

echo "[info] replica4 standalone vLLM-Omni debug JSONL prefix: ${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL}"
exec "${SCRIPT_DIR}/run_qwen3_omni_vllm_omni_full_tp4_standalone_logprob_probe.sh" "$@"
