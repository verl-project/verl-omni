#!/usr/bin/env bash
set -euo pipefail

# 32-GPU vLLM-Omni sampler parity probe.
#
# Reuse the last working multi-stage logprob dump envelope and add one
# vLLM-Omni-internal check inside GPUARModelRunner._sample:
#   manual raw log_softmax(sampled_token) vs sampler_output.logprobs_tensors.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs"

export RUN_ID="${RUN_ID:-full32_vllm_omni_sampler_parity_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_vllm_omni_sampler_parity_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel vLLM-Omni sampler parity probe}"

export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-16}"
export VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT:-16}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS="${VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS:-1}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL="${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL:-${LOG_ROOT}/vllm_omni_logprob_debug/${RUN_ID}.jsonl}"

echo "[info] vLLM-Omni sampler parity debug JSONL prefix: ${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL}"
echo "[info] per-process files enabled: ${VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_vllm_omni_multistage_logprob_dump_probe.sh"
