#!/usr/bin/env bash
set -euo pipefail

# 32-GPU vLLM-Omni route/logprob-output debug probe.
#
# Keep the already-used raw cold no-update envelope and only label + bound the
# vLLM-Omni debug volume. The new StagePool debug emits:
#   - vLLM-Omni AR route debug
#   - vLLM-Omni AR processed output debug
# alongside the existing sampler/scheduler/adapter logprob debug.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_vllm_omni_route_debug_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_vllm_omni_route_debug_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel vLLM-Omni route/logprob-output debug probe}"

export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-8}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL="${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs/vllm_omni_logprob_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-8}"
export VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT="${VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT:-8}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs/rollout_corr_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-4}"

# Route the first vLLM StageCore TCPStore through the per-worker MASTER_PORT
# lease. The generic stage-core slice allocator can hit a probe/release race
# during multi-node startup before this debug probe reaches rollout/logprob.
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}"

export PARITY_CONFIG_ONLY="${PARITY_CONFIG_ONLY:-${CONFIG_ONLY:-0}}"
export PARITY_RAY_ONLY="${PARITY_RAY_ONLY:-${RAY_ONLY:-0}}"

echo "[info] trainer rollout-corr debug JSONL: ${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}"
echo "[info] vLLM-Omni internal logprob debug JSONL: ${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_raw_cold_no_update_probe.sh"
