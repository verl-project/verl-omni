#!/usr/bin/env bash
set -euo pipefail

# 32-GPU vLLM-Omni multi-stage logprob dump probe.
#
# Keep the already-used raw cold no-update envelope and only add aligned dumps
# at the key logprob boundaries:
#   - vLLM-Omni internal sampler/processed/extraction debug
#   - AgentLoop per-sample and batch postprocess
#   - Trainer after queue, after old_log_prob, after ref_log_prob
#   - Existing rollout-corr summary

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs"

export RUN_ID="${RUN_ID:-full32_vllm_omni_multistage_logprob_dump_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_vllm_omni_multistage_logprob_dump_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel vLLM-Omni multi-stage logprob dump probe}"

export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-16}"
export VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT="${VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-8}"

export VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL="${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL:-${LOG_ROOT}/vllm_omni_logprob_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-${LOG_ROOT}/rollout_corr_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL="${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL:-${LOG_ROOT}/multistage_logprob_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS="${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS:-8}"
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS="${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS:-16}"

# Route the first vLLM StageCore TCPStore through the per-worker MASTER_PORT
# lease to avoid the probe/release race seen on multi-node startup.
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}"

export PARITY_CONFIG_ONLY="${PARITY_CONFIG_ONLY:-${CONFIG_ONLY:-0}}"
export PARITY_RAY_ONLY="${PARITY_RAY_ONLY:-${RAY_ONLY:-0}}"

echo "[info] multistage logprob debug JSONL: ${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL}"
echo "[info] trainer rollout-corr debug JSONL: ${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}"
echo "[info] vLLM-Omni internal logprob debug JSONL: ${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_raw_cold_no_update_probe.sh"
