#!/usr/bin/env bash
set -euo pipefail

# 32-GPU vLLM-Omni multi-probe.
#
# Keep the proven raw cold no-update 32-GPU envelope, and enable several
# orthogonal probes in a single submission:
#   - vLLM-Omni sampler parity: manual raw log_softmax vs sampler output
#   - vLLM-Omni stage route / scheduler slice / processed-output debug
#   - AgentLoop + Trainer multi-stage rollout/old/ref logprob dump
#   - Trainer rollout-corr sampled rows
#   - tiny Megatron BSHD/vocab debug as a guardrail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs"

export RUN_ID="${RUN_ID:-full32_vllm_omni_multi_probe_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_vllm_omni_multi_probe_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel vLLM-Omni multi-probe}"

# Preserve the target condition that already exposed the bug.
export ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-raw_logprobs}"
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-1}"
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}"
export VERL_OMNI_SKIP_WEIGHT_UPDATE="${VERL_OMNI_SKIP_WEIGHT_UPDATE:-1}"

# vLLM-Omni internals: sampler parity + route/slice/processed-output rows.
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-24}"
export VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT:-16}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS="${VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS:-1}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL="${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL:-${LOG_ROOT}/vllm_omni_logprob_debug/${RUN_ID}.jsonl}"

# AgentLoop/Trainer boundary dumps.
export VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT="${VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT:-16}"
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL="${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL:-${LOG_ROOT}/multistage_logprob_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS="${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS:-8}"
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS="${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS:-16}"

# Rollout-corr sampled rows.
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-24}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-${LOG_ROOT}/rollout_corr_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-8}"

# Cheap Megatron scorer guardrail. Keep this tiny so the main signal remains
# vLLM-Omni rollout logprob production.
export VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL="${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-${LOG_ROOT}/megatron_bshd_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT="${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-1}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS="${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS:-1}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS="${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS:-16}"
export VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT="${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT:-1}"

# Use the less racy multi-node StageCore rendezvous path from the passing
# recent probes.
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}"

export PARITY_CONFIG_ONLY="${PARITY_CONFIG_ONLY:-${CONFIG_ONLY:-0}}"
export PARITY_RAY_ONLY="${PARITY_RAY_ONLY:-${RAY_ONLY:-0}}"

echo "[info] RUN_ID=${RUN_ID}"
echo "[info] vLLM-Omni debug JSONL prefix: ${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL}"
echo "[info] multistage debug JSONL: ${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL}"
echo "[info] rollout-corr debug JSONL: ${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}"
echo "[info] Megatron BSHD debug JSONL: ${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_raw_cold_no_update_probe.sh"
