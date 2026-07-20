#!/usr/bin/env bash
set -euo pipefail

# Logprob parity probe on the known-good 32-GPU async split topology.
# Keep rollout/actor/ref and weight-update semantics aligned with the passing
# B-gate launcher; this wrapper only shortens the run and enables targeted
# rollout-logprob / corr debug.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# This wrapper is meant to be submitted as a real probe. Avoid inheriting
# CONFIG_ONLY=1 from local/static preflight sessions.
export CONFIG_ONLY="${PARITY_CONFIG_ONLY:-0}"
export RAY_ONLY="${PARITY_RAY_ONLY:-0}"

export RUN_ID="${RUN_ID:-full32_logprob_parity_probe_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_logprob_parity_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel logprob parity probe}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp4_full_async_no_sleep_raw_logprobs.yaml}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))}"

export ROLLOUT_CORR_BYPASS_MODE="${ROLLOUT_CORR_BYPASS_MODE:-False}"
export ROLLOUT_CALCULATE_LOG_PROBS="${ROLLOUT_CALCULATE_LOG_PROBS:-True}"
export ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-raw_logprobs}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-1}"
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}"
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}"
export VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT="${VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs/rollout_corr_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-8}"

# Match the completed B-gate by default. If we need a no-update isolate, submit
# with VERL_OMNI_SKIP_WEIGHT_UPDATE=1 explicitly instead of changing the
# baseline probe.
export VERL_OMNI_SKIP_WEIGHT_UPDATE="${VERL_OMNI_SKIP_WEIGHT_UPDATE:-0}"
export VERL_OMNI_WEIGHT_SYNC_DEBUG="${VERL_OMNI_WEIGHT_SYNC_DEBUG:-1}"
export VERL_OMNI_WEIGHT_ROUTE_DEBUG="${VERL_OMNI_WEIGHT_ROUTE_DEBUG:-1}"
export VERL_OMNI_WEIGHT_VALUE_DEBUG="${VERL_OMNI_WEIGHT_VALUE_DEBUG:-0}"

export VERL_OMNI_RESPECT_EXISTING_VLLM_PORT="${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT:-0}"
if [[ "${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT}" != "1" ]]; then
  unset VLLM_PORT
fi
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-0}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh"
