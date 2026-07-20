#!/usr/bin/env bash
set -euo pipefail

# Full-model B-gate for Megatron -> vLLM-Omni weight sync.
# Runs two tiny fully-async steps:
#   step 0: cold rollout, actor update, real Megatron -> vLLM update
#   step 1: rollout after that update, then corr metrics expose post-sync drift
#
# Keep the real full topology from the reference launcher:
#   train Megatron: 16 GPUs, TP2 PP2 EP4
#   rollout vLLM-Omni: 16 GPUs, four standalone TP4 replicas

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_weight_sync_b_gate_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_weight_sync_b_gate_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel weight-sync B-gate}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp4_full_async_no_sleep_raw_logprobs.yaml}"

# Keep the completed probe topology, but use the dynamic per-actor vLLM port
# slots that survived the 2026-07-03 B-gate run.  Replaying a single fixed
# VLLM_PORT=61512 is useful for forensics, but fragile on shared Luban hosts.
export VERL_OMNI_RESPECT_EXISTING_VLLM_PORT="${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT:-0}"
if [[ "${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT}" != "1" ]]; then
  unset VLLM_PORT
fi
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))}"

export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-4}"
export ROLLOUT_CORR_BYPASS_MODE="${ROLLOUT_CORR_BYPASS_MODE:-False}"
export ROLLOUT_CALCULATE_LOG_PROBS="${ROLLOUT_CALCULATE_LOG_PROBS:-True}"
export ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-raw_logprobs}"
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-1}"
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}"
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-0}"

# This is B, so real update must stay enabled.
export VERL_OMNI_SKIP_WEIGHT_UPDATE=0

# Leave sleep/free-cache disabled through the base launcher. Enable existing
# weight-sync diagnostics without turning on very heavy value dumps by default.
export VERL_OMNI_WEIGHT_SYNC_DEBUG="${VERL_OMNI_WEIGHT_SYNC_DEBUG:-1}"
export VERL_OMNI_WEIGHT_ROUTE_DEBUG="${VERL_OMNI_WEIGHT_ROUTE_DEBUG:-1}"
export VERL_OMNI_WEIGHT_VALUE_DEBUG="${VERL_OMNI_WEIGHT_VALUE_DEBUG:-0}"
export VERL_OMNI_PRE_IPC_LOCAL_COPY_DEBUG="${VERL_OMNI_PRE_IPC_LOCAL_COPY_DEBUG:-0}"
export VERL_OMNI_PRE_LOAD_LOCAL_COPY_DEBUG="${VERL_OMNI_PRE_LOAD_LOCAL_COPY_DEBUG:-0}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh"
