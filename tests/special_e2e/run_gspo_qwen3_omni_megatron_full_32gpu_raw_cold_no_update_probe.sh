#!/usr/bin/env bash
set -euo pipefail

# 32-GPU cold-start logprob parity isolate.
# Submit this shell directly when the only goal is to inspect the first
# rollout/actor/ref logprob alignment before any Megatron -> vLLM weight update.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_raw_cold_no_update_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_raw_cold_no_update_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU raw cold logprob parity no-update probe}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"

export ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-raw_logprobs}"
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-1}"
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}"
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}"
export VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT="${VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs/rollout_corr_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-8}"

export VERL_OMNI_SKIP_WEIGHT_UPDATE="${VERL_OMNI_SKIP_WEIGHT_UPDATE:-1}"
export VERL_OMNI_WEIGHT_SYNC_DEBUG="${VERL_OMNI_WEIGHT_SYNC_DEBUG:-1}"
export VERL_OMNI_WEIGHT_ROUTE_DEBUG="${VERL_OMNI_WEIGHT_ROUTE_DEBUG:-1}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_logprob_parity_probe.sh"
