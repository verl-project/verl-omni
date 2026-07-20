#!/usr/bin/env bash
set -xeuo pipefail

# Statistical-scale 32-GPU async-RL E2E. Preserve the validated topology while
# doubling prompt groups per update: 32 prompts x 4 responses = 128 responses.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID=${RUN_ID:-full32_async_e2e_150step_b128_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_async_e2e_150step_b128_${RUN_ID}}
export PROFILE_LABEL=${PROFILE_LABEL:-32-GPU 150-step async-RL E2E 128-response update}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
export TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))}

# Exercise the complete live loop: rollout, reward, actor update, real weight
# sync, and post-sync vLLM-Omni generation. The base launcher owns the checked
# standalone 16/16 resource split, rollout DP=1, and port contract.
export VERL_OMNI_SKIP_PIPELINES=${VERL_OMNI_SKIP_PIPELINES:-0}
export VERL_OMNI_SKIP_REWARD_LOOP=${VERL_OMNI_SKIP_REWARD_LOOP:-0}
export VERL_OMNI_SKIP_WEIGHT_UPDATE=${VERL_OMNI_SKIP_WEIGHT_UPDATE:-0}
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS=${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-0}
export VERL_OMNI_WEIGHT_SYNC_DEBUG=${VERL_OMNI_WEIGHT_SYNC_DEBUG:-1}
export VERL_OMNI_PHASE_GENERATION_DEBUG=${VERL_OMNI_PHASE_GENERATION_DEBUG:-1}
export VERL_OMNI_PHASE_GENERATION_STRICT=${VERL_OMNI_PHASE_GENERATION_STRICT:-0}
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT=${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}

exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh" "$@"
