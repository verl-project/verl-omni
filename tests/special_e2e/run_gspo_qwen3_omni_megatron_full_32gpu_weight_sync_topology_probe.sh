#!/usr/bin/env bash
set -euo pipefail

# 32-GPU fullmodel async probe focused on the multi-node weight-sync topology.
# Keep the known 4-node same-node split launcher as the base; only shorten the
# run and enable gated diagnostics for:
#   trainer rank0 -> NCCL checkpoint engine -> rollout adapters -> vLLM-Omni IPC receivers.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# This wrapper is meant to be submitted as a real probe. Avoid inheriting local
# static preflight settings.
export CONFIG_ONLY="${TOPO_CONFIG_ONLY:-0}"
export RAY_ONLY="${TOPO_RAY_ONLY:-0}"

export RUN_ID="${RUN_ID:-full32_weight_sync_topology_probe_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_weight_sync_topology_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel weight-sync topology probe}"

# Minimal RL envelope: still exercises rollout -> Megatron update -> real
# Megatron-to-vLLM weight sync, but keeps generation/training payload tiny.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS="${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-1}"
export REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-320}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-4}"

# Preserve the current vLLM-Omni async-safe topology: four TP4 standalone
# replicas over the 16 rollout GPUs, no vLLM internal DP.
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export ROLLOUT_DP="${ROLLOUT_DP:-1}"
export VERL_OMNI_FORCE_STANDALONE_ROLLOUT="${VERL_OMNI_FORCE_STANDALONE_ROLLOUT:-1}"
export VERL_OMNI_RESOURCE_SPLIT_IMPL="${VERL_OMNI_RESOURCE_SPLIT_IMPL:-standalone_rollout}"

# This is the path under test.
export VERL_OMNI_SKIP_WEIGHT_UPDATE="${VERL_OMNI_SKIP_WEIGHT_UPDATE:-0}"
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME="${VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME:-1}"
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP="${VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP:-1}"
export VERL_OMNI_WEIGHT_SYNC_DEBUG="${VERL_OMNI_WEIGHT_SYNC_DEBUG:-1}"
export VERL_OMNI_WEIGHT_SYNC_TOPO_DEBUG="${VERL_OMNI_WEIGHT_SYNC_TOPO_DEBUG:-1}"
export VERL_OMNI_WEIGHT_ROUTE_DEBUG="${VERL_OMNI_WEIGHT_ROUTE_DEBUG:-1}"
export VERL_OMNI_WEIGHT_ROUTE_DEBUG_BUCKETS="${VERL_OMNI_WEIGHT_ROUTE_DEBUG_BUCKETS:-2}"
export VERL_OMNI_WEIGHT_ROUTE_DEBUG_SAMPLE="${VERL_OMNI_WEIGHT_ROUTE_DEBUG_SAMPLE:-8}"

# Keep corr/logprob debug available, but small enough not to bury topology lines.
export ROLLOUT_CORR_BYPASS_MODE="${ROLLOUT_CORR_BYPASS_MODE:-False}"
export ROLLOUT_CALCULATE_LOG_PROBS="${ROLLOUT_CALCULATE_LOG_PROBS:-True}"
export ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-processed_logprobs}"
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-1}"
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-8}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-8}"

# Phase generation is useful around update, but should not fail the topology
# probe by itself.
export VERL_OMNI_PHASE_GENERATION_DEBUG="${VERL_OMNI_PHASE_GENERATION_DEBUG:-1}"
export VERL_OMNI_PHASE_GENERATION_STRICT="${VERL_OMNI_PHASE_GENERATION_STRICT:-0}"
export VERL_OMNI_PHASE_GENERATION_MAX_TOKENS="${VERL_OMNI_PHASE_GENERATION_MAX_TOKENS:-32}"

# Let the base launcher own port selection by default; only respect an explicit
# VLLM_PORT if caller opts in.
export VERL_OMNI_RESPECT_EXISTING_VLLM_PORT="${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT:-0}"
if [[ "${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT}" != "1" ]]; then
  unset VLLM_PORT
fi
# Match the last known-good 32-GPU envelope: let stage-core use its own
# allocator slice instead of forcing the first torch TCPStore port to the
# worker MASTER_PORT.
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-0}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh" "$@"
