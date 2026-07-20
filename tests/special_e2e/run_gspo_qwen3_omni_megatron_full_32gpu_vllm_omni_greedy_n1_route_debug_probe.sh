#!/usr/bin/env bash
set -euo pipefail

# 32-GPU raw-logprobs greedy control.  With n=1 and temperature=0, processed
# sampling semantics tend to collapse selected-token logprobs toward 0; raw
# model logprobs should still be nontrivial.  This helps detect an accidental
# processed/sampling-space value leaking through the raw path.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_vllm_omni_greedy_n1_route_debug_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_vllm_omni_greedy_n1_route_debug_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel vLLM-Omni greedy n=1 raw route debug probe}"

export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
export REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))}"

export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-2}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
export OMNI_LB_POLICY="${OMNI_LB_POLICY:-round-robin}"

export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-16}"
export VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL="${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs/vllm_omni_logprob_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}"
export VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT="${VERL_OMNI_AGENT_LOOP_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs/rollout_corr_debug/${RUN_ID}.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-8}"

export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}"

export PARITY_CONFIG_ONLY="${PARITY_CONFIG_ONLY:-${CONFIG_ONLY:-0}}"
export PARITY_RAY_ONLY="${PARITY_RAY_ONLY:-${RAY_ONLY:-0}}"

echo "[info] trainer rollout-corr debug JSONL: ${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}"
echo "[info] vLLM-Omni internal logprob debug JSONL: ${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_raw_cold_no_update_probe.sh"
