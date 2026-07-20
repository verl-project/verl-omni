#!/usr/bin/env bash
# 4-GPU local multi-replica rollout logprob parity probe.
#
# This keeps the trainer on Megatron and uses two standalone vLLM-Omni rollout
# replicas on the two rollout GPUs (TP1 x DP2). It isolates the 32-GPU
# multi-replica rollout path from multi-node scheduling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-dp2_logprob_parity_$(date +%Y%m%d_%H%M%S)}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp1_pruned_async_raw_logprobs.yaml}"

export ROLLOUT_TP=1
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}"
export VERL_OMNI_SKIP_WEIGHT_UPDATE="${VERL_OMNI_SKIP_WEIGHT_UPDATE:-0}"
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS="${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-1}"
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME="${VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME:-1}"
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP="${VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP:-1}"

export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS=1
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-24}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-24}"

export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-2}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_fully_async_4gpu_local_gate.sh" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    algorithm.rollout_correction.bypass_mode=False \
    trainer.experiment_name="qwen3_omni_megatron_pruned_4gpu_dp2_logprob_parity_${RUN_ID}" \
    "$@"
