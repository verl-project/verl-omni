#!/usr/bin/env bash
# 4-GPU post-weight-sync logprob parity probe for pruned Qwen3-Omni.
#
# This is the paired probe for cold logprob parity: keep standalone rollout,
# raw rollout logprobs, and real safetensors loading, but allow the
# Megatron -> vLLM-Omni weight update path to run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-post_sync_logprob_parity_$(date +%Y%m%d_%H%M%S)}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp2_pruned_async_raw_logprobs.yaml}"

# This is the only intended behavioral difference from the cold parity probe.
export VERL_OMNI_SKIP_WEIGHT_UPDATE=0
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
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    algorithm.rollout_correction.bypass_mode=False \
    trainer.experiment_name="qwen3_omni_megatron_pruned_4gpu_post_sync_logprob_parity_${RUN_ID}" \
    "$@"
