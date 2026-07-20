#!/usr/bin/env bash
# 4-GPU cold logprob parity probe for pruned Qwen3-Omni Megatron + vLLM-Omni.
#
# Purpose:
#   Verify that a cold-loaded vLLM-Omni AR rollout model and the Megatron actor
#   assign comparable logprobs to the same sampled response tokens before any
#   trainer -> rollout weight update happens.
#
# This intentionally avoids the 32-GPU path and avoids weight sync. If this
# fails, the next target is token/logprob/mask or cold-load model parity, not
# Megatron -> vLLM update.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-cold_logprob_parity_$(date +%Y%m%d_%H%M%S)}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp2_pruned_async_raw_logprobs.yaml}"

# Cold parity must not perform the initial fully-async param sync.
export VERL_OMNI_SKIP_WEIGHT_UPDATE=1
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS="${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-1}"
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME="${VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME:-1}"
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP="${VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP:-1}"

# Force the rollout side to load real pruned weights. The older local gate uses
# dummy rollout weights by default, which is useful for startup but invalid for
# parity.
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"

# Make rollout logprobs comparable with actor recomputation.
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS=1
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-24}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-24}"

# Keep the probe small and deterministic-ish. We still compare logprobs on the
# actually sampled tokens, but avoid top-p/temperature processors changing the
# sampled-token probability semantics.
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-2}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_fully_async_4gpu_local_gate.sh" \
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    algorithm.rollout_correction.bypass_mode=False \
    trainer.experiment_name="qwen3_omni_megatron_pruned_4gpu_cold_logprob_parity_${RUN_ID}" \
    "$@"
