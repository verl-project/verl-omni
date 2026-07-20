#!/usr/bin/env bash
# Corr smoke for pruned Qwen3-Omni Megatron fully-async training.
# Disable bypass so actor recomputes old_log_probs and compares them with
# rollout_log_probs via training/rollout_actor_probs_pearson_corr.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-corr_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_fully_async_smoke.sh" \
    algorithm.rollout_correction.bypass_mode=False \
    "$@"
