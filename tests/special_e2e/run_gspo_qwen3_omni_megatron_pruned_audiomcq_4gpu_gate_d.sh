#!/usr/bin/env bash
# Gate D1: one real AudioMCQ prompt, one response, one language-model update,
# and one actor-to-rollout parameter synchronization. The audio tower is frozen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-audiomcq_pruned_4gpu_gate_d_$(date +%Y%m%d_%H%M%S)}"
export VERL_OMNI_SKIP_WEIGHT_UPDATE=0
export TOTAL_ROLLOUT_STEPS=2
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-1}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_audiomcq_4gpu_gate.sh" \
  data.train_max_samples=1 \
  trainer.total_epochs=2 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  rollout.n=1 \
  +async_training.max_concurrent_samples=1 \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_audio_model=True \
  "$@"
