#!/usr/bin/env bash
# Gate D2: one local 4-GPU AudioMCQ batch with target n=2, reward enabled,
# one actor update, rollout weight synchronization, and fail-closed acceptance.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-audiomcq_pruned_4gpu_gate_d2_$(date +%Y%m%d_%H%M%S)}"
export VERL_OMNI_SKIP_WEIGHT_UPDATE=0
export TOTAL_ROLLOUT_STEPS=2
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS=2

LOCAL_ROOT="${LOCAL_ROOT:-/tmp-data/liuwei/audiomcq_4gpu_gate}"
export LOG_ROOT="${LOG_ROOT:-${LOCAL_ROOT}/logs}"
export LOG_FILE="${LOG_FILE:-${LOG_ROOT}/qwen3_omni_megatron_pruned_fully_async_smoke_${RUN_ID}.log}"
RESULT_FILE="${LOCAL_ROOT}/results/${RUN_ID}.gate_d2.json"

mkdir -p "${LOG_ROOT}" "$(dirname -- "${RESULT_FILE}")"

"${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_audiomcq_4gpu_gate.sh" \
  data.train_max_samples=2 \
  trainer.total_epochs=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  rollout.n=2 \
  +async_training.max_queue_size=2 \
  +async_training.max_concurrent_samples=2 \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_audio_model=True \
  "$@"

exec /nfs/ml-training-ssd/users/liuwei/verl_mega_async/bin/python \
  "${SCRIPT_DIR}/analyze_qwen3_omni_audiomcq_rl_gate.py" \
  "${LOG_FILE}" \
  --allow-zero-reward \
  --json-out "${RESULT_FILE}"
