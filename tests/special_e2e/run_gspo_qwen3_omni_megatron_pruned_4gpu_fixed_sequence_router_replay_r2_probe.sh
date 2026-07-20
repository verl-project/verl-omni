#!/usr/bin/env bash
# 4-GPU pruned fixed-sequence Megatron scorer probe with R2 router replay.
#
# Input is a rollout-corr JSONL produced by
# run_gspo_qwen3_omni_megatron_pruned_4gpu_fixed_sequence_dump_probe.sh.
# This path skips vLLM-Omni rollout and weight sync in the scorer run; it only
# checks whether Megatron old run1/run2 becomes self-consistent when replaying
# the recorded MoE routing decisions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
PROBE_ROOT="${PROBE_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ID="${RUN_ID:-pruned4_fixed_sequence_router_replay_r2_${RUN_STAMP}}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_pruned_4gpu_fixed_sequence_router_replay_r2_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-4-GPU pruned fixed-sequence R2 router replay scorer probe}"

export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL="${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-${PROBE_ROOT}/${RUN_ID}.rollout_corr_samples.jsonl}"
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT="${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}"
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS="${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-4}"
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1
export VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=1

if [[ ! -f "${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL}" ]]; then
  echo "[error] VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL not found: ${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL}" >&2
  echo "[hint] First run: ${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_4gpu_fixed_sequence_dump_probe.sh" >&2
  echo "[hint] Then re-run this shell with VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=/path/to/*.rollout_corr_samples.jsonl" >&2
  exit 1
fi

mkdir -p "$(dirname "${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT}")"
echo "[info] fixed-sequence input: ${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL}"
echo "[info] fixed-sequence output: ${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT}"

export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-0}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-0}"
export VERL_OMNI_SKIP_WEIGHT_UPDATE="${VERL_OMNI_SKIP_WEIGHT_UPDATE:-1}"
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS="${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-1}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_fully_async_4gpu_local_gate.sh" \
    actor_rollout_ref.actor.router_replay.mode=R2 \
    actor_rollout_ref.actor.megatron.router_replay.mode=R2 \
    trainer.experiment_name="${EXP_NAME}" \
    "$@"
