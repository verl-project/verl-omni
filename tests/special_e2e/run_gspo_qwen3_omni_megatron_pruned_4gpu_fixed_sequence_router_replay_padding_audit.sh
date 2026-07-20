#!/usr/bin/env bash
# 4-GPU pruned R2 gate for THD sequence-parallel router-padding replay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID="${RUN_ID:-pruned4_router_replay_padding_audit_${RUN_STAMP}}"
export REPRO_ID="${REPRO_ID:-${RUN_ID}}"
export SCORE_RUN_ID="${SCORE_RUN_ID:-${RUN_ID}_score}"
export FIXED_SEQUENCE_SCORE_OUTPUT="${FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${SCORE_RUN_ID}.jsonl}"
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT="${FIXED_SEQUENCE_SCORE_OUTPUT}"
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1
export VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=1
export VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL="${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-${FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.bshd_debug}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT="${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-4}"
export VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT=1
export VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT_LIMIT=2
export VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT=1
export VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS="${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS:-1}"
export VERL_OMNI_MEGATRON_MOE_REPLAY_METADATA_AUDIT=1
export VERL_OMNI_MEGATRON_ROUTER_REPLAY_AUDIT=1
export VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE=1
export VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS="${VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS:-1}"

# The defect only exists when Megatron's sequence-parallel TP split recreates
# THD alignment tokens. Keep the 2-train/2-rollout resource split, but make
# the two train GPUs one TP=2 actor/ref group.
export TRAIN_GPUS="${TRAIN_GPUS:-2}"
export ACTOR_TP="${ACTOR_TP:-2}"
export REF_TP="${REF_TP:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-True}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-unfused}"
if [[ "${TRAIN_GPUS}" != "2" || "${ACTOR_TP}" != "2" || "${REF_TP}" != "2" || "${SEQUENCE_PARALLEL,,}" != "true" || "${ATTENTION_BACKEND}" == "local" ]]; then
  echo "[error] padding audit requires TRAIN_GPUS=2, ACTOR_TP=2, REF_TP=2, SEQUENCE_PARALLEL=True, and a Transformer Engine attention backend" >&2
  exit 2
fi

bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_4gpu_fixed_sequence_router_replay_r2_local_repro.sh" "$@"

REPLAY_AUDIT_OUTPUT="${FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.router_replay_compare.json"
INDEX_TRACE_OUTPUT="${FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.router_replay_index_compare.json"
python3 "${SCRIPT_DIR}/qwen3_omni_compare_megatron_router_replay.py" \
  --megatron-decoder-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.decoder.rank*.jsonl" \
  --output-file "${REPLAY_AUDIT_OUTPUT}"
python3 "${SCRIPT_DIR}/qwen3_omni_compare_router_replay_index_trace.py" \
  --trace-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.router_replay_indices.rank*.jsonl" \
  --output-file "${INDEX_TRACE_OUTPUT}"

echo "FIXED_SEQUENCE_SCORE_OUTPUT=${FIXED_SEQUENCE_SCORE_OUTPUT}"
echo "REPLAY_AUDIT_OUTPUT=${REPLAY_AUDIT_OUTPUT}"
echo "INDEX_TRACE_OUTPUT=${INDEX_TRACE_OUTPUT}"
