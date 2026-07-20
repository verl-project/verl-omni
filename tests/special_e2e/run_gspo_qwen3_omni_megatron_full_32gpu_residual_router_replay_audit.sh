#!/usr/bin/env bash
set -euo pipefail

# One 32-GPU fixed-sequence run that captures the residual/norm boundary,
# raw MoE top-k margins, and R2 record-vs-replay self-consistency.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID=${RUN_ID:-full32_residual_router_replay_audit_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_residual_router_replay_audit_${RUN_ID}}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1
export VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=1
export VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT=1
# Count 0 is R2 RECORD (old run1); count 1 is REPLAY_FORWARD (old run2).
export VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT_LIMIT=2
export VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT=1
export VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS=${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS:-1,4,6,16,32,48}
export VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT=1
export VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT_LAYERS=${VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT_LAYERS:-1}
export VERL_OMNI_MEGATRON_MOE_REPLAY_METADATA_AUDIT=1
export VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE=1
export VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS=${VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS:-1}
export VERL_OMNI_MEGATRON_ROUTER_REPLAY_AUDIT=1

exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_component_audit.sh" \
  actor_rollout_ref.actor.router_replay.mode=R2 \
  actor_rollout_ref.actor.megatron.router_replay.mode=R2 \
  "$@"
