#!/usr/bin/env bash
set -xeuo pipefail

# Pure Megatron actor-HF parity control with the NeMo-RL-inspired layout.
# It differs from the production-topology control only in Megatron parallelism:
# TP1/PP1/EP8 with sequence parallel disabled. alltoall remains required for
# veRL's variable-length fixed rollout records.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID=${RUN_ID:-full32_megatron_bridge_hf_parity_nemo_topology_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_bridge_hf_parity_nemo_topology_${RUN_ID}}
export PROFILE_LABEL=${PROFILE_LABEL:-32-GPU offline Megatron-Bridge NeMo-topology actor-HF parity control}

export ACTOR_TP=${ACTOR_TP:-1}
export ACTOR_PP=${ACTOR_PP:-1}
export ACTOR_CP=${ACTOR_CP:-1}
export ACTOR_EP=${ACTOR_EP:-8}
export ACTOR_ETP=${ACTOR_ETP:-1}
export REF_TP=${REF_TP:-${ACTOR_TP}}
export REF_PP=${REF_PP:-${ACTOR_PP}}
export REF_CP=${REF_CP:-${ACTOR_CP}}
export REF_EP=${REF_EP:-${ACTOR_EP}}
export REF_ETP=${REF_ETP:-${ACTOR_ETP}}
export SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-False}
export MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}

# TP1/PP1 yields attention-DP16 on the 16-GPU trainer half. The fixed batch
# must be divisible by that dispatcher width; the frozen source dump has 16
# records, so score all of them by default.
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-16}

exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_bridge_hf_parity_gate.sh" "$@"
