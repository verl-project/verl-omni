#!/usr/bin/env bash
set -xeuo pipefail

# NeMo-RL #2356 compatibility control for the frozen-sequence scorer.
# Keep the 16 trainer GPUs in the async split, but mirror the relevant NeMo
# Megatron path: TP1, PP1, EP8, and no sequence parallel. veRL feeds
# variable-length batches, so Megatron Core requires alltoall rather than
# NeMo's allgather dispatcher here.
# This gives DP2 on our split instead of NeMo's DP4 on a dedicated 32-GPU
# policy worker group, while removing the TP/PP/SP/alltoall path under test.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID=${RUN_ID:-full32_nemo_topology_component_audit_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_nemo_topology_component_audit_${RUN_ID}}

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

exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_component_audit.sh" "$@"
