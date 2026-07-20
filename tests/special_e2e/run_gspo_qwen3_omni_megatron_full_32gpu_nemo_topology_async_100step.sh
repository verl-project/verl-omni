#!/usr/bin/env bash
set -xeuo pipefail

# 100-step async-RL control aligned with the Megatron topology exercised by
# NeMo-RL #2356. It keeps our 16/16 trainer-rollout resource split, while
# removing TP/PP/SP from the Megatron actor/ref path. veRL's variable-length
# rollout batches require Megatron Core's alltoall dispatcher.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID=${RUN_ID:-full32_nemo_topology_async_100step_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_nemo_topology_async_100step_${RUN_ID}}
export PROFILE_LABEL=${PROFILE_LABEL:-32-GPU 100-step async-RL NeMo-topology control}

# The default is intentionally modest for a PR control. Set
# TOTAL_TRAINING_STEPS=200 at submission time for the longer variant.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))}

# Mirror NeMo's policy-side Megatron layout as closely as our 16-GPU trainer
# half permits: TP1/PP1/EP8 and no sequence parallel. NeMo's allgather choice
# cannot be used here because Megatron Core rejects it for variable sequences.
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

# This is a real control rather than a frozen-sequence or phase-debug probe.
export VERL_OMNI_SKIP_PIPELINES=${VERL_OMNI_SKIP_PIPELINES:-0}
export VERL_OMNI_SKIP_REWARD_LOOP=${VERL_OMNI_SKIP_REWARD_LOOP:-0}
export VERL_OMNI_SKIP_WEIGHT_UPDATE=${VERL_OMNI_SKIP_WEIGHT_UPDATE:-0}
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS=${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-0}
export VERL_OMNI_PHASE_GENERATION_DEBUG=${VERL_OMNI_PHASE_GENERATION_DEBUG:-0}
export VERL_OMNI_WEIGHT_SYNC_DEBUG=${VERL_OMNI_WEIGHT_SYNC_DEBUG:-1}

exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh" "$@"
