#!/usr/bin/env bash
set -euo pipefail

# Replay the latest route-debug rollout-corr dump through the fixed-sequence
# Megatron scorer, but compute old run1 directly without CPU save/restore.
# This isolates whether save_model_to_cpu/restore_model_from_cpu is polluting
# the multi-node scorer self-consistency signal.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ID=${RUN_ID:-full32_fixed_sequence_score_direct_old_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_score_direct_old_${RUN_ID}}

export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-${LOG_DIR}/rollout_corr_debug/full32_vllm_omni_route_debug_20260714_192838.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-8}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1

# Reuse the Ray seed shape from the successful 20260714_204303 route-debug
# fixed-score run. Do not narrow the worker pool: the normal launcher already
# probes and locks a free full-size block per node, and narrowing the pool made
# worker nodes fail before joining Ray.
export RAY_PORT_SEED=${RAY_PORT_SEED:-k8s-vj-ibw0io-1784032954672}
export RAY_WORKER_PORT_SEED=${RAY_WORKER_PORT_SEED:-${RAY_PORT_SEED}}

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_score_probe.sh" "$@"
