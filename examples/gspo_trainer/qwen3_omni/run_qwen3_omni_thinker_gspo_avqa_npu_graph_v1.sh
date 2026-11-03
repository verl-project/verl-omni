#!/usr/bin/env bash
# Graph profiles for the existing full-parameter AVQA recipe.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

export DEVICE=npu
export PYTHONUNBUFFERED=1
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-8}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
NPU_GRAPH_MODE=${NPU_GRAPH_MODE:-PIECEWISE}

if [[ ! "${NUM_GPUS_ACTOR_ROLLOUT_REWARD}" =~ ^[1-9][0-9]*$ || ! "${ROLLOUT_TP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS_ACTOR_ROLLOUT_REWARD and ROLLOUT_TP must be positive integers." >&2
    exit 2
fi
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % ROLLOUT_TP != 0 )); then
    echo "NUM_GPUS_ACTOR_ROLLOUT_REWARD must be divisible by ROLLOUT_TP." >&2
    exit 2
fi
case "${NPU_GRAPH_MODE}" in
    PIECEWISE|FULL_AND_PIECEWISE) ;;
    *) echo "NPU_GRAPH_MODE must be PIECEWISE or FULL_AND_PIECEWISE." >&2; exit 2 ;;
esac

# FULL decode requires a compatible vllm-omni attention-backend binding patch.
# Keep the base recipe's workload, reward, checkpoint and validation settings.
cd -- "${REPO_ROOT}"
exec bash -o pipefail "${SCRIPT_DIR}/run_qwen3_omni_thinker_gspo_avqa_v1.sh" \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_mode=${NPU_GRAPH_MODE}" \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_capture_sizes=[1,16,128,1536]' \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.max_cudagraph_capture_size=1536 \
    "$@"
