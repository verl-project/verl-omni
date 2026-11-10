#!/usr/bin/env bash
# Ascend port of GPU omni entry 3: two full-parameter GRPO updates.
# Requires pinned qwen-tts, audio helpers, SoX and the cached tiny source.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export WORK_DIR="${WORK_DIR:-${HOME}/data/qwen3_tts_grpo_npu/$(date +%Y%m%d_%H%M%S)_$$}"
unset MODEL_PATH TRAIN_FILES VAL_FILES

mini_batch_size=$(( (4 + NUM_NPUS - 1) / NUM_NPUS * NUM_NPUS ))

exec bash "${SCRIPT_DIR}/run_qwen3_tts_grpo_smoke.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    "data.train_batch_size=${mini_batch_size}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}" \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.attention_config.backend=null' \
    "$@"
