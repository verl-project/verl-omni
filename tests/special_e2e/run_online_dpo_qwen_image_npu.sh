#!/usr/bin/env bash
# Ascend port of GPU Qwen-Image online DPO with a tiny Qwen2.5-VL OCR reward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}"
export TOKENIZER_PATH="${MODEL_PATH}/tokenizer"
export REWARD_MODEL_PATH="${QWEN_IMAGE_REWARD_MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Qwen2.5-VL-Reward-NPU}"
DATA_DIR="${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-online-dpo-npu.XXXXXX")}"
export DATA_DIR
unset TRAIN_FILES VAL_FILES

if [[ -n "${QWEN_IMAGE_REWARD_MODEL_PATH:-}" ]]; then
    if [[ ! -d "${REWARD_MODEL_PATH}" ]]; then
        echo "QWEN_IMAGE_REWARD_MODEL_PATH must be an existing local checkpoint directory: ${REWARD_MODEL_PATH}" >&2
        exit 2
    fi
else
    python3 - "${REWARD_MODEL_PATH}" <<'PY'
import sys

from tests.reward_loop.build_qwen25_vl_reward_tiny_random import ensure_tiny_qwen25_vl_reward_checkpoint

ensure_tiny_qwen25_vl_reward_checkpoint(sys.argv[1])
PY
fi

exec bash "${SCRIPT_DIR}/run_online_dpo_qwen_image.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    trainer.experiment_name=online-dpo-qwen-image-v1-sync-npu-e2e \
    "$@"
