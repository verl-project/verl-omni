#!/usr/bin/env bash
# FlowGRPO V1 sync trainer e2e on Ascend (ports GPU diffusion E2E ID 1).
#
# Reuses the GPU launcher body: parquet load -> vllm_omni rollout ->
# multi-reward (jpeg_compressibility rule + vLLM OCR GenRM, weighted sum) ->
# flow_grpo -> FSDP LoRA -> sync, with trainer.use_v1=true and
# v1.trainer_mode=sync. lib_npu_e2e.sh supplies CANN/ATB sourcing,
# visible-device validation and the trainer.device=npu plus Ray visibility
# runtime args. The GPU launcher selects native/TORCH_SDPA attention
# automatically because fa_available() is false on the pinned Ascend stack.
#
# The OCR reward model defaults to the tiny random Qwen2.5-VL checkpoint built
# for Test 11 (the architecture validated on Ascend in issue #791 probes)
# instead of the GPU smoke's tiny qwen3-vl; the structural multi-reward
# contract is unchanged. Override QWEN_IMAGE_REWARD_MODEL_PATH to select another local
# checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}"
export TOKENIZER_PATH="${MODEL_PATH}/tokenizer"
export REWARD_MODEL_PATH="${QWEN_IMAGE_REWARD_MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Qwen2.5-VL-Reward-NPU}"
DATA_DIR="${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-flowgrpo-v1-npu.XXXXXX")}"
export DATA_DIR
unset TRAIN_FILES VAL_FILES

python3 - "${REWARD_MODEL_PATH}" <<'PY'
import sys

from tests.reward_loop.build_qwen25_vl_reward_tiny_random import ensure_tiny_qwen25_vl_reward_checkpoint

ensure_tiny_qwen25_vl_reward_checkpoint(sys.argv[1])
PY

exec bash "${SCRIPT_DIR}/run_flowgrpo_qwen_image.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    trainer.experiment_name=flowgrpo-diffusion-v1-sync-npu-e2e \
    "$@"
