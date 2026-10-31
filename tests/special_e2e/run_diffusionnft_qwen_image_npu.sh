#!/usr/bin/env bash
# DiffusionNFT V1 sync trainer e2e on Ascend (ports GPU diffusion E2E ID 3).
#
# Reuses the GPU launcher body (dummy data, algorithm knobs, rule-only reward);
# lib_npu_e2e.sh supplies CANN/ATB sourcing, visible-device validation and the
# trainer.device=npu plus Ray visibility runtime args. The GPU launcher picks
# native/TORCH_SDPA attention automatically because fa_available() is false on
# the pinned Ascend stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}"
export TOKENIZER_PATH="${MODEL_PATH}/tokenizer"
DATA_DIR="${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-diffusionnft-npu.XXXXXX")}"
export DATA_DIR
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_diffusionnft_qwen_image.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    trainer.experiment_name=diffusionnft-diffusion-v1-sync-npu-e2e \
    "$@"
