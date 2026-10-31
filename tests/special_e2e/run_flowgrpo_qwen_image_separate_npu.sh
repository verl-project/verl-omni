#!/usr/bin/env bash
# FlowGRPO synchronous separate trainer e2e on Ascend (ports GPU diffusion
# E2E ID 6, the deprecated v0 distributed path with trainer.use_v1=false).
#
# Reuses the GPU launcher body. The launcher's default CKPT_BACKEND=nccl
# resolves to verl's HCCLCheckpointEngine on Ascend (the CUDA/NPU engines
# are registered under the same name and are mutually exclusive by
# construction), so no backend override is needed. lib_npu_e2e.sh supplies
# CANN/ATB sourcing, visible-device validation and the trainer.device=npu
# plus Ray visibility runtime args. The GPU launcher selects native/TORCH_SDPA
# attention automatically because fa_available() is false on the pinned
# Ascend stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}"
export TOKENIZER_PATH="${MODEL_PATH}/tokenizer"
DATA_DIR="${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-flowgrpo-separate-npu.XXXXXX")}"
export DATA_DIR
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_flowgrpo_qwen_image_separate.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    "$@"
