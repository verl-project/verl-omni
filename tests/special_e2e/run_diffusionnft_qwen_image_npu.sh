#!/usr/bin/env bash
# Ascend port of the GPU Qwen-Image DiffusionNFT V1 sync smoke.
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
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2}" \
    trainer.experiment_name=diffusionnft-diffusion-v1-sync-npu-e2e \
    "$@"
