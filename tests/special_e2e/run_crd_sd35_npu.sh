#!/usr/bin/env bash
# Reuse GPU diffusion entry 12, including the distributed EMA/checkpoint oracle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
unset MODEL_PATH TOKENIZER_PATH DATA_DIR TRAIN_FILES VAL_FILES
if [[ -n "${CRD_MODEL_PATH:-}" ]]; then
    if [[ ! -f "${CRD_MODEL_PATH}/model_index.json" ]]; then
        echo "CRD_MODEL_PATH must point at an existing SD3 pipeline." >&2
        exit 2
    fi
    export MODEL_PATH="${CRD_MODEL_PATH}"
fi

exec bash "${SCRIPT_DIR}/run_crd_sd35.sh" "${NPU_RUNTIME_ARGS[@]}" "$@"
