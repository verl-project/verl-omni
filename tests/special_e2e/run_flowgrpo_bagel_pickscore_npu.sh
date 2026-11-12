#!/usr/bin/env bash
# Ascend port of the GPU Bagel PickScore LoRA FlowGRPO smoke.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/BAGEL-MoT}"
export PICKSCORE_PATH="${PICKSCORE_PATH:-${HOME}/.cache/verl-omni-npu-smoke/PickScore}"
DATA_DIR="${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-bagel-pickscore-npu.XXXXXX")}"
export DATA_DIR
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_flowgrpo_bagel_pickscore.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    "$@"
