#!/usr/bin/env bash
# Bagel PickScore LoRA FlowGRPO e2e on Ascend (ports GPU diffusion E2E ID 5,
# the deprecated v0 trainer path kept covered until the Bagel v1 recipe
# lands upstream).
#
# Reuses the GPU launcher body; the BAGEL-MoT and PickScore tiny checkpoints
# are built offline by that launcher. lib_npu_e2e.sh supplies CANN/ATB
# sourcing, visible-device validation and the trainer.device=npu plus Ray
# visibility runtime args.
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
