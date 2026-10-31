#!/usr/bin/env bash
# Diffusion OPD two-teacher smoke on Ascend (ports GPU diffusion E2E IDs 8/9).
#
# SMOKE=sync covers the V1 colocated-teachers path and SMOKE=async the
# separate_async standalone-teachers path, mirroring the GPU launcher modes.
# Reuses the GPU launcher body including its offline tiny SD3 student and
# teacher builds (three seeds); lib_npu_e2e.sh supplies CANN/ATB sourcing,
# visible-device validation and the trainer.device=npu plus Ray visibility
# runtime args.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

case "${SMOKE:-sync}" in
    sync) ;;
    async)
        if (( NUM_NPUS < 4 )); then
            echo "Async teacher smoke requires at least four NPUs." >&2
            exit 2
        fi
        ;;
    *)
        echo "SMOKE must be sync or async." >&2
        exit 2
        ;;
esac

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/sd3-teacher-smoke-student}"
export TEACHER_PATH="${TEACHER_PATH:-${HOME}/.cache/verl-omni-npu-smoke/sd3-teacher-smoke-teacher}"
export TEACHER2_PATH="${TEACHER2_PATH:-${HOME}/.cache/verl-omni-npu-smoke/sd3-teacher-smoke-teacher2}"
DATA_DIR="${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-diffusion-teacher-npu.XXXXXX")}"
export DATA_DIR
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_diffusion_teacher_smoke.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    "$@"
