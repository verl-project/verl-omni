#!/usr/bin/env bash
# Qwen-Image-Edit FlowGRPO V1 sync trainer e2e on Ascend (ports GPU diffusion
# E2E ID 0).
#
# Reuses the GPU launcher body (edit dummy data, rule-only reward, LoRA sync);
# lib_npu_e2e.sh supplies CANN/ATB sourcing, visible-device validation and the
# trainer.device=npu plus Ray visibility runtime args. The GPU launcher selects
# native/TORCH_SDPA attention automatically because fa_available() is false on
# the pinned Ascend stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
DATA_DIR="${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-flowgrpo-edit-npu.XXXXXX")}"
export DATA_DIR
unset TRAIN_FILES VAL_FILES
# The GPU launcher defaults to Hub FA3 at SP=1, which cannot download kernels
# in the offline NPU environment; select the portable native/SDPA pair (the
# launcher's own SP>1 branch default).
export ATTN_BACKEND="${ATTN_BACKEND:-native}"
export ROLLOUT_ATTN_BACKEND="${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}"

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "${MODEL_PATH}" || ! -f "${MODEL_PATH}/model_index.json" ]]; then
        echo "MODEL_PATH must be a local checkpoint directory containing model_index.json: ${MODEL_PATH}" >&2
        exit 2
    fi
else
    MODEL_PATH="${HOME}/.cache/verl-omni-npu-smoke/qwen-image-edit-plus"
    python3 - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
owned = not path.exists() or (path.is_dir() and not any(path.iterdir()))
if path.is_dir() and not owned:
    try:
        metadata = json.loads((path / "tiny_checkpoint_metadata.json").read_text())
    except (OSError, ValueError):
        metadata = None
    owned = (
        isinstance(metadata, dict)
        and set(metadata) == {"format_version", "chat_template"}
        and metadata["format_version"] == 1
        and isinstance(metadata["chat_template"], str)
    )
if path.is_symlink() or not owned:
    raise SystemExit(f"Refusing to build a tiny checkpoint in an unowned path: {path}")
PY
    python3 "${SCRIPT_DIR}/build_qwen_image_edit_plus_tiny_random.py" --output-dir "${MODEL_PATH}"
fi
export MODEL_PATH

exec bash "${SCRIPT_DIR}/run_flowgrpo_qwen_image_edit.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    trainer.experiment_name=flowgrpo-qwen-image-edit-v1-sync-npu-e2e \
    "$@"
