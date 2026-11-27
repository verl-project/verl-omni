#!/usr/bin/env bash
# 8-node, 64-GPU defaults for run_qwen_image_ocr_veomni.sh.
# NUM_GPUS and NUM_NODES stay overridable.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NUM_GPUS=${NUM_GPUS:-64} NUM_NODES=${NUM_NODES:-8} \
    exec bash "$SCRIPT_DIR/run_qwen_image_ocr_veomni.sh" "$@"
