#!/usr/bin/env bash
# MiniMax-H3 T2VA FlowGRPO with FSDP2, actor SP=2, and rollout TP=2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

MINIMAX_H3_MODEL_PATH=${MINIMAX_H3_MODEL_PATH:-${HOME}/models/tiny-random/minimax-h3-npu-offline}
MINIMAX_H3_DATA_DIR="${MINIMAX_H3_DATA_DIR:-${DATA_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/verl-omni-minimax-h3-npu.XXXXXX")}}"
python3 "${SCRIPT_DIR}/build_minimax_h3_tiny_random.py" --offline --output-dir "${MINIMAX_H3_MODEL_PATH}"

python3 "${SCRIPT_DIR}/run_flowgrpo_minimax_h3_tiny.py" \
    --device npu \
    --task t2va \
    --num-gpus "${NUM_NPUS}" \
    --actor-sp 2 \
    --rollout-tp 2 \
    --total-steps 1 \
    --tiny-model-dir "${MINIMAX_H3_MODEL_PATH}" \
    --data-dir "${MINIMAX_H3_DATA_DIR}" \
    --output-dir "${MINIMAX_H3_OUTPUT_DIR:-${SCRIPT_DIR}/../../outputs/flowgrpo_minimax_h3_npu}" \
    "$@"

echo 'MiniMax-H3 FlowGRPO T2VA NPU e2e test passed (training completed successfully).'
