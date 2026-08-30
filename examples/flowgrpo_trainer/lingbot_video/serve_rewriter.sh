#!/usr/bin/env bash
# LingBot prompt rewriter server.
set -euo pipefail

ROOT=${ROOT:-${HOME}}
VLLM=${VLLM:-vllm}
BASE_MODEL=${BASE_MODEL:-$ROOT/models/Qwen3.6-27B}
LORA=${LORA:-$ROOT/models/lingbot-video-rewriter-lora}
SERVED_NAME=${SERVED_NAME:-qwen3_5_27b}

PORT=${PORT:-8137}
TP=${TP:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_UTIL=${GPU_UTIL:-0.92}

COMPILE_CONFIG=${COMPILE_CONFIG:-'{"pass_config": {"fuse_allreduce_rms": false}}'}

EXTRA=()
if [[ -n "${ENFORCE_EAGER:-}" ]]; then
    EXTRA+=(--enforce-eager)
elif [[ -n "$COMPILE_CONFIG" ]]; then
    EXTRA+=(--compilation-config "$COMPILE_CONFIG")
fi

exec "$VLLM" serve "$BASE_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --language-model-only \
    --enable-lora \
    --lora-modules "rewriter=$LORA" \
    --max-lora-rank 32 \
    "${EXTRA[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$PORT"
