#!/usr/bin/env bash
# Frozen vLLM-Omni image-generation sidecar for the agentic recipes.
set -e
set -x

MODEL="${IMAGE_GEN_MODEL:-Qwen/Qwen-Image}"
HOST="${IMAGE_GEN_HOST:-127.0.0.1}"
PORT="${IMAGE_GEN_PORT:-8092}"
NUM_GPUS="${QWEN_IMAGE_NUM_GPUS:-1}"

echo "[INFO] image generation model=${MODEL} endpoint=http://${HOST}:${PORT}"
exec vllm-omni serve "$MODEL" \
  --omni \
  --host "$HOST" \
  --port "$PORT" \
  --num-gpus "$NUM_GPUS" \
  --tensor-parallel-size "$NUM_GPUS" \
  --enable-cpu-offload \
  "$@"
