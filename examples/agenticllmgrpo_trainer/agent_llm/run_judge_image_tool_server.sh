#!/usr/bin/env bash
# Frozen OpenAI-compatible vision-judge sidecar for the agentic recipes.
set -e
set -x

MODEL="${JUDGE_IMAGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${JUDGE_IMAGE_HOST:-127.0.0.1}"
PORT="${JUDGE_IMAGE_PORT:-8093}"
MAX_NUM_SEQS="${AGENTIC_REFLECT_MAX_NUM_SEQS:-2}"
GPU_MEM_UTIL="${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.8}"
MAX_MODEL_LEN="${AGENTIC_REFLECT_MAX_MODEL_LEN:-4096}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "[INFO] image judge model=${MODEL} endpoint=http://${HOST}:${PORT}"
exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  "$@"
