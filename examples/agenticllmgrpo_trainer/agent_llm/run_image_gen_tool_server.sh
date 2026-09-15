#!/usr/bin/env bash
# Frozen vLLM-Omni image-generation sidecar for the agentic recipes.
#
# Request-level batching is on by default. The sweep in ``benchmarking_batching/``
# measured request-level static-wave packing at width 16 with a 10 ms admission
# window as the throughput best (1.06 images/s at client concurrency 16, ~7x the
# unbatched serial baseline) with CPU weight offload on. Step execution
# (``--step-execution``) lost to request packing at every width tested, so it is
# deliberately not passed. Re-run that harness to re-validate or re-tune.
set -e
set -x

MODEL="${IMAGE_GEN_MODEL:-Qwen/Qwen-Image}"
HOST="${IMAGE_GEN_HOST:-127.0.0.1}"
PORT="${IMAGE_GEN_PORT:-8092}"
NUM_GPUS="${QWEN_IMAGE_NUM_GPUS:-1}"
# Request-level packing width and admission coalescing window. Request-level
# mode OOMs above ~32; 16 is the width the sweep validated.
MAX_NUM_SEQS="${IMAGE_GEN_MAX_NUM_SEQS:-16}"
REQUEST_BATCH_MAX_WAIT_MS="${IMAGE_GEN_REQUEST_BATCH_MAX_WAIT_MS:-10}"

echo "[INFO] image generation model=${MODEL} endpoint=http://${HOST}:${PORT}"
exec vllm-omni serve "$MODEL" \
  --omni \
  --host "$HOST" \
  --port "$PORT" \
  --num-gpus "$NUM_GPUS" \
  --tensor-parallel-size "$NUM_GPUS" \
  --enable-cpu-offload \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --request-batch-max-wait-ms "$REQUEST_BATCH_MAX_WAIT_MS" \
  "$@"
