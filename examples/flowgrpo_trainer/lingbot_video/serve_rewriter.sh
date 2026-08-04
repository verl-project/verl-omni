#!/usr/bin/env bash
# Launch a single tensor-parallel vLLM OpenAI server for the LingBot prompt
# rewriter: the Qwen3.5-27B base VLM sharded across all GPUs with the rewriter
# LoRA attached, so one endpoint serves both stages (step1 EXPAND = base model
# id; step2 MAP = `rewriter` LoRA id).  Drive it with `run_rewrite.sh`.
#
#   bash serve_rewriter.sh                      # TP=8 across GPUs 0-7
#   TP=4 bash serve_rewriter.sh                 # TP=4
#
# Notes:
#   --language-model-only  drops the vision tower (rewrite is text-only for T2V),
#                          cutting memory and avoiding LoRA warnings on visual layers.
#   --enable-lora          the step2 MAP adapter is mandatory for structured JSON;
#                          request it per call via the `rewriter` served-model id.
#   --compilation-config   cudagraph + piecewise compile ARE on (big decode speedup
#                          over eager).  We disable only the `fuse_allreduce_rms`
#                          pass, which on this Qwen3.5 hybrid-Mamba + TP path emits a
#                          non-contiguous all-reduce output and trips a contiguity
#                          assert during KV-cache profiling.  Everything else compiles
#                          and captures graphs normally.  Override COMPILE_CONFIG=''
#                          (empty) + set ENFORCE_EAGER=1 to fall back to eager mode.
#   No --reasoning-parser: the client sets enable_thinking=False and reads the raw
#                          `content`, so a reasoning parser would only risk moving
#                          the JSON into `reasoning_content`.
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
# Keep cudagraph on; disable only the fusion that breaks under TP.
COMPILE_CONFIG=${COMPILE_CONFIG:-'{"pass_config": {"fuse_allreduce_rms": false}}'}

EXTRA=()
if [ -n "${ENFORCE_EAGER:-}" ]; then
    EXTRA+=(--enforce-eager)
elif [ -n "$COMPILE_CONFIG" ]; then
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
