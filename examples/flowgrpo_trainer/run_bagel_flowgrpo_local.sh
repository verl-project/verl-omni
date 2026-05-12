#!/usr/bin/env bash
# Local launch wrapper for BAGEL FlowGRPO on this workstation.
#
# Resources (8x H200):
#   - BAGEL-7B-MoT     : /proj-tango-pvc/users/zhipeng.wang/workspace/models/BAGEL-7B-MoT
#   - Qwen3-VL reward  : /proj-tango-pvc/users/zhipeng.wang/workspace/models/Qwen3-VL-8B-Instruct
#   - OCR parquets     : /proj-tango-pvc/users/zhipeng.wang/workspace/data/ocr/{train,test}.parquet
#
# Usage:
#   bash examples/flowgrpo_trainer/run_bagel_flowgrpo_local.sh
#   bash examples/flowgrpo_trainer/run_bagel_flowgrpo_local.sh \
#        actor_rollout_ref.rollout.n=4   # CLI overrides forwarded

set -euo pipefail

# ---------------- workspace / models / data ----------------
WORKSPACE=/proj-tango-pvc/users/zhipeng.wang/workspace
export BAGEL_MODEL_PATH=${BAGEL_MODEL_PATH:-$WORKSPACE/models/BAGEL-7B-MoT}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-8B-Instruct}
export OCR_TRAIN_PATH=${OCR_TRAIN_PATH:-$WORKSPACE/data/ocr/train.parquet}
export OCR_TEST_PATH=${OCR_TEST_PATH:-$WORKSPACE/data/ocr/test.parquet}

# Default stage config lives next to the upstream script and will be picked up
# automatically; override only if you need a custom one.
# export BAGEL_STAGE_CONFIG=$WORKSPACE/verl-omni/examples/flowgrpo_trainer/bagel_stage_config.yaml

# ---------------- runtime / logging ----------------
# Use all 8 H200s; comment this out to let Ray see only a subset.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Quieter NCCL / vllm by default; flip to INFO/DEBUG to diagnose hangs.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Bypass the flashinfer / flashinfer-jit-cache version mismatch in this image
# (`flashinfer-jit-cache==0.6.6+cu130` vs `flashinfer==0.6.8.post1`). vLLM 0.20
# eagerly imports `flashinfer.comm` from VllmConfig._set_compile_ranges(), and
# the AOT-cache version check raises RuntimeError otherwise.
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}

# wandb: set WANDB_API_KEY beforehand, or disable here.
export WANDB_MODE=${WANDB_MODE:-disabled}
# export WANDB_PROJECT=flow_grpo
# export WANDB_API_KEY=...

# Avoid HF hub hitting the network on every launch.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Ray temp dir MUST be a short path: it hosts Unix domain sockets and the
# kernel caps AF_UNIX paths at 107 bytes. The PVC mount point already eats
# ~50 chars, so keep RAY_TMPDIR under /tmp.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-${USER:-$(id -un 2>/dev/null || echo root)}}
mkdir -p "$RAY_TMPDIR"

# ---------------- sanity checks ----------------
for p in "$BAGEL_MODEL_PATH" "$REWARD_MODEL_PATH" "$OCR_TRAIN_PATH" "$OCR_TEST_PATH"; do
    [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 1; }
done

# ---------------- launch ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_bagel_flowgrpo.sh" "$@"
