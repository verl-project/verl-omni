#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}
if [[ "${NUM_GPUS}" -lt 2 ]]; then
    echo "MiniMax H3 LoRA TP=2 regression requires NUM_GPUS>=2, got ${NUM_GPUS}." >&2
    exit 2
fi

python3 - <<'PY'
import importlib

import diffusers

importlib.import_module("vllm_omni.diffusion.models.minimax_h3")
if not hasattr(diffusers, "MiniMaxH3Transformer3DModel"):
    raise RuntimeError("diffusers must export MiniMaxH3Transformer3DModel for this regression.")
PY

python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    tests/special_e2e/minimax_h3_lora_sync_tp2.py
