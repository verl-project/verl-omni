#!/usr/bin/env bash
# ci-e2e-omni GPU smoke tests (2-GPU): end-to-end omni training paths.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "ci-e2e-omni" 2 "$@"

# Fixed at 2 GPUs: the smoke stage config pins tensor_parallel_size=2 and FSDP
# needs >1 GPU to shard (NO_SHARD can't run the offload_to_cpu LoRA-sync summon).
run_test 0 "Qwen3-Omni Thinker GSPO LoRA e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS=2 \
    bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_smoke.sh

gpu_smoke_summary
