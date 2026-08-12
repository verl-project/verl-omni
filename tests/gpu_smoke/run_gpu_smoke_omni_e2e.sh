#!/usr/bin/env bash
# ci-e2e-omni GPU smoke tests (2-GPU): end-to-end omni training paths.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "ci-e2e-omni" 2 "$@"

# Fixed at 2 GPUs: FSDP/FSPD2 needs >1 GPU to shard (NO_SHARD can't run the
# offload_to_cpu LoRA-sync summon).
run_test 0 "Qwen3-Omni Thinker GSPO LoRA e2e (V1)" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS=2 \
    bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_smoke.sh

# Separate-async: 1 trainer GPU + 1 standalone TP=1 rollout replica.
run_test 1 "Qwen3-Omni Thinker GSPO LoRA separate-async e2e (V1)" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 \
    bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_separate_async_smoke.sh

gpu_smoke_summary
