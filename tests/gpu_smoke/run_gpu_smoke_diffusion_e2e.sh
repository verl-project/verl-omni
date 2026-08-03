#!/usr/bin/env bash
# ci-e2e-diffusion GPU smoke tests (4-GPU): end-to-end diffusion training paths.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "ci-e2e-diffusion" 4 "$@"

diffusion_trainer_args=()
if [[ -n "${RAY_MASTER_PORT_RANGE:-}" ]]; then
    diffusion_trainer_args+=("trainer.ray_master_port_range=[${RAY_MASTER_PORT_RANGE}]")
fi

run_test 0 "FlowGRPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 1 "Qwen-Image online DPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_online_dpo_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 2 "DiffusionNFT trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_diffusionnft_qwen_image.sh "${diffusion_trainer_args[@]}"

gpu_smoke_summary
