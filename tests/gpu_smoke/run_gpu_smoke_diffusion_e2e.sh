#!/usr/bin/env bash
# ci-e2e-diffusion GPU smoke tests (4-GPU): end-to-end diffusion training paths.
# Includes FlowGRPO / online DPO / DiffusionNFT (v0) and FlowGRPO v1 separate_async.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "ci-e2e-diffusion" 4 "$@"

diffusion_trainer_args=()
if [[ -n "${RAY_MASTER_PORT_RANGE:-}" ]]; then
    diffusion_trainer_args+=("trainer.ray_master_port_range=[${RAY_MASTER_PORT_RANGE}]")
fi
# The GPU smoke image may contain the ``kernels`` Python package while its
# installed Torch/CUDA combination has no compatible Hub FA3 build variant.
# Select the portable native/SDPA pair explicitly for these E2E tests so the
# production engine can keep its fail-fast attention-backend behavior.
diffusion_trainer_args+=(
    "actor_rollout_ref.model.attn_backend=native"
    "actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA"
)

run_test 0 "FlowGRPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 1 "Qwen-Image online DPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_online_dpo_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 2 "DiffusionNFT trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_diffusionnft_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 3 "FlowGRPO v1 separate_async trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async.sh "${diffusion_trainer_args[@]}"

run_test 4 "Diffusion OPD teacher distill e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" SMOKE=distill \
    bash tests/special_e2e/run_diffusion_teacher_smoke.sh "${diffusion_trainer_args[@]}"

run_test 5 "Diffusion OPD actor+ref+teacher e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" SMOKE=coexistence \
    bash tests/special_e2e/run_diffusion_teacher_smoke.sh "${diffusion_trainer_args[@]}"

run_test 6 "Bagel PickScore LoRA FlowGRPO e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_bagel_pickscore.sh "${diffusion_trainer_args[@]}"

run_test 7 "Diffusion OPD two colocated teachers e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" SMOKE=mopd \
    bash tests/special_e2e/run_diffusion_teacher_smoke.sh

run_test 8 "Diffusion OPD standalone teacher pool e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" SMOKE=standalone \
    bash tests/special_e2e/run_diffusion_teacher_smoke.sh

gpu_smoke_summary
