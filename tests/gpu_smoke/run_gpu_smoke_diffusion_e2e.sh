#!/usr/bin/env bash
# ci-e2e-diffusion GPU smoke tests (4-GPU): end-to-end diffusion training paths.
# Includes FlowGRPO / online DPO / DiffusionNFT (v0), synchronous separate,
# FlowGRPO v1 separate_async, and two-teacher OPD on the v1 sync and
# separate_async trainers.

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

run_qwen_image_edit_flowgrpo_e2e() {
    local model_path="${MODEL_PATH:-${HOME}/models/tiny-random/qwen-image-edit-plus}"
    if ! python tests/special_e2e/build_qwen_image_edit_plus_tiny_random.py \
        --output-dir "${model_path}"; then
        return 1
    fi
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" MODEL_PATH="${model_path}" \
        bash tests/special_e2e/run_flowgrpo_qwen_image_edit.sh "${diffusion_trainer_args[@]}"
}

run_test 0 "Qwen-Image-Edit FlowGRPO trainer e2e" \
    run_qwen_image_edit_flowgrpo_e2e

run_test 1 "FlowGRPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 2 "Qwen-Image online DPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_online_dpo_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 3 "DiffusionNFT trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_diffusionnft_qwen_image.sh "${diffusion_trainer_args[@]}"

run_test 4 "FlowGRPO v1 separate_async trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async.sh "${diffusion_trainer_args[@]}"

run_test 5 "Bagel PickScore LoRA FlowGRPO e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_bagel_pickscore.sh "${diffusion_trainer_args[@]}"

run_test 6 "FlowGRPO synchronous separate trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_separate.sh "${diffusion_trainer_args[@]}"

run_test 7 "MiniMax-H3 FlowGRPO T2VA trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" ROLLOUT_TP=2 TOTAL_TRAINING_STEPS=1 \
    python3 tests/special_e2e/run_flowgrpo_minimax_h3_tiny.py --task t2va

run_test 8 "Diffusion OPD v1 sync colocated teachers e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" SMOKE=sync \
    bash tests/special_e2e/run_diffusion_teacher_smoke.sh

run_test 9 "Diffusion OPD v1 separate_async standalone teachers e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" SMOKE=async \
    bash tests/special_e2e/run_diffusion_teacher_smoke.sh

gpu_smoke_summary
