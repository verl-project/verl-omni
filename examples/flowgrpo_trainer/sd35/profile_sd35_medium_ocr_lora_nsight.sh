#!/usr/bin/env bash
# Profile the legacy SD3.5 Medium OCR LoRA actor with Nsight Systems or Compute.
# Run from the repository root. Requires the OCR dataset and three CUDA GPUs.
set -euo pipefail

PROFILE_TOOL=${PROFILE_TOOL:-nsys}
PROFILE_DIR=${PROFILE_DIR:-./outputs/profile_sd35_nsight}
MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-3.5-medium}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT="$SCRIPT_DIR/run_sd35_medium_ocr_lora.sh"

profile_args=(
    "actor_rollout_ref.model.path=$MODEL_PATH"
    data.train_batch_size=4
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4
    actor_rollout_ref.rollout.pipeline.height=256
    actor_rollout_ref.rollout.pipeline.width=256
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,4]'
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    trainer.experiment_name=sd35_medium_ocr_lora_nsight
    trainer.total_training_steps=5
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.resume_mode=disable
    'trainer.logger=[console]'
    'global_profiler.steps=[3]'
    global_profiler.profile_continuous_steps=False
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.all_ranks=False
    'actor_rollout_ref.actor.profiler.ranks=[0]'
    actor_rollout_ref.actor.profiler.tool=nsys
    actor_rollout_ref.actor.profiler.tool_config.nsys.discrete=False
)

case "$PROFILE_TOOL" in
    nsys)
        command -v nsys >/dev/null 2>&1 || {
            echo "nsys was not found on PATH" >&2
            exit 127
        }
        exec bash "$TRAIN_SCRIPT" \
            "${profile_args[@]}" \
            global_profiler.tool=nsys \
            "$@"
        ;;
    ncu)
        command -v ncu >/dev/null 2>&1 || {
            echo "ncu was not found on PATH" >&2
            exit 127
        }
        if [[ -z "${NCU_KERNEL_REGEX:-}" ]]; then
            echo "Set NCU_KERNEL_REGEX to a compute-kernel name pattern from the Systems report" >&2
            exit 2
        fi
        mkdir -p "$PROFILE_DIR"
        exec ncu \
            --target-processes all \
            --profile-from-start off \
            --nvtx \
            --nvtx-include 'actor_update' \
            --kernel-name "regex:${NCU_KERNEL_REGEX}" \
            --launch-count 1 \
            --set basic \
            --force-overwrite \
            --export "$PROFILE_DIR/sd35_medium_actor" \
            bash "$TRAIN_SCRIPT" \
            "${profile_args[@]}" \
            global_profiler.tool=torch \
            "$@"
        ;;
    *)
        echo "PROFILE_TOOL must be 'nsys' or 'ncu' (got '$PROFILE_TOOL')" >&2
        exit 2
        ;;
esac
