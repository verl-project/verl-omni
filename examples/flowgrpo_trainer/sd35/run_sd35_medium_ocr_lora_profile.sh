#!/usr/bin/env bash
# Lightweight torch-profiler run of the SD3.5-Medium OCR LoRA recipe: fewer
# rollouts, denoising steps, a smaller batch and a smaller image keep the
# traces fast to capture and small to open. Profiles a single step, covering
# the actor train phase (e2e trace) and the vLLM-Omni rollout servers
# (one trace per profiled replica). See docs/perf/profiler.md.
set -x
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PROFILE_STEP=${PROFILE_STEP:-1}
SAVE_PATH=${SAVE_PATH:-./outputs/profile_sd35}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4} \
ROLLOUT_N=${ROLLOUT_N:-2} \
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-4} \
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-256} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-$PROFILE_STEP} \
WANDB_MODE=${WANDB_MODE:-disabled} \
    bash "$SCRIPT_DIR/run_sd35_medium_ocr_lora.sh" \
    global_profiler.tool=torch \
    global_profiler.steps=[$PROFILE_STEP] \
    global_profiler.save_path=$SAVE_PATH \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.ranks=[0] \
    actor_rollout_ref.actor.profiler.tool=torch \
    actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda] \
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.ranks=[0] \
    actor_rollout_ref.rollout.profiler.tool=torch \
    actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cpu,cuda] \
    actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True \
    "$@"
