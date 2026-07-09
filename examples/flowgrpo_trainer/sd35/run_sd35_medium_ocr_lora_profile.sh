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
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-4}

# The base recipe's SDE window (size 3, start range [0,5]) assumes 10
# denoising steps; with fewer steps a window past the last step yields
# ragged per-sample tensors. Keep the window inside the schedule. Requires
# NUM_INFERENCE_STEPS >= sde_window_size (3).
SDE_WINDOW_RANGE=${SDE_WINDOW_RANGE:-"[0,$NUM_INFERENCE_STEPS]"}

# The diffusion engine chunks batches statically: per-GPU sample count
# (TRAIN_BATCH_SIZE * ROLLOUT_N / num actor GPUs) must be divisible by the
# micro batch size. The base recipe's micro batch size (8) assumes the full
# footprint; shrink it along with the batch.
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}

# A profiling run always ends on its last step, which force-triggers
# checkpoint saving and validation when save_freq/test_freq > 0 — neither is
# useful here, so both are disabled. resume_mode=disable keeps a leftover
# checkpoint from shifting the step counter past PROFILE_STEP, which would
# silently skip profiling.

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4} \
ROLLOUT_N=${ROLLOUT_N:-2} \
NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS \
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-256} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-$PROFILE_STEP} \
WANDB_MODE=${WANDB_MODE:-disabled} \
    bash "$SCRIPT_DIR/run_sd35_medium_ocr_lora.sh" \
    actor_rollout_ref.rollout.algo.sde_window_range="$SDE_WINDOW_RANGE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
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
