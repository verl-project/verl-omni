#!/usr/bin/env bash
# Text-to-video variant of the Lance FlowGRPO recipe. Run from the repository root.
# Requires vllm-omni#7907 for video trajectory recording.
set -euo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-"$WORKSPACE/models/bytedance-research/Lance/Lance_3B_Video"}
IMGBIND_MODEL_PATH=${IMGBIND_MODEL_PATH:-.checkpoints/imagebind_huge.pth}
VIDEO_HEIGHT=${VIDEO_HEIGHT:-480}
VIDEO_WIDTH=${VIDEO_WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-25}
INFER_STEPS=${INFER_STEPS:-30}
repo_root=$(cd "$(dirname "$0")/../../.." && pwd)

export MODEL_PATH
exec bash "$(dirname "$0")/run_lance_pickscore_lora.sh" \
    actor_rollout_ref.rollout.pipeline.height="$VIDEO_HEIGHT" \
    actor_rollout_ref.rollout.pipeline.width="$VIDEO_WIDTH" \
    actor_rollout_ref.rollout.pipeline.num_frames="$NUM_FRAMES" \
    actor_rollout_ref.rollout.pipeline.num_inference_steps="$INFER_STEPS" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height="$VIDEO_HEIGHT" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width="$VIDEO_WIDTH" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames="$NUM_FRAMES" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=30 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.imagebind.path=$repo_root/verl_omni/utils/reward_score/imagebind.py" \
    '+reward.reward_functions.imagebind.name=compute_score' \
    '+reward.reward_functions.imagebind.weight=1.0' \
    '+reward.reward_functions.imagebind.required=true' \
    '+reward.reward_functions.imagebind.device=cuda:0' \
    "+reward.reward_functions.imagebind.model_name_or_path=$IMGBIND_MODEL_PATH" \
    '+reward.reward_functions.imagebind.mode=text_video' \
    reward.aggregation=weighted_sum \
    trainer.experiment_name=lance_t2v_imagebind_lora \
    "$@"
