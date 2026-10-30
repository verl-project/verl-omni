#!/usr/bin/env bash
# Lance video_edit FlowGRPO; see README.md for data preparation and dependencies.
set -euo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
exec bash "$(dirname "$0")/run_lance_t2v_imagebind_lora.sh" \
    "data.train_files=$WORKSPACE/data/lance_video_edit/lance/train.parquet" \
    "data.val_files=$WORKSPACE/data/lance_video_edit/lance/test.parquet" \
    "actor_rollout_ref.model.tokenizer_path=$WORKSPACE/data/lance_video_edit/lance/tokenizer" \
    actor_rollout_ref.rollout.agent.default_agent_loop=lance_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.pipeline.frame_rate=12 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=12 \
    trainer.experiment_name=lance_video_edit_imagebind_lora \
    "$@"
