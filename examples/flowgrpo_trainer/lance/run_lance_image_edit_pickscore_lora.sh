#!/usr/bin/env bash
# Lance image_edit FlowGRPO; see README.md for data preparation and dependencies.
set -euo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
exec bash "$(dirname "$0")/run_lance_pickscore_lora.sh" \
    "data.train_files=$WORKSPACE/data/lance_image_edit/lance/train.parquet" \
    "data.val_files=$WORKSPACE/data/lance_image_edit/lance/test.parquet" \
    "actor_rollout_ref.model.tokenizer_path=$WORKSPACE/data/lance_image_edit/lance/tokenizer" \
    actor_rollout_ref.rollout.agent.default_agent_loop=lance_diffusion_single_turn_agent \
    trainer.experiment_name=lance_image_edit_pickscore_lora \
    "$@"
