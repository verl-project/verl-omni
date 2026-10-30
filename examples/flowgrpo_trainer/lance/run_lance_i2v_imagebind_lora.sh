#!/usr/bin/env bash
# Lance i2v FlowGRPO; see README.md for data preparation and dependencies.
set -euo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
exec bash "$(dirname "$0")/run_lance_t2v_imagebind_lora.sh" \
    "data.train_files=$WORKSPACE/data/lance_i2v/lance/train.parquet" \
    "data.val_files=$WORKSPACE/data/lance_i2v/lance/test.parquet" \
    "actor_rollout_ref.model.tokenizer_path=$WORKSPACE/data/lance_i2v/lance/tokenizer" \
    actor_rollout_ref.rollout.agent.default_agent_loop=lance_diffusion_single_turn_agent \
    trainer.experiment_name=lance_i2v_imagebind_lora \
    "$@"
