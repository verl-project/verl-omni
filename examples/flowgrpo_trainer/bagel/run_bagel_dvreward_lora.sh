#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the directory containing DVReward train.parquet and test.parquet}"

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

bash "$recipe_dir/run_bagel_ocr_lora.sh" \
    "data.train_files=$DATA_DIR/train.parquet" \
    "data.val_files=$DATA_DIR/test.parquet" \
    data.max_prompt_length=1024 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    reward.reward_model.model_path=Qwen/Qwen3-VL-30B-A3B-Instruct \
    reward.reward_model.rollout.response_length=10 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/dvreward.py \
    reward.custom_reward_function.name=compute_score_dvreward \
    trainer.experiment_name=bagel_dvreward_lora "$@"
