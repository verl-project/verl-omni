#!/usr/bin/env bash
# Multi-GPU DMD2 production smoke, including atomic checkpoints and student export.
set -euo pipefail

export NUM_GPUS=${NUM_GPUS:-2}
export TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3}
export MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/dmd2_smoke}
DATA_DIR=${DATA_DIR:-${OUTPUT_DIR}/data}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "$((NUM_GPUS * TOTAL_TRAIN_STEPS * 3))" \
    --val_size "${NUM_GPUS}" \
    --user_prompt_only

export TRAIN_FILES=${DATA_DIR}/train.parquet
export VAL_FILES=${DATA_DIR}/test.parquet
export SAVE_FREQ=1
export RESUME_MODE=${RESUME_MODE:-disable}

bash examples/dmd2_trainer/qwen_image/run_qwen_image_dmd2_lora.sh \
    actor_rollout_ref.model.lora_rank=2 \
    actor_rollout_ref.model.lora_alpha=2 \
    actor_rollout_ref.model.pipeline.height=64 \
    actor_rollout_ref.model.pipeline.width=64 \
    actor_rollout_ref.model.pipeline.max_sequence_length=64 \
    data.max_prompt_length=64 \
    trainer.logger=console \
    "$@"

test -f "${OUTPUT_DIR}/inference/adapter_model.safetensors"
test -f "${OUTPUT_DIR}/global_step_${TOTAL_TRAIN_STEPS}/trainer.pt"
echo 'DMD2 training, checkpoint and student export completed.'
