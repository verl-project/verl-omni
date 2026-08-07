#!/usr/bin/env bash
# Qwen-Image SFT LoRA on Ascend NPU.
set -euo pipefail
set -x

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
if [[ -f "$ASCEND_HOME_PATH/set_env.sh" ]]; then
  source "$ASCEND_HOME_PATH/set_env.sh"
fi
if [[ -f "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh" ]]; then
  source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"
fi

export DEVICE_NAME=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen-Image-Edit}
TRAIN_FILES=${TRAIN_FILES:-$WORKSPACE/data/cort/qwen_image_sft/train.jsonl}
VAL_FILES=${VAL_FILES:-$WORKSPACE/data/cort/qwen_image_sft/val.jsonl}
CORT_INTERMEDIATE_DIR=${CORT_INTERMEDIATE_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE/checkpoints/qwen_image_sft_lora_npu}
TRAIN_ENTRY_TYPES=${TRAIN_ENTRY_TYPES:-edit}

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29600}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}

INTERMEDIATE_ARGS=()
if [[ -n "$CORT_INTERMEDIATE_DIR" ]]; then
  INTERMEDIATE_ARGS+=(--cort_intermediate_dirs "$CORT_INTERMEDIATE_DIR")
fi

VAL_ARGS=()
if [[ -n "$VAL_FILES" ]]; then
  VAL_ARGS+=(--val_files "$VAL_FILES")
fi

torchrun \
  --nnodes "$NNODES" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  examples/qwen_image_sft_trainer/qwen_image_sft.py \
  --device npu \
  --model_name_or_path "$MODEL_NAME" \
  --pipeline_class auto \
  --train_files "$TRAIN_FILES" \
  "${VAL_ARGS[@]}" \
  "${INTERMEDIATE_ARGS[@]}" \
  --train_entry_types "$TRAIN_ENTRY_TYPES" \
  --cort_t2i_target final \
  --edit_prompt_mode prompt_fix \
  --height "$IMAGE_RESOLUTION" \
  --width "$IMAGE_RESOLUTION" \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --warmup_steps 100 \
  --total_training_steps 1000 \
  --save_freq 500 \
  --test_freq 100 \
  --dtype bfloat16 \
  --gradient_checkpointing \
  --fsdp \
  --lora_rank 64 \
  --lora_alpha 128 \
  --output_dir "$OUTPUT_DIR" \
  "$@"
