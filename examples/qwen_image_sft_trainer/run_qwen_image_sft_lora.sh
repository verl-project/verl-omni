#!/usr/bin/env bash
# Qwen-Image SFT with CoRT-style t2i/edit atomic data.
set -euo pipefail
set -x

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen-Image-Edit}
TRAIN_FILES=${TRAIN_FILES:-$WORKSPACE/data/cort/qwen_image_sft/train.jsonl}
VAL_FILES=${VAL_FILES:-$WORKSPACE/data/cort/qwen_image_sft/val.jsonl}
CORT_INTERMEDIATE_DIR=${CORT_INTERMEDIATE_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE/checkpoints/qwen_image_sft_lora}
TRAIN_ENTRY_TYPES=${TRAIN_ENTRY_TYPES:-edit}

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
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
  examples/qwen_image_sft_trainer/qwen_image_sft.py \
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
