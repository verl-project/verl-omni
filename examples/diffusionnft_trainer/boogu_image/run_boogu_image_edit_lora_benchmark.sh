#!/usr/bin/env bash
# Boogu-Image edit (TI2I) DiffusionNFT throughput benchmark.
#
# The edit sibling of run_boogu_image_ocr_lora_benchmark.sh, overriding
# run_boogu_image_edit_lora.sh with exactly the same benchmark deltas so the
# two modes' step times are directly comparable: periodic validation and
# checkpointing are disabled.
#
# The edit recipe carries one extra cost the T2I one does not -- the reference
# image is encoded through the VAE and the reference-image refiner, and the
# prompt budget is 512 rather than 256 tokens because the `<image>` placeholder
# expands into a long vision-token span. Comparing this benchmark against the
# T2I one is how much of the extra step time is the edit path itself.
#
# Usage (from the repo root, as the recipe requires):
#   NUM_GPUS=4 bash examples/diffusionnft_trainer/boogu_image/run_boogu_image_edit_lora_benchmark.sh
#
# The step count is the recipe's own env var:
#   TOTAL_TRAIN_STEPS=8 NUM_GPUS=4 bash .../run_boogu_image_edit_lora_benchmark.sh
#
# Any extra args are forwarded to the recipe as Hydra overrides.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
NUM_GPUS=${NUM_GPUS:-4}

NUM_GPUS=$NUM_GPUS bash "$SCRIPT_DIR/run_boogu_image_edit_lora.sh" \
    trainer.test_freq=0 \
    trainer.val_before_train=False \
    trainer.save_freq=0 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.experiment_name=boogu_image_edit_lora_benchmark \
    "$@"
