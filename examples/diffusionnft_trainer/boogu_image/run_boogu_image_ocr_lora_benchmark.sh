#!/usr/bin/env bash
# Boogu-Image T2I DiffusionNFT throughput benchmark.
#
# A thin override of run_boogu_image_ocr_lora.sh, mirroring the Qwen sibling
# examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark.sh:
# periodic validation and checkpointing are disabled so that a fixed number of
# training steps measures steady-state step time instead of val/save cost, and
# the reward engine is allowed CUDA graphs so its batching does not serialize
# the step.
#
# Deliberately NOT enabled here, unlike the Qwen benchmark:
#   - `actor.strategy=fsdp2` / `forward_prefetch`. The Boogu recipe runs FSDP1
#     with param + optimizer offload and does not OOM, so switching strategies
#     would benchmark a configuration the recipe does not ship.
#   - `rollout.step_execution=True` and regional `torch.compile`. The Boogu
#     rollout goes through vllm-omni's diffusion pipeline, which is not the
#     code path those knobs were tuned against.
#
# Usage (from the repo root, as the recipe requires):
#   NUM_GPUS=4 bash examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora_benchmark.sh
#
# The step count is the recipe's own env var:
#   TOTAL_TRAIN_STEPS=8 NUM_GPUS=4 bash .../run_boogu_image_ocr_lora_benchmark.sh
#
# Any extra args are forwarded to the recipe as Hydra overrides.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
NUM_GPUS=${NUM_GPUS:-4}

NUM_GPUS=$NUM_GPUS bash "$SCRIPT_DIR/run_boogu_image_ocr_lora.sh" \
    trainer.test_freq=0 \
    trainer.val_before_train=False \
    trainer.save_freq=0 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.experiment_name=boogu_image_ocr_lora_benchmark \
    reward.reward_model.rollout.enforce_eager=False \
    reward.reward_model.rollout.max_num_seqs=128 \
    "$@"
