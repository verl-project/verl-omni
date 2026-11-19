#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$script_dir/run_qwen_image_ocr_lora_v1.sh" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.lora_init_weights=pissa_niter_4 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.actor.checkpoint.save_lora_only=false \
    trainer.experiment_name=qwen_image_ocr_pissa_v1 \
    "$@"
