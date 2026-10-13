#!/usr/bin/env bash
# SD3.5 Medium FlowGRPO: V1 async queue, colocated actor and rollout GPUs.
# Reuse the sync recipe's model, loss, data and reward settings.
set -euo pipefail

bash "$(dirname "$0")/run_sd35_medium_ocr_lora_v1.sh" \
    trainer.experiment_name=sd35_medium_ocr_lora_v1_colocate_async \
    trainer.v1.trainer_mode=colocate_async \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    "$@"
