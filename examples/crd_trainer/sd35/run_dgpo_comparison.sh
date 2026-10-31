#!/usr/bin/env bash
# Match the CRD experiment settings while retaining DGPO's loss and sampling EMA.
set -euo pipefail
export OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/dgpo/crd_comparison}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/run_sd35_medium_ocr_lora_v1.sh" \
    actor_rollout_ref.model.algorithm=dgpo \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.02 \
    algorithm.old_policy_decay_schedule=linear_to_0_3 \
    trainer.project_name=dgpo \
    trainer.experiment_name=sd35_medium_crd_comparison \
    "$@"
