#!/usr/bin/env bash
# CRD with the existing SD3.5 OCR data/reward stack. Run from the repository root.
set -euo pipefail

export NUM_GPUS_ACTOR_ROLLOUT=${NUM_GPUS_ACTOR_ROLLOUT:-1}
export NUM_GPUS_REWARD=${NUM_GPUS_REWARD:-1}
GROUP_SIZE=${GROUP_SIZE:-6}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/crd/sd35_medium_ocr_lora_v1}
for count in "$NUM_GPUS_ACTOR_ROLLOUT" "$NUM_GPUS_REWARD" "$GROUP_SIZE"; do
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU counts and GROUP_SIZE must be positive integers." >&2
        exit 1
    fi
done
if (( GROUP_SIZE < 2 )); then
    echo "CRD requires GROUP_SIZE >= 2." >&2
    exit 1
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$script_dir/../../dgpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh" \
    actor_rollout_ref.model.algorithm=crd \
    actor_rollout_ref.model.policy_state_adapters='["default","old","moving_reference"]' \
    actor_rollout_ref.actor.diffusion_loss.crd_beta=1.0 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.1 \
    actor_rollout_ref.actor.diffusion_loss.crd_kl_min=0.0001 \
    actor_rollout_ref.actor.optim.lr=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$NUM_GPUS_ACTOR_ROLLOUT" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$GROUP_SIZE" \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    actor_rollout_ref.rollout.pipeline.height="$IMAGE_RESOLUTION" \
    actor_rollout_ref.rollout.pipeline.width="$IMAGE_RESOLUTION" \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=4.5 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    algorithm.old_policy_decay_schedule=linear_to_0_5 \
    algorithm.reference_policy_decay_schedule=linear_from_0_25_to_0_5 \
    algorithm.train_timestep_range='[0,7]' \
    algorithm.train_timestep_count=4 \
    data.train_batch_size="$((4 * NUM_GPUS_ACTOR_ROLLOUT))" \
    +reward.models.ocr.rollout.temperature=0.0 \
    trainer.project_name=crd \
    trainer.experiment_name=sd35_medium_ocr_lora_v1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=disable \
    trainer.test_freq=10 \
    "$@"
