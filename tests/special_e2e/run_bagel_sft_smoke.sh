#!/usr/bin/env bash
# BAGEL SFT e2e smoke test (minimal runtime).
#
# Builds a tiny random-weight BAGEL checkpoint, then runs a couple of
# actor-only supervised training steps through verl_omni.trainer.main_omni.
# This validates the offline SFT path:
#   custom dataset -> BAGEL SFT FSDP engine -> CE + image MSE loss -> optimizer.
#
# Requires: verl, verl-omni, and GPU training dependencies installed.
# Override via env: NUM_GPUS, MODEL_PATH, TOTAL_TRAIN_STEPS
set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VERL_USE_EXTERNAL_MODULES=verl_omni

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/BAGEL-SFT}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ADAPTER="${REPO_ROOT}/tests/special_e2e/bagel_sft_smoke_dataset.py"

python3 "${REPO_ROOT}/tests/special_e2e/build_bagel_sft_tiny_random.py" \
    --output-dir "${MODEL_PATH}" --force

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${MODEL_PATH}" \
    data.val_files="${MODEL_PATH}" \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.dataloader_num_workers=0 \
    data.trust_remote_code=false \
    "data.custom_cls.path=file://${DATASET_ADAPTER}" \
    data.custom_cls.name=BagelSFTSmokeDataset \
    data.custom_cls.collate_fn=bagel_sft_smoke_collate_fn \
    algorithm.trainer_type=sft \
    algorithm.sample_source=offline \
    algorithm.paired_preference=false \
    algorithm.adv_estimator=bagel_sft \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.model_type=omni_sft_model \
    actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=false \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.omni_loss.loss_mode=omni_sft \
    actor_rollout_ref.actor.omni_loss.ce_weight=1.0 \
    actor_rollout_ref.actor.omni_loss.mse_weight=1.0 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=bagel-sft-e2e-smoke \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "BAGEL SFT e2e smoke test passed (training completed successfully)."
