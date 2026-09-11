#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen-Image}
NUM_GPUS=${NUM_GPUS:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${NUM_GPUS}}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen_image_dmd2}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1000}

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.trainer_type=distribution_matching \
    algorithm.sample_source=offline \
    data.train_files="${TRAIN_FILES:?Set TRAIN_FILES to prompt parquet}" \
    data.val_files="${VAL_FILES:?Set VAL_FILES to prompt parquet}" \
    data.train_batch_size="${GLOBAL_BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.algorithm=dmd2 \
    actor_rollout_ref.model.model_type=diffusion_dmd_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules='[to_q,to_k,to_v,to_out.0]' \
    actor_rollout_ref.model.pipeline.height=1024 \
    actor_rollout_ref.model.pipeline.width=1024 \
    actor_rollout_ref.model.pipeline.num_inference_steps=4 \
    actor_rollout_ref.model.pipeline.max_sequence_length=1024 \
    actor_rollout_ref.actor.strategy="${STRATEGY:-fsdp2}" \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.001 \
    dmd.fake_update_ratio=2 \
    dmd.student_micro_batch_size_per_gpu="${STUDENT_MICRO_BATCH_SIZE:-1}" \
    dmd.fake_score_micro_batch_size_per_gpu="${FAKE_MICRO_BATCH_SIZE:-1}" \
    dmd.export_role=student \
    trainer.logger='[console,tensorboard]' \
    trainer.project_name=qwen-image-dmd2 \
    trainer.experiment_name="${EXPERIMENT_NAME:-distribution-only}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq="${SAVE_FREQ:-100}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-32}" \
    "$@"
