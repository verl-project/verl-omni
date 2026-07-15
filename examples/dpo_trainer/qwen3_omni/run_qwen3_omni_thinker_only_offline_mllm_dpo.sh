#!/usr/bin/env bash
# Qwen3-Omni offline MLLM DPO through verl-omni entrypoint with VeOmni backend.
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL_PATH=${MODEL_PATH:-${HOME}/models/Qwen/Qwen3-Omni-30B-A3B-Thinking}
DATA_DIR=${DATA_DIR:-${HOME}/data/omni_preference_dpo}
TRAIN_FILES=${TRAIN_FILES:-"['${DATA_DIR}/image/train.parquet','${DATA_DIR}/video/train.parquet','${DATA_DIR}/audio/train.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/image/test.parquet','${DATA_DIR}/video/test.parquet','${DATA_DIR}/audio/test.parquet']"}
NUM_GPUS_ACTOR=${NUM_GPUS_ACTOR:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${NUM_GPUS_ACTOR}}
LR=${LR:-1.0e-6}
DPO_BETA=${DPO_BETA:-0.1}
IMAGE_RATIO=${IMAGE_RATIO:-1.0}
VIDEO_RATIO=${VIDEO_RATIO:-1.0}
AUDIO_RATIO=${AUDIO_RATIO:-1.0}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-262144}

export PYTHONPATH="${SCRIPT_DIR}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 -m verl_omni.trainer.main_omni \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=offline \
    algorithm.paired_preference=true \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=256 \
    data.trust_remote_code=true \
    data.filter_overlong_prompts=false \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    data.custom_cls.name=OfflineMLLMDPODataset \
    data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    data.sampler.class_name=ModalityBatchSampler \
    data.sampler.drop_last=true \
    data.sampler.modality_ratios.image="${IMAGE_RATIO}" \
    data.sampler.modality_ratios.video="${VIDEO_RATIO}" \
    data.sampler.modality_ratios.audio="${AUDIO_RATIO}" \
    +data.mm_configs="{scale_factor:28,image_min_pixels:3136,image_max_pixels:${IMAGE_MAX_PIXELS},video_min_pixels:3136,video_max_pixels:262144,max_ratio:200,min_frames:2,max_frames:2,frame_factor:1,sample_rate:16000,fps:1.0,use_audio_in_video:false}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.architecture=Qwen3OmniMoeForConditionalGeneration \
    actor_rollout_ref.model.algorithm=dpo \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.model_path="${MODEL_PATH}" \
    actor_rollout_ref.model.config_path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.external_lib='["verl_omni.models.transformers.qwen3_omni_thinker","verl_omni.pipelines.qwen3_omni_dpo"]' \
    actor_rollout_ref.actor.omni_loss.loss_mode=dpo \
    actor_rollout_ref.actor.omni_loss.beta="${DPO_BETA}" \
    actor_rollout_ref.actor.omni_loss.label_smoothing=0.0 \
    actor_rollout_ref.actor.omni_loss.loss_type=sigmoid \
    actor_rollout_ref.actor.omni_loss.reference_free=false \
    actor_rollout_ref.actor.omni_loss.average_log_prob=false \
    actor_rollout_ref.actor.omni_loss.refer_model_precision=bfloat16 \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.veomni_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.veomni_config.init_device=meta \
    actor_rollout_ref.actor.veomni_config.param_offload=true \
    actor_rollout_ref.actor.veomni_config.reshard_after_forward=false \
    actor_rollout_ref.actor.veomni_config.optimizer_offload=true \
    actor_rollout_ref.actor.veomni_config.enable_activation_offload=true \
    actor_rollout_ref.actor.veomni_config.activation_gpu_limit=4.0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.veomni_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.veomni_config.init_device=meta \
    actor_rollout_ref.ref.veomni_config.forward_only=true \
    actor_rollout_ref.ref.veomni_config.param_offload=true \
    actor_rollout_ref.ref.veomni_config.reshard_after_forward=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    trainer.resume_mode=disable \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=qwen3_omni_offline_dpo \
    trainer.experiment_name=qwen3_omni_thinker_veomni_dpo \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node="${NUM_GPUS_ACTOR}" \
    trainer.save_freq=30 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" "$@"
