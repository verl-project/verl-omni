#!/usr/bin/env bash
# MiniCPM-o style offline MLLM DPO + LoRA e2e smoke test (minimal runtime).
#
# This is a plumbing smoke test with a local tiny-random remote-code checkpoint:
# it validates AutoModel loading, MiniCPM transform dispatch, offline DPO loss,
# and FSDP LoRA checkpointing. It does not measure model quality.
set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/MiniCPM-o-4_5}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_minicpm_preference_dpo}
TRAIN_SIZE=${TRAIN_SIZE:-8}
VAL_SIZE=${VAL_SIZE:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
TEST_FREQ=${TEST_FREQ:-1}
IMAGE_RATIO=${IMAGE_RATIO:-1.0}
VIDEO_RATIO=${VIDEO_RATIO:-0.0}
AUDIO_RATIO=${AUDIO_RATIO:-1.0}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_NAME=verl-test
EXPERIMENT_NAME=minicpm-o-multimodal-offline-mllm-dpo-lora-smoke
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ADAPTER_PATH="${CHECKPOINT_DIR}/global_step_${TOTAL_TRAINING_STEPS}/actor"

# Keep MiniCPM understanding modules trainable; exclude only generation-only audio/TTS paths.
EXCLUDE_MODULES=${EXCLUDE_MODULES:-".*talker.*|.*code2wav.*|.*code_predictor.*|.*codec.*|.*audio_decoder.*|.*audio_generator.*|.*audio_head.*|.*tts.*|.*vocoder.*"}

python3 "${REPO_ROOT}/tests/special_e2e/build_minicpm_o_tiny_random.py" \
    --output-dir "${MODEL_PATH}" \
    --force

if [ ! -f "${DATA_DIR}/image/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_omni_preference_dpo_data.py" \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${TRAIN_SIZE}" \
        --val_size "${VAL_SIZE}"
fi

TRAIN_FILES="['${DATA_DIR}/image/train.parquet','${DATA_DIR}/audio/train.parquet']"
VAL_FILES="['${DATA_DIR}/image/test.parquet','${DATA_DIR}/audio/test.parquet']"

python3 -m verl_omni.trainer.main_omni \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=offline \
    algorithm.paired_preference=true \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=512 \
    data.trust_remote_code=true \
    data.filter_overlong_prompts=false \
    +data.base_transform=minicpm \
    +data.pad_mode=no_padding \
    +data.max_length=1024 \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    data.custom_cls.name=OfflineMLLMDPODataset \
    data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    +data.train_sampler.class_path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    +data.train_sampler.class_name=ModalityGroupedBatchSampler \
    +data.train_sampler.sampler_kwargs="{batch_size:${TRAIN_BATCH_SIZE},drop_last:true,modality_sample_weights:{image:${IMAGE_RATIO},video:${VIDEO_RATIO},audio:${AUDIO_RATIO}}}" \
    +data.val_sampler.class_path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    +data.val_sampler.class_name=ModalityGroupedBatchSampler \
    +data.val_sampler.sampler_kwargs="{batch_size:${VAL_BATCH_SIZE},shuffle:false,drop_last:true,replacement:false}" \
    +data.mm_configs="{image_min_pixels:3136,image_max_pixels:3136,max_ratio:20,sample_rate:16000,max_slice_nums:1}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.hf_config_path="${MODEL_PATH}" \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +actor_rollout_ref.model.override_config.init_tts=false \
    +actor_rollout_ref.model.override_config.use_cache=false \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj"]' \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.trainer_type=direct_preference \
    actor_rollout_ref.actor.omni_loss.loss_mode=dpo \
    actor_rollout_ref.actor.omni_loss.beta=0.1 \
    actor_rollout_ref.actor.omni_loss.label_smoothing=0.0 \
    actor_rollout_ref.actor.omni_loss.loss_type=sigmoid \
    actor_rollout_ref.actor.omni_loss.average_log_prob=false \
    actor_rollout_ref.actor.omni_loss.refer_model_precision=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1.0e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.shuffle=false \
    trainer.resume_mode=disable \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.val_before_train=false \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"

if [ ! -f "${ADAPTER_PATH}/fsdp_config.json" ] || [ ! -f "${ADAPTER_PATH}/lora_train_meta.json" ]; then
    echo "Expected FSDP LoRA actor checkpoint not found at ${ADAPTER_PATH}" >&2
    exit 1
fi

echo "MiniCPM-o style offline MLLM DPO + LoRA smoke test passed."
