#!/usr/bin/env bash
# Qwen3-Omni offline DPO + LoRA training on Omni-Preference.
#
# Defaults are set for a short real-data run (~100 optimizer steps). Override
# paths and batch sizes with environment variables, or append Hydra overrides.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/.venv/bin/activate"

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export WANDB_MODE=${WANDB_MODE:-online}
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/cuda-compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    echo "CONDA_PREFIX is unset; skipping cuda-compat LD_LIBRARY_PATH injection." >&2
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH=${MODEL_PATH:-/scratch/fq9hpsac/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}
DATA_DIR=${DATA_DIR:-${HOME}/didan-new/Omni-Preference/parquet_dpo}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
NUM_GPUS=${NUM_GPUS:-4}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${NUM_GPUS}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}

LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']"}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LR=${LR:-1.0e-6}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:--1}

IMAGE_RATIO=${IMAGE_RATIO:-1.0}
VIDEO_RATIO=${VIDEO_RATIO:-1.0}
AUDIO_RATIO=${AUDIO_RATIO:-1.0}

TRAIN_FILES=${TRAIN_FILES:-"['${DATA_DIR}/image/train.parquet','${DATA_DIR}/video/train.parquet','${DATA_DIR}/audio/train.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/image/test.parquet','${DATA_DIR}/video/test.parquet','${DATA_DIR}/audio/test.parquet']"}

for parquet in \
    "${DATA_DIR}/image/train.parquet" \
    "${DATA_DIR}/video/train.parquet" \
    "${DATA_DIR}/audio/train.parquet" \
    "${DATA_DIR}/image/test.parquet" \
    "${DATA_DIR}/video/test.parquet" \
    "${DATA_DIR}/audio/test.parquet"; do
    if [ ! -f "${parquet}" ]; then
        echo "Missing Omni-Preference parquet: ${parquet}" >&2
        exit 1
    fi
done

# Thinker-only LoRA: leave talker / code2wav / visual / audio towers frozen.
EXCLUDE_MODULES=${EXCLUDE_MODULES:-".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"}

python3 -m verl_omni.trainer.main_omni \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=offline \
    algorithm.paired_preference=true \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=512 \
    data.trust_remote_code=true \
    data.filter_overlong_prompts=false \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    data.custom_cls.name=OfflineMLLMDPODataset \
    data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    data.sampler.class_path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    data.sampler.class_name=ModalityGroupedBatchSampler \
    +data.sampler.sampler_kwargs="{batch_size:${TRAIN_BATCH_SIZE},drop_last:true,num_batches:${TOTAL_TRAINING_STEPS},modality_sample_weights:{image:${IMAGE_RATIO},video:${VIDEO_RATIO},audio:${AUDIO_RATIO}}}" \
    +data.mm_configs="{scale_factor:28,image_min_pixels:3136,image_max_pixels:12845056,video_min_pixels:3136,video_max_pixels:602112,max_ratio:200,min_frames:2,max_frames:4,frame_factor:1,sample_rate:16000,fps:2.0,use_audio_in_video:false}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.hf_config_path="${MODEL_PATH}" \
    actor_rollout_ref.model.architecture=Qwen3OmniMoeForConditionalGeneration \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.trainer_type=direct_preference \
    actor_rollout_ref.actor.omni_loss.loss_mode=dpo \
    actor_rollout_ref.actor.omni_loss.beta=0.1 \
    actor_rollout_ref.actor.omni_loss.label_smoothing=0.0 \
    actor_rollout_ref.actor.omni_loss.loss_type=sigmoid \
    actor_rollout_ref.actor.omni_loss.average_log_prob=false \
    actor_rollout_ref.actor.omni_loss.refer_model_precision=bfloat16 \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.shuffle=false \
    trainer.resume_mode=disable \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=omni-preference-dpo \
    trainer.experiment_name=qwen3-omni-offline-dpo-lora-100steps \
    trainer.default_local_dir="checkpoints/omni-preference-dpo/qwen3-omni-offline-dpo-lora-100steps" \
    trainer.val_before_train=false \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
