#!/usr/bin/env bash
# MiniCPM offline DPO + LoRA training on Omni-Preference style parquet data.
#
# Prepare data first with:
#   python examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py \
#       --dataset_root "$HOME/Omni-Preference" \
#       --output_dir "$HOME/Omni-Preference/parquet_dpo"
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export WANDB_MODE=${WANDB_MODE:-online}
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/cuda-compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    echo "CONDA_PREFIX is unset; skipping cuda-compat LD_LIBRARY_PATH injection." >&2
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_ROOT=${DATASET_ROOT:-${HOME}/Omni-Preference}
MODEL_PATH=${MODEL_PATH:-openbmb/MiniCPM-o-4_5}
DATA_DIR=${DATA_DIR:-${DATASET_ROOT}/parquet_dpo}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
NUM_GPUS=${NUM_GPUS:-4}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'["q_proj","k_proj","v_proj","o_proj"]'}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
LR=${LR:-1.0e-6}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-100}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-96}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_SLICE_NUMS=${MAX_SLICE_NUMS:-1}

IMAGE_RATIO=${IMAGE_RATIO:-1.0}
AUDIO_RATIO=${AUDIO_RATIO:-1.0}

# MiniCPM training uses image-only and audio-only batches only
TRAIN_FILES=${TRAIN_FILES:-"['${DATA_DIR}/image/train.parquet','${DATA_DIR}/audio/train.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/image/test.parquet','${DATA_DIR}/audio/test.parquet']"}

for parquet in \
    "${DATA_DIR}/image/train.parquet" \
    "${DATA_DIR}/audio/train.parquet" \
    "${DATA_DIR}/image/test.parquet" \
    "${DATA_DIR}/audio/test.parquet"; do
    if [ ! -f "${parquet}" ]; then
        echo "Missing MiniCPM offline DPO parquet: ${parquet}" >&2
        exit 1
    fi
done

# Train MiniCPM understanding modules; exclude only generation-only audio/TTS paths.
EXCLUDE_MODULES=${EXCLUDE_MODULES:-".*talker.*|.*code2wav.*|.*code_predictor.*|.*codec.*|.*audio_decoder.*|.*audio_generator.*|.*audio_head.*|.*tts.*|.*vocoder.*"}

python3 -m verl_omni.trainer.main_omni \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=offline \
    algorithm.paired_preference=true \
    data.train_files="${TRAIN_FILES}" \
    data.dataloader_num_workers=2 \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    +data.balance_max_samples_by_modality=true \
    +data.base_transform=minicpm \
    +data.pad_mode=no_padding \
    +data.max_length="${MAX_LENGTH}" \
    data.truncation=right \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    data.custom_cls.name=OfflineMLLMDPODataset \
    data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    +data.train_sampler.class_path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    +data.train_sampler.class_name=ModalityGroupedBatchSampler \
    +data.train_sampler.sampler_kwargs="{batch_size:${TRAIN_BATCH_SIZE},drop_last:true,num_batches:${TOTAL_TRAINING_STEPS},modality_sample_weights:{image:${IMAGE_RATIO},audio:${AUDIO_RATIO}}}" \
    +data.val_sampler.class_path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    +data.val_sampler.class_name=ModalityGroupedBatchSampler \
    +data.val_sampler.sampler_kwargs="{batch_size:${VAL_BATCH_SIZE},shuffle:false,drop_last:true,replacement:false}" \
    +data.mm_configs="{image_min_pixels:3136,image_max_pixels:602112,max_ratio:20,sample_rate:16000,max_slice_nums:${MAX_SLICE_NUMS}}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.hf_config_path="${MODEL_PATH}" \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
    +actor_rollout_ref.model.override_config.init_tts=false \
    +actor_rollout_ref.model.override_config.use_cache=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.lora.dropout="${LORA_DROPOUT}" \
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.trainer_type=direct_preference \
    actor_rollout_ref.actor.omni_loss.loss_mode=dpo \
    actor_rollout_ref.actor.omni_loss.beta=0.1 \
    actor_rollout_ref.actor.omni_loss.label_smoothing=0.1 \
    actor_rollout_ref.actor.omni_loss.loss_type=sigmoid \
    actor_rollout_ref.actor.omni_loss.average_log_prob=false \
    actor_rollout_ref.actor.omni_loss.refer_model_precision=bfloat16 \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.num_cycles=0.5 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.shuffle=false \
    trainer.resume_mode=disable \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=omni-preference-dpo \
    trainer.experiment_name=minicpm-offline-dpo-lora \
    trainer.default_local_dir=checkpoints/omni-preference-dpo/minicpm-offline-dpo-lora \
    trainer.val_before_train=true \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
