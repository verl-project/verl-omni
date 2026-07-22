#!/usr/bin/env bash
# Qwen3-Omni multimodal offline MLLM DPO + LoRA e2e smoke test (minimal runtime).
#
# Flow:
#   build multimodal tiny-random checkpoint ->
#   dummy Omni-Preference parquet (image/video/audio) ->
#   OfflineMLLMDPODataset + ModalityGroupedBatchSampler ->
#   OmniDirectPreferenceRayTrainer (actor-only, offline) ->
#   FSDP LoRA actor update with ref-in-actor (base weights as reference).
#
# This is a plumbing smoke test (random weights, 1–2 steps): it checks the
# pipeline runs without errors, NOT model quality.
#
# Requires: verl, verl-omni installed with GPU support.
#   * tiny multimodal checkpoint at MODEL_PATH (auto-built if missing)
#   * dummy parquet under DATA_DIR/{image,video,audio}/{train,test}.parquet
#   * multimodal deps for real tokenization: torchvision, av, librosa
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TRAIN_SIZE, VAL_SIZE,
# TOTAL_TRAINING_STEPS, PPO_MINI_BATCH_SIZE, PPO_MICRO_BATCH_SIZE_PER_GPU,
# LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES, LORA_TARGET_PARAMETERS,
# IMAGE_RATIO, VIDEO_RATIO, AUDIO_RATIO
set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_omni_preference_dpo}
TRAIN_SIZE=${TRAIN_SIZE:-2}
VAL_SIZE=${VAL_SIZE:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}
LORA_TARGET_PARAMETERS=${LORA_TARGET_PARAMETERS:-'["gate_up_proj","down_proj"]'}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
IMAGE_RATIO=${IMAGE_RATIO:-1.0}
VIDEO_RATIO=${VIDEO_RATIO:-1.0}
AUDIO_RATIO=${AUDIO_RATIO:-1.0}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Thinker-only LoRA: strip talker / code2wav / vision tower / audio tower.
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="${HOME}/models/tiny-random/Qwen3-Omni-Multimodal"
fi

if [ "${NUM_GPUS}" -lt 2 ]; then
    echo "Warning: NUM_GPUS=${NUM_GPUS}; LoRA FSDP smoke is most reliable with NUM_GPUS>=2." >&2
fi

# ── Build multimodal tiny-random checkpoint if missing ───────────────────────
if [ ! -f "${MODEL_PATH}/config.json" ] || [ ! -f "${MODEL_PATH}/chat_template.json" ] || [ ! -f "${MODEL_PATH}/preprocessor_config.json" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_multimodal_tiny_random.py" \
        --output-dir "${MODEL_PATH}"
fi

# ── Build dummy Omni-Preference parquet if missing ───────────────────────────
if [ ! -f "${DATA_DIR}/image/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_omni_preference_dpo_data.py" \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${TRAIN_SIZE}" \
        --val_size "${VAL_SIZE}"
fi

TRAIN_FILES="['${DATA_DIR}/image/train.parquet','${DATA_DIR}/video/train.parquet','${DATA_DIR}/audio/train.parquet']"
VAL_FILES="['${DATA_DIR}/image/test.parquet','${DATA_DIR}/video/test.parquet','${DATA_DIR}/audio/test.parquet']"

# ── Run offline multimodal DPO + LoRA (FSDP actor-only; no rollout/reward) ───
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
    +data.sampler.sampler_kwargs="{batch_size:${TRAIN_BATCH_SIZE},drop_last:true,modality_sample_weights:{image:${IMAGE_RATIO},video:${VIDEO_RATIO},audio:${AUDIO_RATIO}}}" \
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
    actor_rollout_ref.model.target_parameters="${LORA_TARGET_PARAMETERS}" \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=false \
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
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.shuffle=false \
    trainer.resume_mode=disable \
    trainer.logger='["console"]' \
    trainer.project_name=verl-test \
    trainer.experiment_name=qwen3-omni-multimodal-offline-mllm-dpo-lora-smoke \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"

echo "Qwen3-Omni multimodal offline MLLM DPO + LoRA smoke test passed."
