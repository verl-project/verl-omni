#!/usr/bin/env bash
# Qwen3-Omni multimodal offline MLLM DPO + LoRA e2e smoke test (minimal runtime).
#
# Flow:
#   build multimodal tiny-random checkpoint ->
#   dummy Omni-Preference parquet (image/video/audio) ->
#   OfflineMLLMDPODataset + ModalityGroupedBatchSampler ->
#   OmniDirectPreferenceRayTrainer (actor-only, offline) ->
#   FSDP LoRA actor update with ref-in-actor (base weights as reference) ->
#   save actor checkpoint ->
#   vlm_as_judge.py --stage trained on each saved LoRA checkpoint.
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
# LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES, LORA_TARGET_PARAMS,
# IMAGE_RATIO, VIDEO_RATIO, AUDIO_RATIO, VAL_BATCH_SIZE, TEST_FREQ,
# EVAL_MAX_SAMPLES, EVAL_GENERATION_MAX_TOKENS, EVAL_DEVICE_MAP, EVAL_CUDA_DEVICES
set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_omni_preference_dpo}
TRAIN_SIZE=${TRAIN_SIZE:-8}
VAL_SIZE=${VAL_SIZE:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'["q_proj","k_proj","v_proj","o_proj"]'}
LORA_TARGET_PARAMS=${LORA_TARGET_PARAMS:-'["gate_up_proj","down_proj"]'}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1}
EVAL_GENERATION_MAX_TOKENS=${EVAL_GENERATION_MAX_TOKENS:-16}
EVAL_DEVICE_MAP=${EVAL_DEVICE_MAP:-cuda}
EVAL_CUDA_DEVICES=${EVAL_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
IFS=' ' read -r -a EVAL_MODALITIES <<< "${EVAL_MODALITIES:-image video audio}"
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
TEST_FREQ=${TEST_FREQ:-1}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
IMAGE_RATIO=${IMAGE_RATIO:-1.0}
VIDEO_RATIO=${VIDEO_RATIO:-1.0}
AUDIO_RATIO=${AUDIO_RATIO:-1.0}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_NAME=verl-test
EXPERIMENT_NAME=qwen3-omni-multimodal-offline-mllm-dpo-lora-smoke
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ADAPTER_PATH="${CHECKPOINT_DIR}/global_step_${TOTAL_TRAINING_STEPS}/actor"

# Thinker-only LoRA: strip talker / code2wav / vision tower / audio tower.
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="${HOME}/models/tiny-random/Qwen3-Omni-Multimodal"
fi

if [ "${NUM_GPUS}" -lt 2 ]; then
    echo "Warning: NUM_GPUS=${NUM_GPUS}; LoRA FSDP smoke is most reliable with NUM_GPUS>=2." >&2
fi

# Build or refresh the multimodal tiny-random checkpoint when missing or incomplete.
python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_multimodal_tiny_random.py" \
    --output-dir "${MODEL_PATH}"

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
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=512 \
    data.trust_remote_code=true \
    data.filter_overlong_prompts=false \
    +data.pad_mode=no_padding \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    data.custom_cls.name=OfflineMLLMDPODataset \
    data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    +data.train_sampler.class_path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    +data.train_sampler.class_name=ModalityGroupedBatchSampler \
    +data.train_sampler.sampler_kwargs="{batch_size:${TRAIN_BATCH_SIZE},drop_last:true,modality_sample_weights:{image:${IMAGE_RATIO},video:${VIDEO_RATIO},audio:${AUDIO_RATIO}}}" \
    +data.val_sampler.class_path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    +data.val_sampler.class_name=ModalityGroupedBatchSampler \
    +data.val_sampler.sampler_kwargs="{batch_size:${VAL_BATCH_SIZE},shuffle:false,drop_last:true,replacement:false}" \
    +data.mm_configs="{scale_factor:28,image_min_pixels:3136,image_max_pixels:602112,video_min_pixels:3136,video_max_pixels:602112,max_ratio:200,min_frames:2,max_frames:4,frame_factor:2,sample_rate:16000,fps:2.0,use_audio_in_video:false}" \
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
    actor_rollout_ref.model.target_parameters="${LORA_TARGET_PARAMS}" \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
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
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
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

# Trained-stage generation on saved LoRA checkpoints.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${CHECKPOINT_DIR}/vlm_as_judge}"
mkdir -p "${EVAL_OUT_DIR}"

if [ -n "${EVAL_STEPS:-}" ]; then
    read -r -a STEPS <<< "${EVAL_STEPS}"
else
    STEPS=()
    for ((step = 1; step <= TOTAL_TRAINING_STEPS; step++)); do
        STEPS+=("${step}")
    done
fi

if [ "${#STEPS[@]}" -eq 0 ]; then
    echo "No checkpoints selected for trained-stage eval under ${CHECKPOINT_DIR}" >&2
    exit 1
fi

# export HF_ENABLE_PARALLEL_LOADING=true
# export HF_PARALLEL_LOADING_WORKERS=8
# export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICES}"

# # vlm_as_judge.py imports qwen_omni_utils for Qwen3-Omni multimodal inputs.
# python3 -m pip install --no-cache-dir "qwen-omni-utils"
# pip install --upgrade transformers peft

# for step in "${STEPS[@]}"; do
#     trained_jsonl="${EVAL_OUT_DIR}/global_step_${step}.trained.jsonl"
#     python3 "${REPO_ROOT}/examples/dpo_trainer/qwen3_omni/vlm_as_judge.py" \
#         --data-dir "${DATA_DIR}" \
#         --modalities "${EVAL_MODALITIES[@]}" \
#         --output-jsonl "${trained_jsonl}" \
#         --stage trained \
#         --max-samples "${EVAL_MAX_SAMPLES}" \
#         --model-path "${MODEL_PATH}" \
#         --device-map "${EVAL_DEVICE_MAP}" \
#         --generation-max-tokens "${EVAL_GENERATION_MAX_TOKENS}" \
#         --adapter-path "${CHECKPOINT_DIR}/global_step_${step}"
#     if [ ! -s "${trained_jsonl}" ]; then
#         echo "Expected trained-stage jsonl not found at ${trained_jsonl}" >&2
#         exit 1
#     fi
# done

echo "Qwen3-Omni multimodal offline MLLM DPO + LoRA smoke test passed."
