#!/usr/bin/env bash
# Smoke test for Qwen3-Omni offline MLLM DPO through verl-omni entrypoint.
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_omni_preference_dpo}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
IMAGE_RATIO=${IMAGE_RATIO:-1.0}
VIDEO_RATIO=${VIDEO_RATIO:-1.0}
AUDIO_RATIO=${AUDIO_RATIO:-1.0}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("veomni") is not None else 1)
PY
then
    echo "Skipping Qwen3-Omni verl-omni VeOmni DPO smoke: optional package \`veomni\` is not installed." >&2
    exit 0
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="${HOME}/models/tiny-random/Qwen3-Omni-mllm"
    [ -d "${MODEL_PATH}" ] || python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
        --output-dir "${MODEL_PATH}"
fi

python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_omni_preference_dpo_data.py" \
    --local_save_dir "${DATA_DIR}" \
    --train_size 2 \
    --val_size 1

python3 -m verl_omni.trainer.main_omni \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=offline \
    algorithm.paired_preference=true \
    data.train_files="['${DATA_DIR}/image/train.parquet','${DATA_DIR}/video/train.parquet','${DATA_DIR}/audio/train.parquet']" \
    data.val_files="['${DATA_DIR}/image/test.parquet','${DATA_DIR}/video/test.parquet','${DATA_DIR}/audio/test.parquet']" \
    data.train_batch_size=3 \
    data.max_prompt_length=512 \
    data.trust_remote_code=true \
    data.filter_overlong_prompts=false \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    data.custom_cls.name=OfflineMLLMDPODataset \
    data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    data.sampler.class_name=ModalityBatchSampler \
    data.sampler.batch_size=3 \
    data.sampler.drop_last=true \
    data.sampler.modality_ratios.image="${IMAGE_RATIO}" \
    data.sampler.modality_ratios.video="${VIDEO_RATIO}" \
    data.sampler.modality_ratios.audio="${AUDIO_RATIO}" \
    +data.mm_configs="{scale_factor:28,image_min_pixels:3136,image_max_pixels:12845056,video_min_pixels:3136,video_max_pixels:602112,max_ratio:200,min_frames:2,max_frames:4,frame_factor:1,sample_rate:16000,fps:2.0,use_audio_in_video:false}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.model_path="${MODEL_PATH}" \
    actor_rollout_ref.model.config_path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    actor_rollout_ref.actor.omni_loss.loss_mode=dpo \
    actor_rollout_ref.actor.omni_loss.beta=0.1 \
    actor_rollout_ref.actor.omni_loss.label_smoothing=0.0 \
    actor_rollout_ref.actor.omni_loss.loss_type=sigmoid \
    actor_rollout_ref.actor.omni_loss.reference_free=false \
    actor_rollout_ref.actor.omni_loss.average_log_prob=false \
    actor_rollout_ref.actor.omni_loss.refer_model_precision=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1.0e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.veomni_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.veomni_config.init_device=meta \
    actor_rollout_ref.actor.veomni_config.param_offload=false \
    actor_rollout_ref.actor.veomni_config.optimizer_offload=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    trainer.resume_mode=disable \
    trainer.logger='["console"]' \
    trainer.project_name=verl-test \
    trainer.experiment_name=qwen3-omni-entrypoint-veomni-dpo-smoke \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" "$@"

echo "Qwen3-Omni verl-omni VeOmni DPO smoke test passed."
