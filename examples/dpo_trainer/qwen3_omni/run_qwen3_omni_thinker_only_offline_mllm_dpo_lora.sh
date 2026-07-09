#!/usr/bin/env bash
# Qwen3-Omni Thinker offline MLLM DPO + LoRA (FSDP, 4× H100 80GB).
#
# Trains on LLaVA-Hound-DPO preference pairs (image, text, video multisource).
# Dataset preparation: docs/datasets/llava_hound_dpo_dataset.md
#
# Quick sanity check (no GPU, data pipeline only):
#   python3 examples/dpo_trainer/qwen3_omni/verify_data_pipeline.py \
#       --train_files $DATA_DIR/image/train.parquet $DATA_DIR/text/train.parquet $DATA_DIR/video/train.parquet \
#       --batch_size 4 --max_samples 64
#
# Default image + text + video multisource training:
#   bash examples/dpo_trainer/qwen3_omni/run_qwen3_omni_thinker_only_offline_mllm_dpo_lora.sh
#
# Image + text + video multisource:
#   TRAIN_FILE="[$DATA_DIR/image/train.parquet,$DATA_DIR/text/train.parquet,$DATA_DIR/video/train.parquet]" \
#   VAL_FILE="[$DATA_DIR/image/test.parquet,$DATA_DIR/text/test.parquet,$DATA_DIR/video/test.parquet]" \
#   bash examples/dpo_trainer/qwen3_omni/run_qwen3_omni_thinker_only_offline_mllm_dpo_lora.sh
set -x

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Load verl_omni (custom collate fn + dataset utils) and the Qwen3-Omni model
# patches (processor / AutoModel registration) on both the driver and workers.
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_omni_thinker

# ---------------------------------------------------------------------------
# Paths — override via environment variables.
# ---------------------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}

DATA_DIR=${DATA_DIR:-"$HOME/data/llava_hound_dpo/parquet"}

# Default: image + text + video multisource.
TRAIN_FILE=${TRAIN_FILE:-"[$DATA_DIR/image/train.parquet,$DATA_DIR/text/train.parquet,$DATA_DIR/video/train.parquet]"}
VAL_FILE=${VAL_FILE:-"[$DATA_DIR/image/test.parquet,$DATA_DIR/text/test.parquet,$DATA_DIR/video/test.parquet]"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/../../../gspo_trainer/qwen3_omni/qwen3_omni_thinker_only.yaml"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
python3 -m verl_omni.trainer.main_omni \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_omni_thinker_offline_mllm_dpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.val_max_samples=64 \
    data.filter_overlong_prompts=true \
    data.truncation=left \
    ++data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    ++data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*' \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=dpo \
    actor_rollout_ref.actor.policy_loss.dpo_beta=0.1 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=100000000 \
    algorithm.sample_source=offline \
    algorithm.adv_estimator=dpo \
    algorithm.use_kl_in_reward=false \
    reward.reward_manager.name=naive \
    trainer.val_before_train=false \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=qwen3_omni_offline_mllm_dpo \
    trainer.experiment_name=offline_dpo_lora_llava_hound \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=3 \
    "$@"
