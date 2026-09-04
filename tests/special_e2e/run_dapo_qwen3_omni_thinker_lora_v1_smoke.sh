#!/usr/bin/env bash
# Qwen3-Omni Thinker DAPO + LoRA V1 smoke without dynamic sampling.

set -xeuo pipefail

if [[ "${SKIP_COMPAT_DEPS_INSTALL:-0}" != "1" ]]; then
    uv pip install --system --break-system-packages transformers==5.12.1 accelerate==1.14.0 peft==0.19.1
fi

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VERL_USE_EXTERNAL_MODULES=verl_omni

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

MODEL_PATH="${MODEL_PATH:-${HOME}/models/tiny-random/Qwen3-Omni}"
python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
    --output-dir "${MODEL_PATH}" --force

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_math_data.py" \
        --local_save_dir "${DATA_DIR}"
fi

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.val_max_samples=4 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.filter_groups.enable=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.max_resp_len=512 \
    reward.reward_kwargs.overlong_buffer_cfg.enable=true \
    reward.reward_kwargs.overlong_buffer_cfg.len=128 \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    reward.reward_kwargs.overlong_buffer_cfg.log=true \
    trainer.val_before_train=false \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=dapo-qwen3-omni-thinker-lora-e2e-v1-wo-dynamic-sampling \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "Qwen3-Omni Thinker DAPO+LoRA e2e V1 smoke without dynamic sampling passed."
