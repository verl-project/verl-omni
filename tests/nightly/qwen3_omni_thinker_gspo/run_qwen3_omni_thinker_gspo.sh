#!/usr/bin/env bash
# Nightly Qwen3-Omni Thinker GSPO on the v1 omni trainer.
#
# Same wiring as tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_smoke.sh,
# on 1 GPU for 20 steps. Prompts are length-varied on purpose: equal-length 4-channel
# mRoPE position_ids hit a pinned-verl jagged layout bug (split_with_sizes) during
# compute_log_prob. ppo_micro_batch_size_per_gpu matches ppo_mini_batch_size so the
# actor update is one micro-batch: a smaller one enables FSDP2 no_sync, and torch 2.13
# then crashes on unused vision/audio params (FSDPParam has no _unsharded_param).
# The tiny checkpoint is built locally.
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export L3_TEST_CASE=${L3_TEST_CASE:-qwen3_omni_thinker_gspo}
export L3_ALGORITHM=${L3_ALGORITHM:-gspo}
export L3_CASE_LABEL=${L3_CASE_LABEL:-"Qwen3-Omni Thinker GSPO"}
export L3_MODEL=${L3_MODEL:-qwen3-omni}
export L3_DATA_SOURCE=${L3_DATA_SOURCE:-${L3_TEST_CASE}}
export NUM_GPUS=${NUM_GPUS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export L3_SKIP_DATASET=1
export NIGHTLY_TRAINER_MODULE=verl_omni.trainer.main_omni
export NIGHTLY_OMNI_DUMP=1
export VERL_USE_EXTERNAL_MODULES=verl_omni

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

DATA_DIR=${DATA_DIR:-${HOME}/data/${L3_TEST_CASE}}
export DATA_DIR
export TRAIN_FILES=${DATA_DIR}/train.parquet
export VAL_FILES=${DATA_DIR}/test.parquet
# Tiny model: always built locally (no external Hub dependency).
POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-${HOME}/models/tiny-random/Qwen3-Omni}

EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

# ── Build tiny model ──────────────────────────────────────────────────────────
python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
    --output-dir "${POLICY_MODEL_PATH}" --force

# ── Build dummy math dataset if not present ───────────────────────────────────
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_math_data.py" \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${TRAIN_BATCH_SIZE}" \
        --val_size 4
fi

# ── Run V1 training (20 steps, LoRA, GSPO, vLLM-Omni AR rollout) ───────────────
bash "${REPO_ROOT}/tests/nightly/run_l3_case.sh" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.val_max_samples=4 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.shuffle=false \
    data.seed=42 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="${POLICY_MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.rollout.do_sample=false \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    trainer.val_before_train=false \
    trainer.balance_batch=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl-nightly \
    trainer.experiment_name=qwen3-omni-thinker-gspo-nightly \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS:-20} \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=omni_sync \
    "$@"
