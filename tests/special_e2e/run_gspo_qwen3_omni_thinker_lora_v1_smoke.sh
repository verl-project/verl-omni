#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA e2e V1 smoke test (minimal runtime).
#
# Uses the V1 trainer (verl_omni.trainer.main_omni) with pipeline_name=qwen3_omni_moe
# to auto-generate the deploy config — no stage YAML or external_lib needed.
# Follows examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh
# settings; only down-scales batch size, rollout.n, prompt/response length, and LoRA
# for smoke testing (tiny random model, 2 steps).
#
# Requires: verl, verl-omni, vllm-omni installed.
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -xeuo pipefail

# The image is built with transformers 5.3.0; uv pip install ".[gpu,dev]" upgrades
# to latest (5.14.x), which breaks vllm-omni (tie_word_embeddings removed) and
# accelerate (unpatched _is_hf_initialized). Match our known-good local versions.
uv pip install --system --break-system-packages transformers==5.12.1 accelerate==1.14.0 peft==0.19.1

# NCCL / accelerator env guards
export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# V1 path: verl_omni only — pipeline registration is automatic.
export VERL_USE_EXTERNAL_MODULES=verl_omni

NUM_GPUS=${NUM_GPUS:-2}
# Tiny model: always built locally (no external Hub dependency).
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Same Thinker-only module filter as the V1 example recipe.
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

# ── Build tiny model ──────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/tiny-random/Qwen3-Omni}"
python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
    --output-dir "${MODEL_PATH}" --force

# ── Build dummy math dataset if not present ───────────────────────────────────
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_math_data.py" \
        --local_save_dir "${DATA_DIR}"
fi

# ── Run V1 training (2 steps, LoRA, GSPO, vLLM-Omni AR rollout) ───────────────
# Mirrors examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh
# with smoke-scale overrides (batch=4, rollout.n=2, short prompt/response, r=8).
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
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
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
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
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=gspo-qwen3-omni-thinker-lora-e2e-v1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "Qwen3-Omni Thinker GSPO+LoRA e2e V1 smoke test passed (training completed successfully)."
