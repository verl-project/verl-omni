#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker GSPO + LoRA e2e smoke test (minimal runtime, collocated sync).
#
# Uses the V1 trainer (verl_omni.trainer.main_omni) with
# pipeline_name=minicpmo_4_5 (the runtime-cloned thinker-only stage) and a tiny
# random remote-code checkpoint. Mirrors
# examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh
# wiring; only down-scales model size (tiny random weights), batch, rollout.n,
# prompt/response length, LoRA rank, and GPU count (2 steps). Coverage is
# structural: random-model reward accuracy is not a gate.
#
# Requires: verl, verl-omni, vllm-omni, soundfile, torchaudio, Pillow.
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -xeuo pipefail

# NCCL / accelerator env guards
export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# V1 path: verl_omni only — pipeline registration is automatic.
export VERL_USE_EXTERNAL_MODULES=verl_omni

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/minicpmo_avqa_smoke}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Same frozen-tower exclusion as the recipe: vpm/apm stay in the module graph
# (FSDP2 ignored params) but never receive LoRA or gradients.
EXCLUDE_MODULES=".*vpm.*|.*apm.*|.*talker.*|.*code2wav.*|.*code_predictor.*|.*codec.*|.*audio_decoder.*|.*audio_generator.*|.*audio_head.*|.*tts.*|.*vocoder.*"

# ── Build tiny model (pinned remote-code download + random init) ─────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/tiny-random/MiniCPM-o-4_5}"
python3 "${REPO_ROOT}/tests/special_e2e/build_minicpmo_tiny_random.py" \
    --output-dir "${MODEL_PATH}" --force

# ── Build dummy AVQA data (image + audio -> text) ────────────────────────────
# The trainer builds the val dataset at setup even with val_before_train=false,
# so the rebuild must trigger when either split is missing.
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/validation.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_minicpmo_avqa_data.py" \
        --output-dir "${DATA_DIR}"
fi

# ── Run V1 training (2 steps, LoRA, GSPO, vLLM-Omni AR rollout) ──────────────
# Mirrors the AVQA recipe; smoke-scale overrides: batch=4, rollout.n=2, short
# prompt/response, r=8, TP=NUM_GPUS, 2 GPUs.
python3 -m verl_omni.trainer.main_omni \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/validation.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.val_max_samples=2 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=MiniCPMORLHFDataset \
    +data.mm_processor_kwargs.sampling_rate=16000 \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=576 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=16 \
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
    reward.reward_manager.name=minicpm_naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=gspo-minicpmo-thinker-lora-e2e-v1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "MiniCPM-o 4.5 Thinker GSPO+LoRA e2e smoke test passed (training completed successfully)."
