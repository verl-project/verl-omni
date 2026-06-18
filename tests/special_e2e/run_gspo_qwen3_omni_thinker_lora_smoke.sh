#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA e2e smoke test (minimal runtime).
#
# Builds a tiny random-weight Qwen3-Omni model, then runs a couple of training
# steps end-to-end to verify the full path our PR adds wires up correctly:
#   parquet load -> vLLM-Omni AR rollout (output_mode=ar) -> GSPO advantage/loss
#   -> FSDP LoRA actor back-prop -> LoRA weight sync -> validation.
#
# This is a smoke test (random weights, 2 steps): it checks the pipeline runs
# without errors, NOT model quality.
#
# Requires: verl, verl-omni, vllm-omni installed.
#   * dummy model built at MODEL_PATH (auto-built by this script if missing)
#   * a small math parquet dataset at DATA_DIR/{train,test}.parquet
#     (same default location as examples/gspo_trainer/run_qwen3_omni_thinker_gspo_lora.sh)
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -xeuo pipefail

# NCCL / accelerator env guards (mirror the example recipe; without these the
# NCCL net plugin can segfault at init on some single-node setups).
export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Load verl_omni on the driver (rollout adapter) + the Qwen3-Omni patches (processor / automodel); workers also load the model patch via external_lib below.
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_omni_thinker

NUM_GPUS=${NUM_GPUS:-2}
# Tiny model: prefer the community-hosted Hub checkpoint; build one locally if it
# is not available yet (not uploaded / offline CI). Override with MODEL_PATH.
MODEL_REPO=${MODEL_REPO:-ShowMaker27/Qwen3-Omni-tiny-random}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/math}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 2-GPU AR stage config (output_mode=ar). FSDP needs >1 GPU to shard (FULL_SHARD);
# on a single GPU it falls into NO_SHARD, which can't run the offload_to_cpu summon
# used during LoRA weight sync. TP is fixed in the stage YAML (vLLM-Omni strips
# top-level CLI engine args when a stage config is set).
STAGE_CONFIG="${REPO_ROOT}/tests/special_e2e/qwen3_omni_thinker_only_smoke.yaml"

# Same Thinker-only module filter as the example recipe.
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

# ── Resolve the tiny model: Hub checkpoint if present, else build locally ──────
if [ -z "${MODEL_PATH}" ]; then
    if python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}')" 2>/dev/null; then
        MODEL_PATH="${MODEL_REPO}"
    else
        MODEL_PATH="${HOME}/models/tiny-random/Qwen3-Omni"
        [ -d "${MODEL_PATH}" ] || python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
            --output-dir "${MODEL_PATH}"
    fi
fi

# ── Build dummy math dataset if not present ───────────────────────────────────
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_math_data.py" \
        --local_save_dir "${DATA_DIR}"
fi

# ── Run training (tiny: 2 steps, LoRA, GSPO, vLLM-Omni AR rollout) ────────────
python3 -m verl.trainer.main_ppo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.val_max_samples=4 \
    data.truncation='left' \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules="all-linear" \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=True \
    ++actor_rollout_ref.actor.freeze_vision_tower=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    reward.reward_manager.name=dapo \
    \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=gspo-qwen3-omni-thinker-lora-e2e \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "Qwen3-Omni Thinker GSPO+LoRA e2e smoke test passed (training completed successfully)."
