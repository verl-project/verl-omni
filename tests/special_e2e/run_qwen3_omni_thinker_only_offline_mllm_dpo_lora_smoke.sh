#!/usr/bin/env bash
# Qwen3-Omni Thinker-only offline MLLM DPO + LoRA e2e smoke test.
#
# Builds a tiny random-weight Qwen3-Omni model, creates a small
# LLaVA-Hound-DPO-style image/text/video parquet dataset, then runs a couple of
# offline DPO training steps. This checks plumbing only, not model quality.
#
# Requires: verl, verl-omni, vllm-omni installed.
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_omni_thinker

# Keep this aligned with the existing Qwen3-Omni thinker smoke test.
pip install --no-cache-dir TransferQueue==0.1.8 accelerate==1.14.0
python3 -c "import transformers, accelerate; print('smoke deps: transformers', transformers.__version__, '| accelerate', accelerate.__version__)"

NUM_GPUS=${NUM_GPUS:-2}
MODEL_REPO=${MODEL_REPO:-ShowMaker27/Qwen3-Omni-tiny-random}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_llava_hound_dpo}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_CONFIG="${REPO_ROOT}/tests/special_e2e/qwen3_omni_thinker_only_mllm_smoke.yaml"
EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

# ── Resolve the tiny model: Hub checkpoint if present, else build locally ──────
if [ -z "${MODEL_PATH}" ]; then
    if python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}')" 2>/dev/null; then
        MODEL_PATH="${MODEL_REPO}"
    else
        MODEL_PATH="${HOME}/models/tiny-random/Qwen3-Omni-mllm"
        [ -d "${MODEL_PATH}" ] || python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
            --output-dir "${MODEL_PATH}"
    fi
fi

# ── Build dummy LLaVA-Hound-DPO-style multisource data if not present ──────────
if [ ! -f "${DATA_DIR}/image/train.parquet" ] || [ ! -f "${DATA_DIR}/text/train.parquet" ] || [ ! -f "${DATA_DIR}/video/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_llava_hound_dpo_data.py" \
        --local_save_dir "${DATA_DIR}"
fi

TRAIN_FILE="[${DATA_DIR}/image/train.parquet,${DATA_DIR}/text/train.parquet,${DATA_DIR}/video/train.parquet]"
VAL_FILE="[${DATA_DIR}/image/test.parquet,${DATA_DIR}/text/test.parquet,${DATA_DIR}/video/test.parquet]"

# ── Run training (tiny: 2 steps, offline DPO, Thinker-only LoRA) ───────────────
python3 -m verl_omni.trainer.main_omni \
    --config-path="${REPO_ROOT}/examples/dpo_trainer/qwen3_omni/config" \
    --config-name=qwen3_omni_thinker_offline_mllm_dpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.val_max_samples=6 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    ++data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset \
    ++data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    'actor_rollout_ref.model.target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"' \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.actor.freeze_vision_tower=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.actor.policy_loss.loss_mode=dpo \
    actor_rollout_ref.actor.policy_loss.dpo_beta=0.1 \
    \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=100000000 \
    \
    algorithm.sample_source=offline \
    algorithm.adv_estimator=dpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.name=naive \
    \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=qwen3-omni-thinker-offline-mllm-dpo-lora-e2e \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "Qwen3-Omni Thinker-only offline MLLM DPO+LoRA e2e smoke test passed."
