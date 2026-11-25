#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
#
# MiniCPM-o 4.5 streaming duplex GRPO (LoRA, colocated) — RFC verl-omni #905.
# One 4xH800 node; rollout TP=2; replayed two-track event episodes
# (pause / turn / interruption) with CPU rule rewards.
#
# Requires a vLLM-Omni build with the duplex action-capture surface
# (vllm-project/vllm-omni#10627). verl-omni's pinned revision stays turn-based
# (issue #919); override the pin for this experiment:
#   uv pip install --no-deps "vllm-omni @ git+https://github.com/vllm-project/vllm-omni@<capture-revision>"

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local MiniCPM-o 4.5 directory}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the mined duplex episode parquet}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to the fixed validation episode parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minicpmo_4_5_duplex_grpo}"
NUM_GPUS="${NUM_GPUS:-4}"
SEED="${SEED:-2026}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-20}"
RESUME_MODE="${RESUME_MODE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for path in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [[ -e "${path}" ]] || { printf 'Missing required path: %s\n' "${path}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONHASHSEED="${SEED}"
export TOKENIZERS_PARALLELISM=false
export VERL_USE_EXTERNAL_MODULES=verl_omni

# Non-obvious settings (kept out of the arg block: a comment line inside a
# backslash-continued command would split it):
# - data.max_prompt_length=128: the replay context is ~70 tokens (system prompt
#   + bundled ref audio); 64 silently left-truncated it.
# - data.max_response_length=1024: bounded episodes with tails and injected
#   interruptions overflow 384.
# - actor_rollout_ref.model.exclude_modules: LoRA the thinker only; the
#   audio/vision towers (apm/vpm/resampler) stay frozen.
# - actor_rollout_ref.actor.strategy=fsdp2: fully-sharded LoRA on 4 GPUs.
# - actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm with
#   loss_scale_factor=1024: Dr.GRPO's constant token denominator — the rollout
#   horizon (data.max_response_length), not the realized token count.
# - algorithm.norm_adv_by_std_in_grpo=false: Dr.GRPO group advantage R - mean(R);
#   the 1 Hz-quantized rewards make 15-tie + 1-outlier groups whose
#   std-normalized advantages are amplified by sqrt(G).
# - actor_rollout_ref.rollout.max_model_len=8192: a duplex session's KV window
#   is 8097 tokens (96-id prefix + 8000-id high watermark + the sample slot);
#   the earlier 4096 batched-token budget could not carry it, and max_model_len
#   binds the window geometry, not the model's 40960 context.
# - actor_rollout_ref.rollout.enable_prefix_caching=false: duplex sessions own
#   their KV lifecycle; block-level prefix reuse across sessions is not sound
#   under paced appends.
# - actor_rollout_ref.rollout.calculate_log_probs=false: old log-probs come
#   from the trainer's own actor forward; never build a captured-logprob
#   channel (see docs/algo/duplex_grpo.md "Log-probability semantics").
# - +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=1800:
#   a cold duplex stage needs far more than the engine's 300 s init default
#   (the orchestrator forwards this to OmniConfig.stage_init_timeout and
#   raises init_timeout to match).
# - reward.reward_manager.name=minicpm_naive: the default visual manager
#   expects image-reward payloads; minicpm_naive feeds the rule scorer
#   directly (and demotes MiniCPM-o's special answer tags so choice scores
#   decode).

"${PYTHON_BIN}" -m verl_omni.trainer.main_omni \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    "data.seed=${SEED}" \
    "data.custom_cls.path=verl_omni/utils/dataset/duplex/duplex_rl_dataset.py" \
    "data.custom_cls.name=DuplexEpisodeDataset" \
    data.train_batch_size=16 \
    data.dataloader_num_workers=2 \
    data.max_prompt_length=128 \
    data.max_response_length=1024 \
    data.val_max_samples=32 \
    data.validation_shuffle=false \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.exclude_modules='(apm|vpm|resampler)\..*' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor=1024 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.optim.lr=5.0e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    "actor_rollout_ref.actor.data_loader_seed=${SEED}" \
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=${NUM_GPUS}" \
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float32 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}' \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=0 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=100 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.agent.default_agent_loop=omni_duplex_agent \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.seed=${SEED}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=duplex \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5_duplex \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=duplex \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.async_chunk=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=1800 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.strategy=fsdp \
    "actor_rollout_ref.ref.fsdp_config.fsdp_size=${NUM_GPUS}" \
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}" \
    actor_rollout_ref.ref.fsdp_config.model_dtype=float32 \
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
    '+actor_rollout_ref.ref.fsdp_config.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}' \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=true \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.use_kl_in_reward=false \
    reward.num_workers=1 \
    reward.reward_manager.name=minicpm_naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/duplex_turn_taking.py \
    reward.custom_reward_function.name=compute_score_duplex_turn_taking \
    trainer.val_before_train=true \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=minicpmo_4_5_duplex_grpo \
    trainer.experiment_name=minicpmo_4_5_duplex_grpo_lora_v1 \
    "trainer.n_gpus_per_node=${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs=100 \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
    "trainer.test_freq=${TEST_FREQ}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.resume_mode=${RESUME_MODE}" \
    trainer.max_actor_ckpt_to_keep=10 \
    "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
    "trainer.validation_data_dir=${OUTPUT_DIR}/validation" \
    "trainer.rollout_data_dir=${OUTPUT_DIR}/rollout" \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
