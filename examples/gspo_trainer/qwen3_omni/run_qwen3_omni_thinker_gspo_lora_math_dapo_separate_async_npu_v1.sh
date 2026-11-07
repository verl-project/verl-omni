#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA math training on Ascend NPU with the omni
# separate-async V1 trainer.
#
# Default topology: 16 NPUs, 12 for the standalone rollout pool and 4 for the
# FSDP2 trainer. Rollout uses TP=2, so the pool runs 6 TP=2 replicas. Generation
# runs one batch ahead of training; weights are synced to the rollout replicas
# every trainer.v1.separate_async.parameter_sync_step steps.
#
# Requirements:
#   - actor_rollout_ref.rollout.nnodes > 0  (standalone rollout on dedicated NPUs)
#   - actor_rollout_ref.rollout.checkpoint_engine.backend != naive
#   - data.train_batch_size == parameter_sync_step * actor.ppo_mini_batch_size
#
# Prerequisites on every Ray node:
#   - Ascend NPU environment with torch_npu; source the CANN `set_env.sh` before
#     launching (this script does not do it, so the same script works on GPU
#     clusters that expose NPUs through Ray).
#   - ASCEND_RT_VISIBLE_DEVICES must cover TRAINER_DIES + ROLLOUT_DIES devices.
#   - pip install -e ".[train]" plus the vllm-omni build matching the installed
#     CANN/torch_npu versions (see docs/start/install.md).
#
# Policy loss is GSPO (sequence-level importance ratio, clip 3e-4 / 4e-4,
# seq-mean-token-mean) with GRPO advantages and no KL penalty. Dynamic sampling
# is disabled. `reward.reward_manager.name=dapo` selects the math_dapo reward
# (rule-based \boxed{} exact match, +1 correct / -1 wrong), not the optimizer.
#
# Data: one parquet row per problem, with data_source="math_dapo",
# reward_model.ground_truth, and a chat-style prompt column. No learned reward
# model or critic is used.

set -x

# Make verl_omni visible to Ray workers.
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}

MODEL_PATH=${MODEL_PATH:-"$WORKSPACE/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$WORKSPACE/data/dapo_math/train.parquet"}
VAL_FILE=${VAL_FILE:-"$WORKSPACE/data/dapo_math/val.parquet"}

TRAINER_DIES=${TRAINER_DIES:-4}
ROLLOUT_DIES=${ROLLOUT_DIES:-12}
ROLLOUT_TP=${ROLLOUT_TP:-2}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}
PARAM_SYNC_STEP=${PARAM_SYNC_STEP:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

if (( TRAIN_BATCH_SIZE != PARAM_SYNC_STEP * MINI_BATCH_SIZE )); then
    echo "data.train_batch_size (${TRAIN_BATCH_SIZE}) must equal" \
         "parameter_sync_step (${PARAM_SYNC_STEP}) * ppo_mini_batch_size (${MINI_BATCH_SIZE})" >&2
    exit 1
fi

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=false \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
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
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=2048 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_DIES}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,16,64,128,512,1024]' \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=2112 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.offload_policy=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.filter_groups.enable=false \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    reward.reward_model.enable=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=dapo \
    trainer.v1.trainer_mode=omni_separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step="${PARAM_SYNC_STEP}" \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qwen3_omni_gspo_npu \
    trainer.experiment_name=qwen3_omni_thinker_gspo_lora_math_dapo_separate_async_npu \
    trainer.val_before_train=false \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node="${TRAINER_DIES}" \
    trainer.nnodes=1 \
    "$@"
