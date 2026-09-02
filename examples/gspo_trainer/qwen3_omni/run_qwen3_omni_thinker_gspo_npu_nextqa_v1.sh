#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO full-parameter NextQA training with the omni V1 trainer.
# Modalities: video frames + video audio + text -> text.
# Default topology: Atlas 800T A3 (16 x Ascend 910C 64GB); override
# N_GPUS_PER_NODE and NNODES for an existing multi-node Ray cluster.
set -x
set -o pipefail

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TQ_NUM_THREADS=${TQ_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-torchcodec}

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/nextqa/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/nextqa/validation.parquet"}

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}
NNODES=${NNODES:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO:-true}
FILTER_OVERLONG_PROMPTS_WORKERS=${FILTER_OVERLONG_PROMPTS_WORKERS:-8}
TOTAL_NPUS=$((N_GPUS_PER_NODE * NNODES))

if (( TOTAL_NPUS % ROLLOUT_TP != 0 )); then
    echo "TOTAL_NPUS (${TOTAL_NPUS}) must be divisible by ROLLOUT_TP (${ROLLOUT_TP})." >&2
    exit 2
fi

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.shuffle=true \
    data.seed=42 \
    data.val_max_samples=-1 \
    data.validation_shuffle=false \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS} \
    data.truncation=error \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=QwenOmniRLHFDataset \
    ++data.mm_processor_kwargs.use_audio_in_video=${USE_AUDIO_IN_VIDEO} \
    ++data.mm_processor_kwargs.sampling_rate=16000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=2048 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.agent.num_workers=$((TOTAL_NPUS / ROLLOUT_TP)) \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=true \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qwen3_omni_nextqa \
    trainer.experiment_name=gspo_nextqa_npu \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    "$@" \
    2>&1 | tee run_qwen3omni_npu_nextqa_v1.log
