#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO full-parameter QI training with the omni V1 trainer.
# Modalities: video + its audio stream + text -> grounded reasoning + text.
# Hardware default: Atlas 800T A3 (16 x Ascend 910C 64GB).
set -euo pipefail
set -x

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
set +u
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"
set -u

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/omnivideo_r1_qi/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/omnivideo_r1_qi/validation.parquet"}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-16}
ROLLOUT_TP=${ROLLOUT_TP:-2}
REWARD_TP=${REWARD_TP:-8}
if ((REWARD_TP <= 0)); then
    echo "REWARD_TP must be greater than 0" >&2
    exit 1
fi
if ((NUM_GPUS_ACTOR_ROLLOUT_REWARD % REWARD_TP != 0)); then
    echo "NUM_GPUS_ACTOR_ROLLOUT_REWARD must be divisible by REWARD_TP" >&2
    exit 1
fi
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP))}
REWARD_GPU_MEMORY_UTILIZATION=${REWARD_GPU_MEMORY_UTILIZATION:-0.4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-8}

export OMNIVIDEO_QI_JUDGE_MODEL=${OMNIVIDEO_QI_JUDGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
if [[ -n "${OMNIVIDEO_QI_JUDGE_URL:-}" ]]; then
    REWARD_MODEL_ENABLE=false
else
    REWARD_MODEL_ENABLE=true
fi

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=24576 \
    data.max_response_length=8192 \
    data.shuffle=true \
    data.seed=42 \
    data.val_max_samples=256 \
    data.validation_shuffle=false \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=QwenOmniRLHFDataset \
    +data.use_audio_in_video=true \
    ++data.mm_processor_kwargs.fps=2.0 \
    ++data.mm_processor_kwargs.sampling_rate=16000 \
    ++data.mm_processor_kwargs.use_audio_in_video=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=2048 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.03 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=24576 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward.num_workers=${REWARD_NUM_WORKERS} \
    reward.reward_model.enable=${REWARD_MODEL_ENABLE} \
    reward.reward_model.enable_resource_pool=false \
    reward.reward_model.model_path="${OMNIVIDEO_QI_JUDGE_MODEL}" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP} \
    reward.reward_model.rollout.gpu_memory_utilization=${REWARD_GPU_MEMORY_UTILIZATION} \
    reward.reward_model.rollout.free_cache_engine=true \
    reward.reward_model.rollout.enforce_eager=false \
    reward.reward_model.rollout.do_sample=false \
    reward.reward_model.rollout.temperature=0 \
    reward.reward_model.rollout.top_p=1.0 \
    reward.reward_model.rollout.top_k=-1 \
    reward.reward_model.rollout.prompt_length=32768 \
    reward.reward_model.rollout.response_length=256 \
    reward.reward_model.rollout.max_model_len=33024 \
    reward.reward_model.rollout.max_num_seqs=32 \
    reward.reward_model.rollout.limit_images=32 \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/omnivideo_qi.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=true \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qwen3_omni_omnivideo_r1 \
    trainer.experiment_name=gspo_omnivideo_qi_npu_v1 \
    trainer.n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    "$@" \
    2>&1 | tee run_qwen3omni_npu_omnivideo_qi_v1.log
