#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA + On-Policy Distillation (OPD) training on MMK12 with omni V1 trainer.
#
# Data preparation (run once):
#   pip install math-verify
#   python examples/gspo_trainer/data_process/mmk12.py \
#       --local_dataset_path <path_to_raw_mmk12> \
#       --local_save_dir ~/data/mmk12
#
# Runtime dependencies (all Ray worker nodes):
#   pip install math-verify    # required by mmk12_reward.py
#   pip install qwen-vl-utils  # required for multimodal data processing
#
# Validated on 2x Ascend 910C machines: student rollout/actor on 16 GPUs of node 1,
# teacher model on 16 GPUs of node 2.
#
# Start the task as follows (replace <head_ip> and <port> with the actual values from the head node):
#   1. On the master node (node 1): ray start --head
#   2. On the slave node (node 2): ray start --address='<head_ip>:<port>'
#   3. Run this script on the master node to start the training task.

set -x

export VLLM_ASCEND_ENABLE_NZ=0
# Make verl_omni available to Ray workers
export VERL_USE_EXTERNAL_MODULES=verl_omni

STUDENT_MODEL=${STUDENT_MODEL:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct-Noised"}
TEACHER_MODEL=${TEACHER_MODEL:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}

TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/mmk12/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/mmk12/test.parquet"}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-"verl_omni/utils/reward_score/mmk12_reward.py"}

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}
TEACHER_N_GPUS_PER_NODE=${TEACHER_N_GPUS_PER_NODE:-16}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="[1,2,4,16,64,128,512,1024,2048,3072,4096]" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=gspo \
    trainer.experiment_name=qwen3_omni_thinker_lora_mmk12_opd_npu \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    distillation.enabled=true \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node=${TEACHER_N_GPUS_PER_NODE} \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm_omni \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.6 \
    distillation.teacher_models.teacher_model.inference.max_model_len=16640 \
    distillation.teacher_models.teacher_model.inference.prompt_length=4160 \
    distillation.teacher_models.teacher_model.inference.response_length=12288 \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.output_mode="ar" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    distillation.distillation_loss.loss_mode=kl \
    distillation.distillation_loss.use_policy_gradient=true \
    "$@"
