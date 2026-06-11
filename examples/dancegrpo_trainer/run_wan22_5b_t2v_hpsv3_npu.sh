#!/bin/bash
# Wan2.2 LoRA RL with DanceGRPO - 8-NPU Global Distribution Strategy
#
# Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (text+image-to-video, used in T2V mode)
# Algorithm: DanceGRPO (reuses FlowGRPO's advantage estimator and loss)
# Reward: HPSv3 (Human Preference Score v3) - custom reward model
#
# Reference: https://github.com/XueZeyue/DanceGRPO
#
set -x
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
source $ASCEND_HOME_PATH/set_env.sh
source $ASCEND_HOME_PATH/../nnal/atb/set_env.sh
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

hpsv3_train_path=$WORKSPACE/data/hpsv3/train.parquet
hpsv3_test_path=$WORKSPACE/data/hpsv3/test.parquet

model_name=Wan-AI/Wan2.2-TI2V-5B-Diffusers
export custom_reward_model_path=$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors
custom_reward_function_path=verl_omni/utils/reward_score/hpsv3_reward.py

# 8-NPU Global Distribution
NUM_GPUS_ACTOR_ROLLOUT_REWARD=8
ROLLOUT_TP=1

ENGINE=vllm_omni

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=npu \
    algorithm.adv_estimator=dance_grpo \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    data.train_files=$hpsv3_train_path \
    data.val_files=$hpsv3_test_path \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.attn_backend='_native_npu' \
    actor_rollout_ref.model.custom_chat_template='"{% if messages %}{% for message in messages %}{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] }}{% endif %}{% endfor %}{% endif %}</s>"' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=704 \
    actor_rollout_ref.rollout.pipeline.width=1280 \
    actor_rollout_ref.rollout.pipeline.num_frames=8 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="dance_sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$custom_reward_function_path \
    reward.custom_reward_function.name=compute_score_hpsv3 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=dance_grpo_npu \
    trainer.experiment_name=wan22_5b_t2v_hpsv3_npu \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@" \
    2>&1 | tee run_wan22_5b_t2v_hpsv3_npu.log
