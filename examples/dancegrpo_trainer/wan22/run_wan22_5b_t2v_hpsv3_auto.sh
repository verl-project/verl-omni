#!/bin/bash
# Wan2.2 LoRA RL with DanceGRPO
#
# Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (text+image-to-video, used in T2V mode)
# Algorithm: DanceGRPO (reuses FlowGRPO's advantage estimator and loss)
# Reward: HPSv3 (Human Preference Score v3) - custom reward model
#
# Auto-detects NPU or GPU and runs with appropriate configuration.
#
# Reference: https://github.com/XueZeyue/DanceGRPO and https://github.com/verl-project/verl-recipe/blob/main/dance_grpo/dance_grpo_mindspeed_mm/
#
set -x

detect_device() {
    if nvidia-smi &>/dev/null; then
        echo "gpu"
    else
        echo "npu"
    fi
}

DEVICE=$(detect_device)
echo "Detected device: $DEVICE"

if [ "$DEVICE" = "npu" ]; then
    ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
    source $ASCEND_HOME_PATH/set_env.sh
    source $ASCEND_HOME_PATH/../nnal/atb/set_env.sh
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:='expandable_segments:True'}
    export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:=2}

    ATTENTION_BACKEND='_native_npu'
    ROLLOUT_ATTN_BACKEND='TORCH_SDPA'
    NUM_GPUS_ACTOR_ROLLOUT_REWARD=8
    MICRO_BATCH_SIZE=4
    EXPERIMENT_NAME="wan22_5b_t2v_hpsv3_npu"
    PROJECT_NAME="dance_grpo"
else
    ATTENTION_BACKEND='_flash_3_varlen_hub'
    ROLLOUT_ATTN_BACKEND='FLASH_ATTN'
    NUM_GPUS_ACTOR_ROLLOUT_REWARD=8
    MICRO_BATCH_SIZE=4
    EXPERIMENT_NAME="wan22_5b_t2v_hpsv3_gpu"
    PROJECT_NAME="dance_grpo"
fi

WORKSPACE=${WORKSPACE:-$HOME}

hpsv3_train_path=$WORKSPACE/data/hpsv3/train.parquet
hpsv3_test_path=$WORKSPACE/data/hpsv3/test.parquet

model_name=Wan-AI/Wan2.2-TI2V-5B-Diffusers
export custom_reward_model_path=$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors
custom_reward_function_path=verl_omni/utils/reward_score/hpsv3_reward.py

ROLLOUT_TP=1

ENGINE=vllm_omni

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=$DEVICE \
    algorithm.adv_estimator=dance_grpo \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    data.train_files=$hpsv3_train_path \
    data.val_files=$hpsv3_test_path \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.seed=42 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.attn_backend=$ATTENTION_BACKEND \
    actor_rollout_ref.model.custom_chat_template='"{% if messages %}{% for message in messages %}{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] }}{% endif %}{% endfor %}{% endif %}</s>"' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=5.0 \
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
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=120 "$@" \
    2>&1 | tee run_wan22_5b_t2v_hpsv3_${DEVICE}.log