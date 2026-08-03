#!/usr/bin/env bash
# SD3.5-Medium LoRA Flow-GRPO on PickScore-SFW prompts.
# Seven GPUs run training; physical GPU 7 is reserved for the DRM server.
# See docs/start/sd35_drm_flow_grpo.md for the complete setup.
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}

train_path=$WORKSPACE/data/pickscore_sfw/sd3/train.parquet
test_path=$WORKSPACE/data/pickscore_sfw/sd3/test.parquet
model_name=stabilityai/stable-diffusion-3.5-medium
drm_server_url=http://127.0.0.1:8000/v1/score
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

NUM_GPUS_ACTOR_ROLLOUT=7
ROLLOUT_TP=1
IMAGE_RESOLUTION=512
TOTAL_TRAINING_STEPS=80
MAX_NUM_SEQS=8
REWARD_WORKERS=4

experiment_name=sd35m_drm_pickscore_sfw_8gpu_r32_n18_512_kl01_randnoise_lr5e5_clip3e6
run_name=${experiment_name}_$(date +"%Y%m%d_%H%M%S")
output_root=$WORKSPACE/outputs/sd35_drm_pickscore_sfw
checkpoint_dir=$output_root/checkpoints/$experiment_name
validation_data_dir=$output_root/runs/$run_name/validation

export WANDB_MODE=online

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=14 \
    data.val_max_samples=252 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=3e-6 \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=7 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=9 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.dtype=float32 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.n=18 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    +actor_rollout_ref.rollout.pipeline.output_type=latent \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=both \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    reward.num_workers=$REWARD_WORKERS \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.drm.path=pkg://verl_omni.utils.reward_score.latent_http_scorer_client" \
    +reward.reward_functions.drm.name=compute_score \
    +reward.reward_functions.drm.weight=1.0 \
    +reward.reward_functions.drm.required=true \
    "+reward.reward_functions.drm.server_url=$drm_server_url" \
    +reward.reward_functions.drm.noise_level=0.4 \
    +reward.reward_functions.drm.noise_seed=null \
    +reward.reward_functions.drm.score_scale=0.1 \
    +reward.reward_functions.drm.score_bias=1.0 \
    +reward.reward_functions.drm.timeout=120.0 \
    +reward.reward_functions.drm.max_retries=2 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=$run_name \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode=auto "$@"
