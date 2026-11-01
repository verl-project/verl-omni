#!/usr/bin/env bash
# Qwen-Image DiffusionART LoRA RL on the OCR task, V1 sync trainer and vllm_omni rollout.
#
# Same data, reward, LoRA targets and rollout resolution/steps as
# examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh. DiffusionART
# (arXiv:2610.12572) rolls out a deterministic 10-step ODE with the `old` adapter, aggregates
# each prompt's rewards and endpoints into one guidance direction, and regresses the velocity
# at every visited state toward `v_old - G(c)` with a trust-region mask. The paper's two
# disjoint updates per rollout batch come from ppo_mini_batch_size = train_batch_size / 2,
# and the `old` adapter is hard-copied after every step (no EMA, no KL).
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/test.parquet

model_name=Qwen/Qwen-Image
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
ROLLOUT_TP=1
REWARD_TP=4
IMAGE_RESOLUTION=512
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}

ENGINE=vllm_omni
REWARD_ENGINE=vllm


python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_max_samples=7200 \
    data.train_batch_size=24 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=diffusion_art \
    actor_rollout_ref.model.model_type=diffusion_art_model \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_art \
    actor_rollout_ref.actor.diffusion_loss.art_guidance_gain=1.0 \
    actor_rollout_ref.actor.diffusion_loss.art_loss_space=velocity_linear_sigma \
    actor_rollout_ref.actor.diffusion_loss.art_mask_metric=sde_kl \
    actor_rollout_ref.actor.diffusion_loss.art_mask_threshold=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.art_reference_noise_level=0.8 \
    actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=copy \
    algorithm.old_policy_update_interval=1 \
    algorithm.art_pairwise_reward=false \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=diffusion_art \
    trainer.experiment_name=qwen_image_ocr_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
