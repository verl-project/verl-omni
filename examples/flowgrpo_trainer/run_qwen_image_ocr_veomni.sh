# Qwen-Image full-weight RL with the VeOmni engine, vllm_omni rollout.
#
# This recipe mirrors run_qwen_image_ocr.sh (full-weight FSDP2 baseline) and
# differs only in the engine-selection Hydra overrides:
#
#   diffusion/model_engine=veomni_diffusion          # switch the Hydra schema
#   actor_rollout_ref.actor.strategy=veomni          # actor backend = VeOmni
#   actor_rollout_ref.actor.veomni_config.*          # VeOmni-specific knobs
#   actor_rollout_ref.ref.veomni_config.strategy=veomni
#
# Requires VeOmni installed alongside the verl-omni base environment; see
# docs/start/install.md "Optional engine backends" for the install workaround
# (veomni 0.1.11's `[gpu]` extra pins torch 2.9 and conflicts with vllm 0.20.2).
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/test.parquet

model_name=Qwen/Qwen-Image
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS:-4}
NUM_NODES=${NUM_NODES:-1}
ACTOR_SP=1
ROLLOUT_TP=1
REWARD_TP=4
IMAGE_RESOLUTION=512

ENGINE=vllm_omni
REWARD_ENGINE=vllm
TRAINER_BACKEND=veomni

python3 -m verl_omni.trainer.main_diffusion \
    diffusion/model_engine=veomni_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.ulysses_parallel_size=$ACTOR_SP \
    actor_rollout_ref.actor.veomni_config.param_offload=True \
    actor_rollout_ref.actor.veomni_config.optimizer_offload=True \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_ocr_${TRAINER_BACKEND} \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / NUM_NODES)) \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
