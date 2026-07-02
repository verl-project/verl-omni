# Bagel full-weight RL, vllm_omni rollout (FlowGRPO) with PickScore reward
#
# Non-LoRA counterpart of run_bagel_pickscore_lora.sh, following the official
# flow_grpo `pickscore_bagel` config (use_lora=False). Only the generation
# (moe_gen) pathway is trained; the understanding pathway is frozen via the
# `configure_trainable_params` hook in the BAGEL training adapter.
#
# Uses FSDP2 (strategy=fsdp2) which natively supports mixed requires_grad
# (frozen understanding + trainable generation pathway) and reshards layer
# params after forward, reducing peak memory during gradient checkpointing.
#
# Prerequisite: preprocess the PickScore dataset for BAGEL:
#   python examples/flowgrpo_trainer/data_process/bagel_pickscore.py \
#       --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
#       --input_dir ~/data/pickscore \
#       --output_dir ~/data/pickscore/bagel
#
# Raw dataset (train.txt / test.txt) from:
#   https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

pickscore_train_path=$WORKSPACE/data/pickscore/bagel/train.parquet
pickscore_test_path=$WORKSPACE/data/pickscore/bagel/test.parquet

BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-"$(dirname "$0")/bagel_deploy_config.yaml"}

model_name=~/models/ByteDance-Seed/BAGEL-7B-MoT
reward_function_path=verl_omni/utils/reward_score/pickscore_reward.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
ROLLOUT_TP=1

ENGINE=vllm_omni

# enable reward model on 0'th gpu, it is a temporary workaround
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$pickscore_train_path \
    data.val_files=$pickscore_test_path \
    data.train_batch_size=48 \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    algorithm.global_std=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=15 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.3 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,7]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    reward.num_workers=1 \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=bagel_pickscore \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
