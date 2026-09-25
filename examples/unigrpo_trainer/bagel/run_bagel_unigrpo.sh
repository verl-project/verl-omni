# BAGEL UniGRPO: joint AR-"thinking" + image RL with a native rollout (no vLLM).
#
# One shared BAGEL-7B-MoT transformer generates an AR reasoning chain (understanding
# experts) then renders an image (generation `moe_gen` experts) conditioned on prompt +
# thinking. PickScore rewards the image; per-prompt-group GRPO advantages are shared by the
# AR and image tracks; a joint 2-backwards -> 1-step update trains both experts with
# per-expert learning rates. Selected via algorithm.trainer_type=unigrpo and the native
# rollout (actor_rollout_ref.rollout.name=native), which samples on the live FSDP actor
# module through a flat bf16 replica instead of a vLLM server.
#
# Prerequisite: preprocess the PickScore dataset for BAGEL (native prompt_token_ids column):
#   python examples/flowgrpo_trainer/data_process/bagel_pickscore.py \
#       --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
#       --input_dir ~/data/pickscore \
#       --output_dir ~/data/pickscore/bagel
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

pickscore_train_path=$WORKSPACE/data/pickscore/bagel/train.parquet
pickscore_test_path=$WORKSPACE/data/pickscore/bagel/test.parquet

model_name=~/models/ByteDance-Seed/BAGEL-7B-MoT
reward_function_path=pkg://verl_omni.utils.reward_score.pickscore_reward

# Single-node example; scale trainer.nnodes / n_gpus_per_node for a larger run.
NUM_GPUS=8

# num_updates_per_batch == data.train_batch_size / actor.ppo_mini_batch_size (here 8 / 4 = 2):
# update 0 is on-policy (old_logp anchored to the training module), update 1 is off-policy so
# RatioNorm / clipping engage. rollout.n is the GRPO group size G.
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$pickscore_train_path \
    data.val_files=$pickscore_test_path \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    algorithm.trainer_type=unigrpo \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.algorithm=unigrpo \
    actor_rollout_ref.model.model_type=diffusion_model \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim._target_=verl_omni.workers.config.diffusion.FSDPDiffusionOptimizerConfig \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.override_optimizer_config="{foreach: false}" \
    +actor_rollout_ref.actor.optim.param_group_lrs="{moe_gen: 3e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float32 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=unigrpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-6 \
    actor_rollout_ref.actor.diffusion_loss.mse_weight=1.5e-5 \
    actor_rollout_ref.actor.diffusion_loss.ratio_norm=True \
    actor_rollout_ref.rollout.name=native \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.pipeline.height=512 \
    actor_rollout_ref.rollout.pipeline.width=512 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=25 \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    reward.num_workers=1 \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=unigrpo \
    trainer.experiment_name=bagel_unigrpo \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=100 "$@"
