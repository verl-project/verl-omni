#!/usr/bin/env bash
# LTX-2.3 text-to-audio-video full-weight FlowGRPO recipe with VeOmni engine.
#
# This recipe mirrors run_ltx2_3_t2av_lora.sh but switches the actor/ref engine
# from FSDP2 + LoRA to the VeOmni full-weight backend:
#
#   diffusion/model_engine=veomni_diffusion          # switch the Hydra schema
#   actor_rollout_ref.actor.strategy=veomni          # actor backend = VeOmni
#   actor_rollout_ref.actor.veomni_config.*          # VeOmni-specific knobs
#   actor_rollout_ref.ref.veomni_config.strategy=veomni
#
# VeOmni does not support LoRA; this script trains full weights instead.
# Requires VeOmni installed alongside the verl-omni base environment; see
# docs/start/install.md "Optional engine backends" for the install workaround.
set -x

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-dg845/LTX-2.3-Diffusers}
DATA_DIR=${DATA_DIR:-$WORKSPACE/data/vid_prompt/verl_omni}
CLAP_MODEL_PATH=${CLAP_MODEL_PATH:-laion/larger_clap_general}
IMAGEBIND_MODEL_PATH=${IMAGEBIND_MODEL_PATH:-.checkpoints/imagebind_huge.pth}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
REWARD_DEVICE=${REWARD_DEVICE:-cuda}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

train_path=$DATA_DIR/train.parquet
test_path=$DATA_DIR/test.parquet

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

output_dir=${OUTPUT_DIR:-$repo_root/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_videos
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

TRAINER_BACKEND=veomni

python3 -m verl_omni.trainer.main_diffusion \
    diffusion/model_engine=veomni_diffusion \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.val_max_samples=1024 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.']" \
    actor_rollout_ref.actor.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.param_offload=True \
    actor_rollout_ref.actor.veomni_config.optimizer_offload=True \
    actor_rollout_ref.actor.veomni_config.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.999]" \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_frames=81 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=24 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,10]" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_contiguous=False \
    actor_rollout_ref.rollout.algo.sde_window_seed=42 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=384 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=81 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=4.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward.num_workers=$REWARD_NUM_WORKERS \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.clap.path=$repo_root/verl_omni/utils/reward_score/clap.py" \
    '+reward.reward_functions.clap.name=compute_score' \
    '+reward.reward_functions.clap.weight=1.0' \
    "+reward.reward_functions.clap.device=$REWARD_DEVICE:0" \
    "+reward.reward_functions.clap.model_name_or_path=$CLAP_MODEL_PATH" \
    "+reward.reward_functions.imagebind.path=$repo_root/verl_omni/utils/reward_score/imagebind.py" \
    '+reward.reward_functions.imagebind.name=compute_score' \
    '+reward.reward_functions.imagebind.weight=1.0' \
    "+reward.reward_functions.imagebind.device=$REWARD_DEVICE:1" \
    "+reward.reward_functions.imagebind.model_name_or_path=$IMAGEBIND_MODEL_PATH" \
    '+reward.reward_functions.imagebind.mode=audio_video' \
    reward.aggregation=weighted_sum \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=ltx2_3_t2av_veomni \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.rollout_data_dir=$rollout_data_dir \
    trainer.resume_mode=auto \
    trainer.log_val_generations=8 \
    trainer.video_fps=24 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS "$@"
