#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# LTX-2.3 text-to-audio-video DiffusionART (arXiv:2610.12572) LoRA recipe on 8 GPUs: the OmniNFT LTX-2.3
# recipe (PR #640, batch 32 x n 8, 150 steps) with only the algorithm changed, so the two runs compare
# like-for-like: same five rewards and routing weights (VideoAlign 1.0 / HPSv3 1.5 -> video, AudioBox 0.5 /
# CLAP 1.0 -> audio, DeSync 1.0 -> both), LoRA r32/alpha64, lr 3e-5, 256x384x121f, 20 rollout steps, rollout
# CFG video 1.5 / audio 3.0, validation sampler 40 steps / CFG 3 and 7 / modality 3 / rescale 0.7.
# Differences forced by the algorithm: the rollout is an ODE (noise level 0) that keeps every visited state,
# the training forward uses the SAME CFG as the rollout (v_old is recomputed by it), the old adapter is
# hard-copied every step (no EMA), and ppo_mini_batch_size = train_batch_size / 2 gives the paper's two
# disjoint updates per rollout batch. Reward stack: the native scorers, placement and the reserved
# routing_weights key are on main since #663 part 1; preserve_components (pending #664) is not needed
# because DiffusionARTLoss routes the reward/<name> component scores itself. NOT GPU-VALIDATED YET.
# Prepare parquet/model assets first; DATA_DIR, MODEL_PATH and REWARD_ROOT override defaults.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/omninft/vggsound/verl_omni}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/test.parquet}

export WANDB_MODE=${WANDB_MODE:-offline}

MODEL_REVISION=8eee8edcf067e838b843f926ec4d4cc9b2be1aaf
MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/outputs}
default_model_path=$MODEL_ROOT/models--diffusers--LTX-2.3-Diffusers/snapshots/$MODEL_REVISION
default_reward_root=$MODEL_ROOT/omninft-rewards
MODEL_PATH=${MODEL_PATH:-$default_model_path}
REWARD_ROOT=${REWARD_ROOT:-$default_reward_root}
DESYNC_SOURCE_ROOT=${DESYNC_SOURCE_ROOT:-$REWARD_ROOT/OmniNFT-reference}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
VIDEOALIGN_DEVICES=${VIDEOALIGN_DEVICES:-'[0,1]'}
HPSV3_DEVICES=${HPSV3_DEVICES:-'[2,3]'}
AUDIOBOX_DEVICES=${AUDIOBOX_DEVICES:-'[4]'}
CLAP_DEVICES=${CLAP_DEVICES:-'[5]'}
DESYNC_DEVICES=${DESYNC_DEVICES:-'[6,7]'}

ltx_pipeline_config=verl_omni.pipelines.ltx2_diffusion_art.config.LTXARTPipelineConfig
ltx_lora_targets="['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0','attn2.to_q','attn2.to_k','attn2.to_v','attn2.to_out.0','audio_attn1.to_q','audio_attn1.to_k','audio_attn1.to_v','audio_attn1.to_out.0','audio_attn2.to_q','audio_attn2.to_k','audio_attn2.to_v','audio_attn2.to_out.0','audio_to_video_attn.to_q','audio_to_video_attn.to_k','audio_to_video_attn.to_v','audio_to_video_attn.to_out.0','video_to_audio_attn.to_q','video_to_audio_attn.to_k','video_to_audio_attn.to_v','video_to_audio_attn.to_out.0','ff.net.0.proj','ff.net.2','audio_ff.net.0.proj','audio_ff.net.2']"

script_name=$(basename "$0" .sh)
output_dir=${OUTPUT_DIR:-$REPO_ROOT/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_videos
validation_data_dir=$output_dir/logs/$run_timestamp/validation_videos
# The trainer (and its wandb Tracking) runs inside the Ray TaskRunner actor, so export and forward WANDB_DIR.
export WANDB_DIR=$output_dir

mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=cuda \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.return_multi_modal_inputs=False \
    data.train_batch_size=32 \
    data.val_max_samples=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=false \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=copy \
    algorithm.old_policy_update_interval=1 \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.global_std=true \
    algorithm.art_pairwise_reward=false \
    algorithm.art_exclude_final_step=true \
    algorithm.art_log_rollout_velocity_gap=false \
    algorithm.art_modality_reward_weights=null \
    algorithm.art_sample_log_dir="$output_dir/train_samples" \
    actor_rollout_ref.model.algorithm=diffusion_art \
    actor_rollout_ref.model.model_type=diffusion_art_model \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="$ltx_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.']" \
    actor_rollout_ref.model.pipeline._target_=$ltx_pipeline_config \
    +actor_rollout_ref.model.pipeline.video_cfg_scale=1.5 \
    +actor_rollout_ref.model.pipeline.audio_cfg_scale=3.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_art \
    actor_rollout_ref.actor.diffusion_loss.art_guidance_gain=1.0 \
    actor_rollout_ref.actor.diffusion_loss.art_loss_space=velocity_linear_sigma \
    actor_rollout_ref.actor.diffusion_loss.art_mask_metric=sde_kl \
    actor_rollout_ref.actor.diffusion_loss.art_mask_threshold=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.art_reference_noise_level=0.8 \
    actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled=true \
    actor_rollout_ref.actor.diffusion_loss.art_video_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.art_audio_weight=1.0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.offload_policy=false \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LTX2VideoTransformerBlock]' \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.free_cache_engine=true \
    +actor_rollout_ref.rollout.enable_sleep_mode=true \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_use_tiling=true \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_patch_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.pipeline._target_=$ltx_pipeline_config \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=20 \
    +actor_rollout_ref.rollout.pipeline.video_cfg_scale=1.5 \
    +actor_rollout_ref.rollout.pipeline.audio_cfg_scale=3.0 \
    +actor_rollout_ref.rollout.pipeline.video_modality_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.audio_modality_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.video_rescale_scale=0.0 \
    +actor_rollout_ref.rollout.pipeline.audio_rescale_scale=0.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.pipeline._target_=$ltx_pipeline_config \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=384 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.video_cfg_scale=3.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.audio_cfg_scale=7.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.video_modality_scale=3.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.audio_modality_scale=3.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.video_rescale_scale=0.7 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.audio_rescale_scale=0.7 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.reward_model.enable=false \
    reward.reward_model.enable_resource_pool=false \
    reward.accelerator_workers.enabled=false \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.aggregation=weighted_sum \
    +reward.models.video_align.backend=native \
    +reward.models.video_align.offload=true \
    '+reward.models.video_align.executor.model=verl_omni.utils.reward_score.videoalign:VideoAlignModel' \
    "+reward.models.video_align.model_path=$REWARD_ROOT/VideoReward/checkpoint-11352/model.pth" \
    "+reward.models.video_align.placement.devices=$VIDEOALIGN_DEVICES" \
    "+reward.models.video_align.executor.kwargs.base_model_path=$REWARD_ROOT/Qwen2-VL-2B-Instruct" \
    +reward.models.hpsv3.backend=native \
    +reward.models.hpsv3.offload=true \
    '+reward.models.hpsv3.executor.model=verl_omni.utils.reward_score.hpsv3_reward:HPSv3Model' \
    +reward.models.hpsv3.executor.kwargs.use_sequential_position_ids=true \
    "+reward.models.hpsv3.model_path=$REWARD_ROOT/HPSv3/HPSv3.safetensors" \
    "+reward.models.hpsv3.placement.devices=$HPSV3_DEVICES" \
    "+reward.models.hpsv3.executor.kwargs.base_model_path=$REWARD_ROOT/Qwen2-VL-7B-Instruct" \
    +reward.models.audiobox.backend=native \
    +reward.models.audiobox.offload=true \
    '+reward.models.audiobox.executor.model=verl_omni.utils.reward_score.audiobox:AudioBoxModel' \
    "+reward.models.audiobox.model_path=$REWARD_ROOT/audiobox-aesthetics" \
    "+reward.models.audiobox.placement.devices=$AUDIOBOX_DEVICES" \
    +reward.models.clap.backend=native \
    +reward.models.clap.offload=true \
    '+reward.models.clap.executor.model=verl_omni.utils.reward_score.clap:CLAPModel' \
    "+reward.models.clap.model_path=$REWARD_ROOT/checkpoints/clap-htsat-unfused" \
    "+reward.models.clap.placement.devices=$CLAP_DEVICES" \
    +reward.models.desync.backend=native \
    +reward.models.desync.offload=true \
    '+reward.models.desync.executor.model=verl_omni.utils.reward_score.desync:DeSyncModel' \
    "+reward.models.desync.model_path=$REWARD_ROOT/synchformer/synchformer_state_dict.pth" \
    "+reward.models.desync.placement.devices=$DESYNC_DEVICES" \
    "+reward.models.desync.executor.kwargs.source_root=$DESYNC_SOURCE_ROOT" \
    +reward.reward_functions.video_align.path=pkg://verl_omni.utils.reward_score.videoalign \
    +reward.reward_functions.video_align.name=compute_score \
    +reward.reward_functions.video_align.model=video_align \
    +reward.reward_functions.video_align.prompt_key=video \
    '+reward.reward_functions.video_align.score_weights=[0.5,0.0,0.5]' \
    '+reward.reward_functions.video_align.score_means=[3.6757,0.0,2.8105]' \
    '+reward.reward_functions.video_align.score_stds=[2.2476,1.0,2.5121]' \
    +reward.reward_functions.video_align.fps=24.0 \
    +reward.reward_functions.video_align.weight=1.0 \
    +reward.reward_functions.video_align.required=true \
    '+reward.reward_functions.video_align.routing_weights={video:1.0,audio:0.0}' \
    +reward.reward_functions.hpsv3.path=pkg://verl_omni.utils.reward_score.hpsv3_reward \
    +reward.reward_functions.hpsv3.name=compute_score_hpsv3 \
    +reward.reward_functions.hpsv3.model=hpsv3 \
    +reward.reward_functions.hpsv3.prompt_key=video \
    +reward.reward_functions.hpsv3.num_frames=5 \
    +reward.reward_functions.hpsv3.top_fraction=0.3 \
    +reward.reward_functions.hpsv3.score_cap=15.0 \
    +reward.reward_functions.hpsv3.reward_scale=1.0 \
    +reward.reward_functions.hpsv3.max_batch_size=5 \
    +reward.reward_functions.hpsv3.weight=1.0 \
    +reward.reward_functions.hpsv3.required=true \
    '+reward.reward_functions.hpsv3.routing_weights={video:1.5,audio:0.0}' \
    +reward.reward_functions.audiobox.path=pkg://verl_omni.utils.reward_score.audiobox \
    +reward.reward_functions.audiobox.name=compute_score \
    +reward.reward_functions.audiobox.model=audiobox \
    '+reward.reward_functions.audiobox.axis_weights={CE:1.0,CU:1.0,PC:-1.0,PQ:1.0}' \
    +reward.reward_functions.audiobox.score_scale=0.025 \
    +reward.reward_functions.audiobox.weight=1.0 \
    +reward.reward_functions.audiobox.required=true \
    '+reward.reward_functions.audiobox.routing_weights={video:0.0,audio:0.5}' \
    +reward.reward_functions.clap.path=pkg://verl_omni.utils.reward_score.clap \
    +reward.reward_functions.clap.name=compute_score \
    +reward.reward_functions.clap.model=clap \
    +reward.reward_functions.clap.prompt_key=audio \
    +reward.reward_functions.clap.score_scale=0.5 \
    +reward.reward_functions.clap.score_offset=0.5 \
    +reward.reward_functions.clap.score_min=0.0 \
    +reward.reward_functions.clap.score_max=1.0 \
    +reward.reward_functions.clap.weight=1.0 \
    +reward.reward_functions.clap.required=true \
    '+reward.reward_functions.clap.routing_weights={video:0.0,audio:1.0}' \
    +reward.reward_functions.desync.path=pkg://verl_omni.utils.reward_score.desync \
    +reward.reward_functions.desync.name=compute_score \
    +reward.reward_functions.desync.model=desync \
    +reward.reward_functions.desync.fps=24.0 \
    +reward.reward_functions.desync.weight=1.0 \
    +reward.reward_functions.desync.required=true \
    '+reward.reward_functions.desync.routing_weights={video:1.0,audio:1.0}' \
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=$WANDB_DIR" \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name=diffusion_art \
    trainer.experiment_name=ltx2_3_art_lora_bs32_lr_3e-5 \
    trainer.default_local_dir="$checkpoint_dir" \
    trainer.rollout_data_dir="$rollout_data_dir" \
    trainer.rollout_data_save_freq=10 \
    trainer.rollout_data_max_samples=null \
    trainer.validation_data_dir="$validation_data_dir" \
    trainer.validation_data_max_samples=8 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    "$@"
