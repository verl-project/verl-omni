#!/usr/bin/env bash
# LTX-2.3 text-to-audio-video on-policy distillation (V1 trainer, sync mode).
#
# Model: dg845/LTX-2.3-Diffusers
# Algorithm: DiffusionOPD, pure distill_kl
# Rewards: CLAP + ImageBind, monitored only -- they never enter the loss
#
# A frozen LTX-2.3 teacher replays the student's joint audio-video rollout
# trajectories and the student minimizes the KL between its transition means and
# the teacher's. The reward curve therefore shows the student reaching the
# teacher's reward level through distillation alone.
#
# Set TEACHER_PATH to a full diffusers checkpoint from the same pipeline family,
# with its LoRA merged and the same scheduler config as the student. Data layout
# follows examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora_v1.sh.
#
# The teacher is colocated with the actor by default. Export TEACHER_NNODES and
# TEACHER_NPUS to move it onto its own pool when the colocated run runs out of
# memory; size the pool so TEACHER_NNODES * TEACHER_NPUS *
# ref.log_prob_micro_batch_size_per_gpu divides train_batch_size * rollout.n.
#
# Auto-detects NPU or GPU and runs with appropriate configuration.
set -x

export WANDB_MODE=${WANDB_MODE:-offline}

if npu-smi info &>/dev/null; then
    DEVICE="npu"
elif nvidia-smi &>/dev/null; then
    DEVICE="gpu"
else
    echo "Error: Neither NPU (npu-smi) nor GPU (nvidia-smi) detected." >&2
    exit 1
fi
echo "Detected device: $DEVICE"

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-dg845/LTX-2.3-Diffusers}
TEACHER_PATH=${TEACHER_PATH:?set TEACHER_PATH to the frozen LTX-2.3 teacher checkpoint}
TEACHER_NNODES=${TEACHER_NNODES:-0}
TEACHER_NPUS=${TEACHER_NPUS:-0}
DATA_DIR=${DATA_DIR:-$WORKSPACE/data/vid_prompt/verl_omni}
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
validation_data_dir=$output_dir/logs/$run_timestamp/validation_videos
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

ltx_lora_targets="['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0','attn2.to_q','attn2.to_k','attn2.to_v','attn2.to_out.0','audio_attn1.to_q','audio_attn1.to_k','audio_attn1.to_v','audio_attn1.to_out.0','audio_attn2.to_q','audio_attn2.to_k','audio_attn2.to_v','audio_attn2.to_out.0','audio_to_video_attn.to_q','audio_to_video_attn.to_k','audio_to_video_attn.to_v','audio_to_video_attn.to_out.0','video_to_audio_attn.to_q','video_to_audio_attn.to_k','video_to_audio_attn.to_v','video_to_audio_attn.to_out.0','ff.net.0.proj','ff.net.2','audio_ff.net.0.proj','audio_ff.net.2']"

if [ "$DEVICE" = "npu" ]; then
    ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
    source "$ASCEND_HOME_PATH/set_env.sh"
    source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"

    # Ascend cannot unpickle the default DataProto payloads across Ray actors.
    export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy

    NUM_GPUS=${NUM_GPUS:-16}
    ROLLOUT_TP=${ROLLOUT_TP:-4}
    REWARD_DEVICE="npu"
    PARAM_OFFLOAD=True
    OPTIMIZER_OFFLOAD=True
    RESUME_MODE=disable
else
    DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS:-8}}
    ROLLOUT_TP=${ROLLOUT_TP:-2}
    REWARD_DEVICE="cuda"
    PARAM_OFFLOAD=False
    OPTIMIZER_OFFLOAD=False
    RESUME_MODE=auto
fi

CLAP_MODEL_PATH=${CLAP_MODEL_PATH:-laion/larger_clap_general}
IMAGEBIND_MODEL_PATH=${IMAGEBIND_MODEL_PATH:-$repo_root/.checkpoints/imagebind_huge.pth}

python3 -m verl_omni.trainer.main_diffusion_v1 \
    "trainer.device=$DEVICE" \
    "data.train_files=$train_path" \
    "data.val_files=$test_path" \
    data.train_batch_size=32 \
    data.val_max_samples=1024 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=True \
    distillation.enabled=True \
    "distillation.nnodes=$TEACHER_NNODES" \
    "distillation.n_gpus_per_node=$TEACHER_NPUS" \
    "distillation.teacher_models.teacher_model.model_path=$TEACHER_PATH" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="$ltx_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.']" \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LTX2VideoTransformerBlock]' \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.999]" \
    actor_rollout_ref.actor.optim.override_optimizer_config="{eps: 1e-8}" \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    "actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    "actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP))" \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.clap.path=$repo_root/verl_omni/utils/reward_score/clap.py" \
    '+reward.reward_functions.clap.name=compute_score' \
    '+reward.reward_functions.clap.weight=1.0' \
    "+reward.reward_functions.clap.device=${REWARD_DEVICE}:0" \
    "+reward.reward_functions.clap.model_name_or_path=$CLAP_MODEL_PATH" \
    "+reward.reward_functions.imagebind.path=$repo_root/verl_omni/utils/reward_score/imagebind.py" \
    '+reward.reward_functions.imagebind.name=compute_score' \
    '+reward.reward_functions.imagebind.weight=1.0' \
    "+reward.reward_functions.imagebind.device=${REWARD_DEVICE}:1" \
    "+reward.reward_functions.imagebind.model_name_or_path=$IMAGEBIND_MODEL_PATH" \
    '+reward.reward_functions.imagebind.mode=audio_video' \
    reward.aggregation=weighted_sum \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=diffusion_opd \
    trainer.experiment_name=ltx2_3_t2av_opd \
    "trainer.default_local_dir=$checkpoint_dir" \
    "trainer.validation_data_dir=$validation_data_dir" \
    trainer.validation_data_max_samples=8 \
    trainer.log_val_generations=8 \
    trainer.video_fps=24 \
    trainer.val_before_train=True \
    "trainer.n_gpus_per_node=$NUM_GPUS" \
    trainer.nnodes=1 \
    "trainer.resume_mode=$RESUME_MODE" \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
