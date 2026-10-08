#!/usr/bin/env bash
# MiniMax-H3 T2VA DiffusionNFT LoRA on 16 Ascend NPUs with two FSDP8 groups.
set -e

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
source "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"

export NUM_GPUS=${NUM_GPUS:-16}
export ROLLOUT_TP=${ROLLOUT_TP:-4}
export TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_TP}
export ACTOR_SP=1
export ROLLOUT_N=${ROLLOUT_N:-8}
export ACTOR_ATTN_BACKEND=_native_npu
export ROLLOUT_ATTN_BACKEND=TORCH_SDPA
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-30}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
FSDP_SIZE=${FSDP_SIZE:-8}
TIMESTEP_FRACTION=${TIMESTEP_FRACTION:-0.34}
IMAGEBIND_MODEL_PATH=${IMAGEBIND_MODEL_PATH:-.checkpoints/imagebind_huge.pth}
if (( NUM_GPUS <= 0 || FSDP_SIZE <= 0 || NUM_GPUS % FSDP_SIZE != 0 || ROLLOUT_TP <= 0 || NUM_GPUS % ROLLOUT_TP != 0 )); then
    echo "NUM_GPUS must be divisible by positive FSDP_SIZE and ROLLOUT_TP values." >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/../../../outputs/minimax_h3_t2va_lora_npu_fsdp8}
# Reuse the GPU recipe's algorithm, model layout, and LoRA settings.
# The Ascend example uses ImageBind text-video/audio-video rewards (0.8/0.2).
bash "$SCRIPT_DIR/run_minimax_h3_t2va_lora.sh" \
    trainer.device=npu \
    algorithm.timestep_fraction="$TIMESTEP_FRACTION" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$FSDP_SIZE" \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    '~reward.reward_functions.clap' \
    ++reward.reward_functions.imagebind.device=npu:1 \
    ++reward.reward_functions.imagebind.model_name_or_path="$IMAGEBIND_MODEL_PATH" \
    ++reward.reward_functions.imagebind.required=true \
    ++reward.reward_functions.imagebind.mode=all \
    '++reward.reward_functions.imagebind.weights={audio_video:0.2,text_audio:0.0,text_video:0.8}' \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=diffusion_nft_npu \
    trainer.experiment_name=minimax_h3_t2va_lora_npu_fsdp8 \
    trainer.val_before_train=False \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    "$@"
