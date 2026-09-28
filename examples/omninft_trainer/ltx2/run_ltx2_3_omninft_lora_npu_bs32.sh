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
# Launch the batch-32 LTX-2.3 OmniNFT example in an Ascend training environment.
# Prepare parquet/model assets first; DATA_DIR, MODEL_ROOT and REWARD_ROOT override defaults.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/omninft/vggsound/verl_omni}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/test.parquet}

export WANDB_MODE=${WANDB_MODE:-online}
export OMNIFT_ROLLOUT_PROGRESS=${OMNIFT_ROLLOUT_PROGRESS:-1}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"
set -u

MODEL_REVISION=8eee8edcf067e838b843f926ec4d4cc9b2be1aaf
MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/outputs}
default_model_path=$MODEL_ROOT/models--diffusers--LTX-2.3-Diffusers/snapshots/$MODEL_REVISION
default_reward_root=$MODEL_ROOT/omninft-rewards

if [[ ! -d "$default_model_path" && -d "/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/$MODEL_REVISION" ]]; then
    default_model_path=/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/$MODEL_REVISION
fi
if [[ ! -d "$default_reward_root" && -d /hub/omninft-rewards ]]; then
    default_reward_root=/hub/omninft-rewards
elif [[ ! -d "$default_reward_root" && -d /hub/omnift-rewards ]]; then
    default_reward_root=/hub/omnift-rewards
fi
MODEL_PATH=${MODEL_PATH:-$default_model_path}
REWARD_ROOT=${REWARD_ROOT:-$default_reward_root}
DESYNC_SOURCE_ROOT=${DESYNC_SOURCE_ROOT:-$REWARD_ROOT/OmniNFT-reference}
NUM_GPUS=${NUM_GPUS:-16}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
VIDEOALIGN_DEVICES=${VIDEOALIGN_DEVICES:-'[0,1,8,9]'}
HPSV3_DEVICES=${HPSV3_DEVICES:-'[2,3,10,11]'}
AUDIOBOX_DEVICES=${AUDIOBOX_DEVICES:-'[4,12]'}
CLAP_DEVICES=${CLAP_DEVICES:-'[5,13]'}
DESYNC_DEVICES=${DESYNC_DEVICES:-'[6,7,14,15]'}

script_name=$(basename "$0" .sh)
output_dir=${OUTPUT_DIR:-$REPO_ROOT/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_videos
validation_data_dir=$output_dir/logs/$run_timestamp/validation_videos
WANDB_DIR=$output_dir

mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

python3 -m verl_omni.trainer.main_diffusion \
    --config-dir="$SCRIPT_DIR" \
    --config-name=ltx2_omninft \
    trainer.device=npu \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.return_multi_modal_inputs=False \
    data.train_batch_size=32 \
    data.val_max_samples=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_use_tiling=true \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_patch_parallel_size="$ROLLOUT_TP" \
    +reward.models.video_align.model_path="$REWARD_ROOT/VideoReward/checkpoint-11352/model.pth" \
    +reward.models.video_align.placement.devices="$VIDEOALIGN_DEVICES" \
    +reward.models.video_align.executor.kwargs.base_model_path="$REWARD_ROOT/Qwen2-VL-2B-Instruct" \
    +reward.models.hpsv3.model_path="$REWARD_ROOT/HPSv3/HPSv3.safetensors" \
    +reward.models.hpsv3.placement.devices="$HPSV3_DEVICES" \
    +reward.models.hpsv3.executor.kwargs.base_model_path="$REWARD_ROOT/Qwen2-VL-7B-Instruct" \
    +reward.models.audiobox.model_path="$REWARD_ROOT/audiobox-aesthetics" \
    +reward.models.audiobox.placement.devices="$AUDIOBOX_DEVICES" \
    +reward.models.clap.model_path="$REWARD_ROOT/checkpoints/clap-htsat-unfused" \
    +reward.models.clap.placement.devices="$CLAP_DEVICES" \
    +reward.models.desync.model_path="$REWARD_ROOT/synchformer/synchformer_state_dict.pth" \
    +reward.models.desync.placement.devices="$DESYNC_DEVICES" \
    +reward.models.desync.executor.kwargs.source_root="$DESYNC_SOURCE_ROOT" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=omni_nft \
    trainer.experiment_name=ltx2_3_omninft_lora_bs32_lr_3e-5 \
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
