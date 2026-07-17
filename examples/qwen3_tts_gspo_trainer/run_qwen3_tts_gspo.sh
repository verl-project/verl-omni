#!/usr/bin/env bash
# Qwen3-TTS talker GSPO training (FSDP talker actor + vLLM-Omni AR talker rollout,
# GDPO multi-dimension reward on decoded audio). Reference hardware: 1 node x 8 H200.
#
# Recipe config: config/qwen3_tts_gspo.yaml (inherits verl ppo_trainer). Only volatile values
# (paths, GPU/node counts) are set here:
#   MODEL_PATH  - Qwen3-TTS checkpoint (HF id or local dir)
#   TRAIN_FILE / VAL_FILE - parquet from data_process/tts_content_synth.py
#   SPK_EMBED   - clone-voice x-vector JSON from data_process/precompute_spk_embed.py
set -x

export NCCL_IB_DISABLE=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Load verl_omni on the driver (registers the vllm_omni rollout adapter + AudioRewardManager) and
# the Qwen3-TTS model patch (automodel registration + talker-only freeze hints). Workers also load
# the model patch via external_lib in the launch args.
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_tts

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/tts_voice_synth/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/tts_voice_synth/test.parquet"}
SPK_EMBED=${SPK_EMBED:-"$HOME/data/tts_voice_synth/spk_embed.json"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_tts_stages.yaml"

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_tts_gspo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_tts \
    actor_rollout_ref.model.override_config.tts_spk_embed_path="${SPK_EMBED}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    "$@"
