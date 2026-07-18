#!/usr/bin/env bash
# Qwen3-TTS talker online DPO (FSDP talker actor + vLLM-Omni AR rollout, LLM-audio-judge reward).
# Reference hardware: 1 node x 8 H200. A judge server must be reachable at JUDGE_URL, e.g.
#   python3 judge_server.py --provider gemini --port 8901
#
# Recipe config: config/qwen3_tts_dpo.yaml. Reuses the GSPO example's stage yaml. Volatile values:
#   MODEL_PATH  - Qwen3-TTS SFT checkpoint (HF id or local dir)
#   TRAIN_FILE / VAL_FILE - parquet from ../qwen3_tts_gspo_trainer/data_process/tts_content_synth.py
#   SPK_EMBED   - clone-voice x-vector JSON from ../qwen3_tts_gspo_trainer/data_process/precompute_spk_embed.py
#   JUDGE_URL   - the /rank judge endpoint (default http://localhost:8901)
set -x

export NCCL_IB_DISABLE=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Load verl_omni on the driver (registers the vllm_omni rollout adapter + AudioJudgeRewardManager)
# and the Qwen3-TTS model patch. Workers also load the model patch via external_lib in the args.
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_tts

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/tts_voice_synth/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/tts_voice_synth/test.parquet"}
SPK_EMBED=${SPK_EMBED:-"$HOME/data/tts_voice_synth/spk_embed.json"}
JUDGE_URL=${JUDGE_URL:-"http://localhost:8901"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/../qwen3_tts_gspo_trainer/qwen3_tts_stages.yaml"

python3 -m verl_omni.trainer.main_tts \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_tts_dpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_tts \
    actor_rollout_ref.model.override_config.tts_spk_embed_path="${SPK_EMBED}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    reward.judge_urls="[${JUDGE_URL}]" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    "$@"
