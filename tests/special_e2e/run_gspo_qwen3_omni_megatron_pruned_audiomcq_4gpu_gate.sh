#!/usr/bin/env bash
# Local AudioMCQ gate: two Megatron train GPUs and a TP2 vLLM-Omni Thinker
# rollout on the other two GPUs. Intermediates stay on local scratch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ASYNC_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

export RUN_ID="${RUN_ID:-audiomcq_pruned_4gpu_$(date +%Y%m%d_%H%M%S)}"
export MODEL_PATH="${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-smoke-chattemplate}"
export TRAIN_FILES="${TRAIN_FILES:-/nfs/ml-training-ssd/users/liuwei/data/audiomcq_verl_v2/train.parquet}"
export VAL_FILES="${VAL_FILES:-/nfs/ml-training-ssd/users/liuwei/data/audiomcq_verl_v2/validation.parquet}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp2_pruned_audiomcq.yaml}"
export MBRIDGE_ROOT_OVERRIDE="${MBRIDGE_ROOT_OVERRIDE:-${ASYNC_ROOT}/megatron-bridge}"
export MCORE_ROOT_OVERRIDE="${MCORE_ROOT_OVERRIDE:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/megatron-bridge/3rdparty/Megatron-LM}"
export PYTHONPATH="${SCRIPT_DIR}/runtime_compat:${PYTHONPATH:-}"

LOCAL_ROOT="${LOCAL_ROOT:-/tmp-data/liuwei/audiomcq_4gpu_gate}"
export CACHE_ROOT="${CACHE_ROOT:-${LOCAL_ROOT}/cache}"
export LOG_ROOT="${LOG_ROOT:-${LOCAL_ROOT}/logs}"
# Keep this deliberately short: Ray's Unix-domain socket paths are capped at
# 107 bytes after it appends its session and socket names.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp-data/liuwei/ar}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-${LOCAL_ROOT}/results/${RUN_ID}.rollout_corr.jsonl}"
export VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL="${VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL:-${LOCAL_ROOT}/results/${RUN_ID}.audio_binding.jsonl}"

export VERL_OMNI_SKIP_REWARD_LOOP=0
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS=1
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-1}"
export VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT="${VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT:-32}"

# Gate C scores a cold, fixed audio-conditioned sequence without syncing an
# updated actor into rollout. Gate D can override this after parity passes.
export VERL_OMNI_SKIP_WEIGHT_UPDATE="${VERL_OMNI_SKIP_WEIGHT_UPDATE:-1}"
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS="${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-1}"
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16}"

mkdir -p "${LOG_ROOT}" "${RAY_TMPDIR}" "$(dirname "${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}")"
exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_fully_async_4gpu_local_gate.sh" \
  ++ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  ++ray_kwargs.ray_init.include_dashboard=False \
  data.filter_overlong_prompts=False \
  data.train_max_samples=2 \
  data.shuffle=False \
  data.val_max_samples=1 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  rollout.n=2 \
  algorithm.rollout_correction.bypass_mode=False \
  trainer.experiment_name="audiomcq_qwen3_omni_pruned_4gpu_${RUN_ID}" \
  "$@"
