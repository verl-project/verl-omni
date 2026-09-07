#!/usr/bin/env bash
# Four-node, 150-step AudioMCQ E2E: full 48-layer Qwen3-Omni thinker, 16
# Megatron train GPUs, and four standalone TP4 vLLM-Omni rollout replicas.
# Invoke this exact absolute shell once on every pod in the allocation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SUBMITTED_WRAPPER_PATH="${BASH_SOURCE[0]}"

RUN_ID_SOURCE=RUN_ID
if [[ -z "${RUN_ID:-}" ]]; then
  # Keep this list aligned with the shared scheduler IDs accepted by the base
  # launcher for Ray port selection.  In particular, Luban jobs may expose an
  # APP_ID/JOB_ID without defining LUBAN_JOB_ID.
  for RUN_ID_SOURCE in LUBAN_JOB_ID AIP_JOB_ID VC_JOB_ID JOB_ID APP_ID K8S_APP_ID; do
    if [[ -n "${!RUN_ID_SOURCE:-}" ]]; then
      RUN_ID="${!RUN_ID_SOURCE}"
      break
    fi
  done
fi
if [[ -z "${RUN_ID:-}" ]]; then
  echo "[error] No shared scheduler job ID found. Set one shared RUN_ID for all four pods; per-pod timestamps are unsafe." >&2
  exit 1
fi
RUN_ID="$(printf '%s' "${RUN_ID}" | tr -c 'A-Za-z0-9_.-' '_')"
export RUN_ID
echo "[info] AudioMCQ Gate E shared RUN_ID=${RUN_ID} source=${RUN_ID_SOURCE}"

export MODEL_PATH="${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate}"
export TRAIN_FILES="${TRAIN_FILES:-/nfs/ml-training-ssd/users/liuwei/data/audiomcq_verl_v2/train.parquet}"
export VAL_FILES="${VAL_FILES:-/nfs/ml-training-ssd/users/liuwei/data/audiomcq_verl_v2/validation.parquet}"
export STAGE_CONFIG="${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp4_full_audiomcq.yaml}"

# Persist only compact experiment evidence under the run-scoped shared root.
# HF/Torch/Triton/UV caches and Ray temporary state remain node-local.
export CACHE_ROOT="${CACHE_ROOT:-/tmp-data/liuwei/audiomcq_32gpu_cache}"
PERSIST_OUTPUT_BASE="${PERSIST_OUTPUT_BASE:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_audio_text_rl/verl-omni/outputs/audiomcq_32gpu_fullmodel_150step}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PERSIST_OUTPUT_BASE}/${RUN_ID}}"

# A long 32-GPU acceptance run without persistent logs/TensorBoard is not
# reviewable. CONFIG_ONLY may still use node-local output for a write-free
# preflight; an intentional real-run override must be explicit.
case "${OUTPUT_ROOT}" in
  /tmp | /tmp/* | /tmp-data | /tmp-data/*)
    if [[ "${CONFIG_ONLY:-0}" != "1" && "${ALLOW_NODE_LOCAL_OUTPUT:-0}" != "1" ]]; then
      echo "[error] Refusing node-local OUTPUT_ROOT for a real AudioMCQ long run: ${OUTPUT_ROOT}" >&2
      echo "[hint] Use the shared default or set ALLOW_NODE_LOCAL_OUTPUT=1 explicitly for a disposable run." >&2
      exit 1
    fi
    ;;
esac

# /tmp-data is node-local, so it is correct for high-frequency logs but cannot
# carry the one-file Ray head/worker handshake. Keep that tiny control-plane
# artifact on shared storage and poll it slowly to protect the mount.
export RAY_CONTROL_ROOT="${RAY_CONTROL_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_audio_text_rl/run_control/${RUN_ID}}"
export RAY_HEAD_PORT_FILE="${RAY_HEAD_PORT_FILE:-${RAY_CONTROL_ROOT}/ray_head_port.txt}"
export RAY_HEAD_PORT_FILE_POLL_SECONDS="${RAY_HEAD_PORT_FILE_POLL_SECONDS:-5}"

export EXP_NAME="${EXP_NAME:-qwen3_omni_audiomcq_full32_150step_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-AudioMCQ full-model 150-step E2E with validation every 10 steps}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${EXP_NAME}}"
echo "[info] Persistent OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[info] Persistent TENSORBOARD_DIR=${TENSORBOARD_DIR}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
MINIMUM_ROLLOUT_STEPS=$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))
export ROLLOUT_HEADROOM_PERCENT="${ROLLOUT_HEADROOM_PERCENT:-20}"
if [[ ! "${ROLLOUT_HEADROOM_PERCENT}" =~ ^[0-9]+$ ]]; then
  echo "[error] ROLLOUT_HEADROOM_PERCENT must be a non-negative integer, got ${ROLLOUT_HEADROOM_PERCENT}" >&2
  exit 1
fi
# The fully-async producer counts generated samples, while the trainer can drop
# stale samples.  Keep producer budget beyond the exact 150-update target; the
# trainer-side assertion now fails closed if even this budget drains too early.
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$(((MINIMUM_ROLLOUT_STEPS * (100 + ROLLOUT_HEADROOM_PERCENT) + 99) / 100))}"
export ASYNC_MAX_QUEUE_SIZE="${ASYNC_MAX_QUEUE_SIZE:-24}"
export ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-16}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-4}"
export IGNORE_EOS="${IGNORE_EOS:-False}"

export VERL_OMNI_SKIP_REWARD_LOOP=0
export VERL_OMNI_SKIP_WEIGHT_UPDATE=0
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS=1
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS=1
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-8}"
# Keep the expensive per-token rollout-correction dump off for normal long runs.
# It can still be enabled explicitly for a targeted parity/debug probe.
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-0}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-0}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-}"
export VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL="${VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL:-${OUTPUT_ROOT}/audit/${RUN_ID}.audio_binding.jsonl}"
export VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT="${VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT:-64}"

mkdir -p "${OUTPUT_ROOT}/audit"

exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh" \
  data.filter_overlong_prompts=False \
  data.train_max_samples=-1 \
  data.shuffle=True \
  data.validation_shuffle=False \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  trainer.val_before_train=True \
  trainer.test_freq="${TEST_FREQ}" \
  async_training.use_trainer_do_validate=False \
  +async_training.max_queue_size="${ASYNC_MAX_QUEUE_SIZE}" \
  +async_training.max_concurrent_samples="${ASYNC_MAX_CONCURRENT_SAMPLES}" \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_audio_model=True \
  "$@"
