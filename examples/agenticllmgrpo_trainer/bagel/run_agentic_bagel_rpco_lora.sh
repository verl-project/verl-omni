#!/usr/bin/env bash
# BAGEL joint UND token GRPO + GEN FlowGRPO; run from the repository root.
if [[ "${DEBUG:-}" == "1" ]]; then
  set -x
fi
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
_PY="${_REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${_PY}" ]]; then
  _PY="$(command -v python3)"
fi
# A stale sibling must not shadow the installed verl.
_LOCAL_VERL="${_LOCAL_VERL:-${_REPO_ROOT}/../verl}"
if [[ -d "${_LOCAL_VERL}/verl" ]] \
  && PYTHONPATH="${_LOCAL_VERL}" "${_PY}" -c "from verl.utils.profiler import build_rollout_dist_profiler" >/dev/null 2>&1; then
  export PYTHONPATH="${_LOCAL_VERL}:${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  echo "[INFO] sibling verl checkout ${_LOCAL_VERL} is absent or too old for this tree; using the venv's installed verl" >&2
  export PYTHONPATH="${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

_HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

TOTAL_STEPS="${TOTAL_STEPS:-200}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-bagel_corl_${RUN_TS}}"
RUN_DIR="${RUN_DIR:-${_REPO_ROOT}/outputs/${RUN_NAME}}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-}"
echo "[INFO] bagel_corl run dir: ${RUN_DIR} (RUN_TS=${RUN_TS})"

ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-50}"
ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-1.05}"
LOGPROBS_MODE="${LOGPROBS_MODE:-raw_logprobs}"

export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}"
WORKSPACE=${WORKSPACE:-$HOME}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Preserve CUDA discovery for bare-shell launches; explicit CUDA_HOME wins.
# Flashinfer JIT needs an nvcc accepting --compress-mode=size.
_nvcc_release() {
  "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}
_nvcc_accepts_flashinfer_flags() {
  "$1" --dryrun --compress-mode=size -c /dev/null -o "${TMPDIR:-/tmp}/.nvcc-compress-probe.o" 2>&1 \
    | grep -q 'Unknown option' && return 1
  return 0
}
if [[ -z "${CUDA_HOME:-}" ]]; then
  while read -r _nvcc_ver _cand; do
    if [[ -n "${_cand}" ]] && _nvcc_accepts_flashinfer_flags "${_cand}/bin/nvcc"; then
      export CUDA_HOME="${_cand}"
      break
    fi
  done < <(
    ls -d /cm/shared/apps/cuda*/toolkit/*/ /usr/local/cuda 2>/dev/null \
      | while read -r _d; do
          [[ -x "${_d%/}/bin/nvcc" ]] && printf '%s %s\n' "$(_nvcc_release "${_d%/}/bin/nvcc")" "${_d%/}"
        done \
      | sort -rV
  )
fi
if [[ -n "${CUDA_HOME:-}" && ":${PATH}:" != *":${CUDA_HOME}/bin:"* ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# Kernel compilation also needs the venv ninja on PATH.
if [[ -x "${_REPO_ROOT}/.venv/bin/ninja" && ":${PATH}:" != *":${_REPO_ROOT}/.venv/bin:"* ]]; then
  export PATH="${_REPO_ROOT}/.venv/bin:${PATH}"
fi

# Use environment-provided CUDA compatibility libraries when present.
_cuda_compat="${_REPO_ROOT}/.venv/cuda-compat"
_cu13_lib="${_REPO_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "${_cuda_compat}" ]]; then
  export LD_LIBRARY_PATH="${_cuda_compat}:${_cu13_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
fi

model_name=${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}
reward_model_name=${REWARD_MODEL:-${_HF_HUB}/models--CodeGoat24--UnifiedReward-2.0-qwen3vl-2b/snapshots/64b2f51d68a2fa583b194784b6cce3aeb12ed7df}

UNICOT_REFLECTION_DIR=${UNICOT_REFLECTION_DIR:-${_HF_HUB}/datasets--Fr0zencr4nE--UniCoT-Self-Reflection-6K}
UNICOT_BREAKDOWN_DIR=${UNICOT_BREAKDOWN_DIR:-${_HF_HUB}/datasets--Fr0zencr4nE--UniCoT-Breakdown-3K}
# Capped training split; UNICOT_TRAIN_SIZE=full restores the full split.
UNICOT_MIX_RATIO=${UNICOT_MIX_RATIO:-0.65}
UNICOT_VAL_RATIO=${UNICOT_VAL_RATIO:-0.05}
UNICOT_SPLIT_SEED=${UNICOT_SPLIT_SEED:-42}
UNICOT_TRAIN_SIZE=${UNICOT_TRAIN_SIZE:-3000}
if [[ "$UNICOT_TRAIN_SIZE" == "full" ]]; then UNICOT_TRAIN_SIZE=; fi
UNICOT_VAL_SIZE=${UNICOT_VAL_SIZE:-}

TRAIN_FILE=${UNICOT_TRAIN:-"${_REPO_ROOT}/outputs/data/agentic_unicot/train.parquet"}
VAL_FILE=${UNICOT_TEST:-"${_REPO_ROOT}/outputs/data/agentic_unicot/val.parquet"}
if [[ "${REBUILD_UNICOT:-1}" == "1" || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  "${_PY}" -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl \
      --breakdown_dir "$UNICOT_BREAKDOWN_DIR" \
      --reflection_dir "$UNICOT_REFLECTION_DIR" \
      --local_save_dir "$(dirname "$TRAIN_FILE")" \
      --mix_ratio "$UNICOT_MIX_RATIO" \
      --val_ratio "$UNICOT_VAL_RATIO" \
      --seed "$UNICOT_SPLIT_SEED" \
      ${UNICOT_TRAIN_SIZE:+--train_size "$UNICOT_TRAIN_SIZE"} \
      ${UNICOT_VAL_SIZE:+--val_size "$UNICOT_VAL_SIZE"} || {
    echo "[ERROR] UniCoT parquet build failed (traceback above). Refusing to launch:" >&2
    echo "[ERROR] without 'set -e' the trainer would otherwise keep going and train on a" >&2
    echo "[ERROR] stale ${TRAIN_FILE} (or none at all) while silently ignoring any dataset change." >&2
    exit 1
  }
else
  echo "[INFO] reusing existing UniCoT parquet: $TRAIN_FILE / $VAL_FILE (set REBUILD_UNICOT=1 to rebuild)"
fi

_count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local n=0 id
    IFS=',' read -ra _ids <<< "${CUDA_VISIBLE_DEVICES}"
    for id in "${_ids[@]}"; do
      id="${id#"${id%%[![:space:]]*}"}"
      id="${id%"${id##*[![:space:]]}"}"
      [[ -n "$id" ]] && n=$((n + 1))
    done
    echo "$n"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
    return
  fi
  echo 1
}
N_VISIBLE="$(_count_visible_gpus)"
N_VISIBLE="${N_VISIBLE//[[:space:]]/}"
if [[ -z "$N_VISIBLE" || "$N_VISIBLE" -lt 1 ]]; then
  echo "Need at least 1 visible GPU (got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})." >&2
  exit 1
fi
NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-$N_VISIBLE}
# UND stages use logical devices 0 and 1; its slice must divide the actor pool.
UND_N_GPUS=${UND_N_GPUS:-2}
UND_GPU_MEM_UTIL=${UND_GPU_MEM_UTIL:-0.40}
BAGEL_UND_DEPLOY_CONFIG=${BAGEL_UND_DEPLOY_CONFIG:-"$(dirname "$0")/bagel_corl_deploy_ar.yaml"}
# The published UND engine needs >=0.40 KV budget unless explicitly overridden.
if [[ "${UND_AR_ALLOW_TIGHT:-0}" != "1" && "${NUM_GPUS_ACTOR_ROLLOUT_REWARD}" -ge 2 ]]; then
  if (( UND_N_GPUS < 2 )); then
    echo "UND_N_GPUS=${UND_N_GPUS} cannot cover bagel_corl_deploy_ar.yaml (Thinker on logical \"0\", DiT on \"1\")." >&2
    echo "Use UND_N_GPUS=2, or put both stages on \"0\" in the yaml for a 1-GPU smoke test." >&2
    exit 1
  fi
  if awk "BEGIN {exit !(${UND_GPU_MEM_UTIL} < 0.40)}"; then
    echo "UND_GPU_MEM_UTIL=${UND_GPU_MEM_UTIL} is below the ~0.393 AR engine-init floor" >&2
    echo "(28.16GiB consumed + 2.96GiB peak over a 27.54GiB checkpoint on a 79.66GiB card)." >&2
    echo "0.30 and 0.35 abort _init_und with 'No available memory for the cache blocks'." >&2
    exit 1
  fi
fi
BAGEL_UND_AR_SERVING_READY=${BAGEL_UND_AR_SERVING_READY:-1}
echo "bagel_corl 4-device e2e: actor+GEN=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} UND_AR_colocated=${UND_N_GPUS} visible=${N_VISIBLE} und_ready=${BAGEL_UND_AR_SERVING_READY} und_util=${UND_GPU_MEM_UTIL}"
if [[ "${NUM_GPUS_ACTOR_ROLLOUT_REWARD}" -ge 4 ]]; then
  ROLLOUT_TP=${ROLLOUT_TP:-2}
  _GEN_DEPLOY_DEFAULT="$(dirname "$0")/bagel_corl_deploy_tp2.yaml"
else
  ROLLOUT_TP=${ROLLOUT_TP:-1}
  _GEN_DEPLOY_DEFAULT="$(dirname "$0")/bagel_corl_deploy.yaml"
fi
BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-$_GEN_DEPLOY_DEFAULT}
REWARD_TP=${REWARD_TP:-$NUM_GPUS_ACTOR_ROLLOUT_REWARD}
REWARD_ENGINE=${REWARD_ENGINE:-vllm}
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % ROLLOUT_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}." >&2
  exit 1
fi
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % REWARD_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by REWARD_TP=${REWARD_TP}." >&2
  exit 1
fi
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % UND_N_GPUS != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by UND_N_GPUS=${UND_N_GPUS}" >&2
  echo "(the UND AR slice comes from split_resource_pool, which requires an exact split)." >&2
  exit 1
fi
# GEN devices are a per-replica template with exactly ROLLOUT_TP entries.
"${_PY}" - "$BAGEL_DEPLOY_CONFIG" "$ROLLOUT_TP" <<'PY' || exit 1
import sys

import yaml

path, tp = sys.argv[1], int(sys.argv[2])
try:
    stages = yaml.safe_load(open(path)).get("stages") or []
except OSError as exc:
    sys.exit(f"GEN deploy config {path!r} is unreadable: {exc}")
if not stages:
    sys.exit(f"GEN deploy config {path!r} declares no stages")
for stage in stages:
    devices = [d for d in str(stage.get("devices", "")).split(",") if d.strip()]
    if len(devices) != tp:
        sys.exit(
            f"GEN deploy config {path!r}: stage {stage.get('stage_id')} names {len(devices)} "
            f"device(s) {devices} but ROLLOUT_TP={tp}. A diffusion stage's device list is a "
            f"per-replica template whose length must equal tensor_parallel_size; pass a "
            f"matching yaml (bagel_corl_deploy.yaml for TP=1, bagel_corl_deploy_tp2.yaml for "
            f"TP=2) or clear ROLLOUT_TP."
        )
PY
# GPU-budget branches jointly select memory, batch, and sampling defaults.
GEN_N_DEFAULT=2
if [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -eq 1 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.08}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.18}
  TRAIN_BSZ=${TRAIN_BSZ:-1}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
  GEN_STEPS=${GEN_STEPS:-4}
  GEN_HW=${GEN_HW:-256}
  CFG_TEXT_SCALE=${CFG_TEXT_SCALE:-1.0}
  CFG_IMG_SCALE=${CFG_IMG_SCALE:-1.5}
  LORA_RANK=${LORA_RANK:-4}
  LORA_ALPHA=${LORA_ALPHA:-8}
elif [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -le 2 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.12}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
  TRAIN_BSZ=${TRAIN_BSZ:-2}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
  GEN_STEPS=${GEN_STEPS:-15}
  GEN_HW=${GEN_HW:-512}
  CFG_TEXT_SCALE=${CFG_TEXT_SCALE:-1.0}
  CFG_IMG_SCALE=${CFG_IMG_SCALE:-1.5}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
else
  ENABLE_RM=${ENABLE_RM:-1}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.15}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
  TRAIN_BSZ=${TRAIN_BSZ:-4}
  GEN_N_DEFAULT=8
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
  GEN_STEPS=${GEN_STEPS:-10}
  NOISE_LEVEL=${NOISE_LEVEL:-0.8}
  SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-2}
  SDE_WINDOW_RANGE=${SDE_WINDOW_RANGE:-[0,2]}
  GEN_HW=${GEN_HW:-512}
  CFG_TEXT_SCALE=${CFG_TEXT_SCALE:-4.0}
  CFG_IMG_SCALE=${CFG_IMG_SCALE:-1.5}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
fi
if [[ "$ENABLE_RM" != "1" ]]; then
  ENABLE_RM=0
fi

# v2 uses trusted continuous RM scores; without the RM lane retain v1.
if [[ "$ENABLE_RM" == "1" ]]; then
  CORL_REWARD_VERSION=${CORL_REWARD_VERSION:-v2}
else
  CORL_REWARD_VERSION=${CORL_REWARD_VERSION:-v1}
fi

# Explicit KV bytes avoid reward-engine memory profiling on colocated GPUs.
REWARD_MAX_MODEL_LEN=${REWARD_MAX_MODEL_LEN:-8192}
REWARD_KV_CACHE_BYTES=${REWARD_KV_CACHE_BYTES:-2147483648}  # pinned KV per RM GPU
JUDGE_SERVER=${JUDGE_SERVER:-0}  # external judge is opt-in; the RM pool scores by default
JUDGE_URL=${JUDGE_URL:-http://127.0.0.1:8093}
JUDGE_VLLM_URL=""
[[ "$JUDGE_SERVER" == "1" ]] && JUDGE_VLLM_URL="$JUDGE_URL"
JUDGE_GOOD_ENOUGH_THRESHOLD=${JUDGE_GOOD_ENOUGH_THRESHOLD:-0.80}
PICKSCORE_GOOD_ENOUGH_THRESHOLD=${PICKSCORE_GOOD_ENOUGH_THRESHOLD:-1.0}
if [[ -z "${JUDGE_GPU:-}" ]]; then
  JUDGE_GPU="${CUDA_VISIBLE_DEVICES##*,}"
fi
if [[ -z "${JUDGE_IMAGE_MODEL:-}" && -n "${JUDGE_ENV_FILE:-}" ]]; then
  JUDGE_IMAGE_MODEL="$(bash -c "source '${JUDGE_ENV_FILE}' >/dev/null 2>&1; printf '%s' \"\${JUDGE_IMAGE_MODEL:-}\"")"
fi
JUDGE_IMAGE_MODEL="${JUDGE_IMAGE_MODEL:-${_HF_HUB}/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"

_judge_ready() { curl -fsS -m 2 "${JUDGE_URL%/}/v1/models" >/dev/null 2>&1; }
JUDGE_SERVER_PID=""
if [[ "$JUDGE_SERVER" == "1" ]]; then
  if _judge_ready; then
    echo "[INFO] image judge already serving at ${JUDGE_URL}; reusing it"
  else
    if [[ ! -e "${JUDGE_IMAGE_MODEL}" ]]; then
      echo "[ERROR] JUDGE_IMAGE_MODEL=${JUDGE_IMAGE_MODEL} does not exist." >&2
      exit 1
    fi
    JUDGE_LOG="/tmp/bagel_judge_${RUN_TS}.log"
    echo "[INFO] starting image judge on GPU ${JUDGE_GPU} (mem-util ${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.32}); log ${JUDGE_LOG}"
    CUDA_VISIBLE_DEVICES="$JUDGE_GPU" \
      JUDGE_IMAGE_MODEL="$JUDGE_IMAGE_MODEL" \
      AGENTIC_REFLECT_GPU_MEM_UTIL="${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.32}" \
      AGENTIC_REFLECT_MAX_NUM_SEQS="${AGENTIC_REFLECT_MAX_NUM_SEQS:-2}" \
      AGENTIC_REFLECT_MAX_MODEL_LEN="${AGENTIC_REFLECT_MAX_MODEL_LEN:-4096}" \
      PYTHONPATH="${_SCRIPT_DIR}/../agent_llm${PYTHONPATH:+:${PYTHONPATH}}" \
      PATH="${_REPO_ROOT}/.venv/bin:${PATH}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}" \
      bash "${_SCRIPT_DIR}/../agent_llm/run_judge_image_tool_server.sh" >"$JUDGE_LOG" 2>&1 &
    JUDGE_SERVER_PID=$!
    for _ in $(seq 1 240); do
      _judge_ready && break
      if ! kill -0 "$JUDGE_SERVER_PID" 2>/dev/null; then
        echo "[ERROR] image judge exited during startup; last lines of ${JUDGE_LOG}:" >&2
        tail -n 40 "$JUDGE_LOG" >&2 || true
        exit 1
      fi
      sleep 2
    done
    if ! _judge_ready; then
      echo "[ERROR] image judge not ready at ${JUDGE_URL} after 480s; see ${JUDGE_LOG}" >&2
      exit 1
    fi
    echo "[INFO] image judge ready at ${JUDGE_URL} (pid ${JUDGE_SERVER_PID})"
    trap 'kill "${JUDGE_SERVER_PID}" 2>/dev/null || true' EXIT INT TERM
  fi
elif [[ "$ENABLE_RM" == "1" ]]; then
  if _judge_ready; then
    echo "[INFO] image judge detected at ${JUDGE_URL} but unused: the reward pool's router scores"
  else
    echo "[INFO] ENABLE_RM=1: image rewards and Yes/No signals come from the reward loop manager's"
    echo "[INFO]   own router (reward_router_address -> ${REWARD_MODEL:-<reward.reward_model.model_path>})"
  fi
fi

VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-32}

# Artifact-only holdout renders do not enter the validation queue.
export AGENTIC_VAL_VIZ="${AGENTIC_VAL_VIZ:-1}"

# UND uses raw centered rewards; GEN uses one batch-wide standard deviation.
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
GEN_NORM_ADV_BY_STD=${GEN_NORM_ADV_BY_STD:-True}

# UND anchors to LoRA-off base via low_var_kl; GEN has its own ratio trust band.
ACTOR_LR=${ACTOR_LR:-3e-5}
GEN_LR=${GEN_LR:-3e-5}
LR_WARMUP_RATIO=${LR_WARMUP_RATIO:-0.05}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}

# Reject UND rows with excessive rollout/trainer likelihood mismatch.
ROLLOUT_RS=${ROLLOUT_RS:-seq_mean_k1}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-20}

GEN_CLIP_RATIO=${GEN_CLIP_RATIO:-0.02}

# Keep a recoverable checkpoint when the divergence tripwire aborts.
SAVE_FREQ=${SAVE_FREQ:-25}
RESUME_MODE=${RESUME_MODE:-auto}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-2}

# N sibling episodes, S seeds/call; legacy J/K overrides retain precedence.
if [[ -z "${N:-}" ]]; then
  if [[ -n "${J:-}" ]]; then N=$J; else N=${GEN_N_DEFAULT:-2}; fi
fi
if [[ -z "${S:-}" ]]; then
  if [[ -n "${K:-}" && "$K" -ge 2 ]]; then
    S=$K
  else
    S=1
  fi
fi
if [[ "$S" -lt 1 ]]; then
  echo "gen_samples_per_call S must be >= 1 (one image per call; got S=$S)." >&2
  exit 1
fi
if [[ "$N" -lt 1 ]]; then
  echo "rollout.n N must be >= 1 (got N=$N)." >&2
  exit 1
fi
if [[ "$S" -eq 1 && "$N" -lt 2 ]]; then
  echo "S=1 groups GEN by task, so N must be >= 2 or every GEN advantage is zero (got N=$N)." >&2
  exit 1
fi
echo "bagel_corl smoke: N=$N siblings, S=$S seeds/call (in-episode J/K are runtime)"

# Divisibility is over episode count, not prompt count.
PPO_MINI=${PPO_MINI:-$TRAIN_BSZ}
PPO_MICRO=${PPO_MICRO:-1}
if (( TRAIN_BSZ % PPO_MINI != 0 )); then
  echo "data.train_batch_size=${TRAIN_BSZ} must be divisible by ppo_mini_batch_size=${PPO_MINI}." >&2
  exit 1
fi
EPISODES_PER_UPDATE=$(( PPO_MINI * N ))
if (( EPISODES_PER_UPDATE % NUM_GPUS_ACTOR_ROLLOUT_REWARD != 0 )); then
  echo "(ppo_mini_batch_size=${PPO_MINI} x N=${N})=${EPISODES_PER_UPDATE} must be divisible by" >&2
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD}." >&2
  exit 1
fi
PER_RANK_EPISODES=$(( EPISODES_PER_UPDATE / NUM_GPUS_ACTOR_ROLLOUT_REWARD ))
if (( PPO_MICRO > 0 && PER_RANK_EPISODES % PPO_MICRO != 0 )); then
  echo "(ppo_mini_batch_size x N / n_gpus_per_node)=${PER_RANK_EPISODES} must be divisible by" >&2
  echo "ppo_micro_batch_size_per_gpu=${PPO_MICRO}." >&2
  exit 1
fi

# Prompt cache and routing affinity must be enabled as a pair.
AGENT_WORKERS=${AGENT_WORKERS:-8}
ENABLE_PEC=${ENABLE_PEC:-1}
PEC_SIZE=${PEC_SIZE:-$(( S * N * AGENT_WORKERS ))}
if (( PEC_SIZE < 32 )); then PEC_SIZE=32; fi
if [[ "${ENABLE_PEC}" == "1" ]]; then
  PEC_FLAG=True
  PEC_AFFINITY=True
else
  PEC_FLAG=False
  PEC_AFFINITY=False
fi
echo "bagel_corl R2: prompt_embed_cache=${PEC_FLAG} affinity=${PEC_AFFINITY} size=${PEC_SIZE} (S*N*workers=${S}x${N}x${AGENT_WORKERS})"

# SDE retains the trajectories needed by GEN loss; ODE disables that loss.
export BAGEL_CORL_GEN_DECODE=${GEN_DECODE:-sde}

export BAGEL_CORL_DIVERGENCE_ABORT="${BAGEL_CORL_DIVERGENCE_ABORT:-1}"
# The + prefixes append fields absent from the composed upstream schema.
# Caller Hydra overrides remain last.
"${_PY}" -m verl_omni.trainer.main_omni \
    "hydra.run.dir=$RUN_DIR" \
    "agentic_image_gen.run_dir=$RUN_DIR" \
    agentic_image_gen.vllm_url=$JUDGE_VLLM_URL \
    agentic_image_gen.good_enough_threshold=$JUDGE_GOOD_ENOUGH_THRESHOLD \
    agentic_image_gen.score_backend=${SCORE_BACKEND:-unified_reward_rubric} \
    agentic_image_gen.corl_reward_version=$CORL_REWARD_VERSION \
    agentic_image_gen.unified_good_enough_threshold=${UNIFIED_GOOD_ENOUGH_THRESHOLD:-0.62} \
    agentic_image_gen.pickscore_good_enough_threshold=$PICKSCORE_GOOD_ENOUGH_THRESHOLD \
    trainer.v1.trainer_mode=bagel_corl_sync \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD \
    algorithm.gen_norm_adv_by_std_in_grpo=$GEN_NORM_ADV_BY_STD \
    algorithm.rollout_correction.rollout_rs=$ROLLOUT_RS \
    algorithm.rollout_correction.rollout_rs_threshold=$ROLLOUT_RS_THRESHOLD \
    trainer.v1.sampler.sync_refill_failed_groups=${SYNC_REFILL_FAILED_GROUPS:-True} \
    "data.train_files=$TRAIN_FILE" \
    "data.val_files=$VAL_FILE" \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_PROMPT_LEN \
    data.trust_remote_code=True \
    "actor_rollout_ref.model.path=$model_name" \
    "actor_rollout_ref.model.tokenizer_path=$model_name" \
    +actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.model_type=diffusion_model \
    +actor_rollout_ref.model.composite_mode=bagel_corl \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora_dtype=bfloat16 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj','q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_RATIO \
    actor_rollout_ref.actor.optim.lr_scheduler_type=$LR_SCHEDULER \
    actor_rollout_ref.actor.optim.min_lr_ratio=$MIN_LR_RATIO \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    +actor_rollout_ref.actor.diffusion_loss.clip_ratio=$GEN_CLIP_RATIO \
    +actor_rollout_ref.model.lr_gen=$GEN_LR \
    +actor_rollout_ref.actor.diffusion_loss.loss_weight_und=1.0 \
    +actor_rollout_ref.actor.diffusion_loss.loss_weight_gen=1.0 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.response_length=$MAX_PROMPT_LEN \
    +actor_rollout_ref.rollout.pipeline.height=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.width=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.num_inference_steps=$GEN_STEPS \
    +actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LEN \
    +actor_rollout_ref.rollout.pipeline.cfg_text_scale=$CFG_TEXT_SCALE \
    +actor_rollout_ref.rollout.pipeline.cfg_img_scale=$CFG_IMG_SCALE \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.logprobs_mode=$LOGPROBS_MODE \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    +actor_rollout_ref.rollout.repetition_penalty=$ROLLOUT_REPETITION_PENALTY \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.enable_prompt_embed_cache=$PEC_FLAG \
    +actor_rollout_ref.rollout.enable_prompt_embed_cache_routing_affinity=$PEC_AFFINITY \
    +actor_rollout_ref.rollout.prompt_embed_cache_size=$PEC_SIZE \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
    +actor_rollout_ref.rollout.algo.noise_level=${NOISE_LEVEL:-0.7} \
    +actor_rollout_ref.rollout.algo.sde_window_size=${SDE_WINDOW_SIZE:-2} \
    +actor_rollout_ref.rollout.algo.sde_window_range=${SDE_WINDOW_RANGE:-[0,7]} \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=${VAL_GEN_STEPS:-50} \
    +actor_rollout_ref.rollout.val_kwargs.algo.noise_level=${VAL_NOISE_LEVEL:-0.0} \
    actor_rollout_ref.rollout.agent.default_agent_loop=bagel_multiturn_agent \
    +actor_rollout_ref.rollout.agent.gen_samples_per_call=$S \
    +actor_rollout_ref.rollout.agent.max_generate_passes=${MAX_GENERATE_PASSES:-3} \
    +actor_rollout_ref.rollout.agent.max_und_turns=${MAX_UND_TURNS:-8} \
    +actor_rollout_ref.rollout.agent.und_ar_serving_ready=$BAGEL_UND_AR_SERVING_READY \
    "+actor_rollout_ref.rollout.agent.und_deploy_config=$BAGEL_UND_DEPLOY_CONFIG" \
    +actor_rollout_ref.rollout.agent.und_n_gpus=$UND_N_GPUS \
    +actor_rollout_ref.rollout.agent.und_gpu_memory_utilization=$UND_GPU_MEM_UTIL \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG" \
    reward.num_workers=$(( ENABLE_RM == 1 ? NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP : 0 )) \
    reward.reward_model.enable=$ENABLE_RM \
    "reward.reward_model.model_path=$reward_model_name" \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.bagel_rm_image_scorer \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.gpu_memory_utilization=$REWARD_GPU_MEM_UTIL \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.skip_tokenizer_init=False \
    reward.reward_model.rollout.max_model_len=$REWARD_MAX_MODEL_LEN \
    +reward.reward_model.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=$REWARD_KV_CACHE_BYTES \
    trainer.val_before_train=True \
    trainer.test_freq=${TEST_FREQ:-30} \
    data.val_batch_size=${VAL_BSZ:-$TRAIN_BSZ} \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.resume_mode=$RESUME_MODE \
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP \
    trainer.logger="['console','wandb']" \
    trainer.project_name=${PROJECT_NAME:-bagel_corl} \
    trainer.experiment_name=${EXPERIMENT_NAME:-bagel_corl_pr1} \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    ${DEFAULT_LOCAL_DIR:+trainer.default_local_dir="$DEFAULT_LOCAL_DIR"} \
    "$@"
