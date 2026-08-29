#!/usr/bin/env bash
set -xeuo pipefail

# 4-node/32-GPU full Qwen3-Omni bringup for Megatron trainer plus standalone
# vLLM-Omni rollout through verl fully_async_policy resource split.
# Resource default: each node splits 4 train GPUs + 4 rollout GPUs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ASYNC_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
VERL_ROOT=${VERL_ROOT:-${ASYNC_ROOT}/verl}
VLLM_OMNI_ROOT=${VLLM_OMNI_ROOT:-${ASYNC_ROOT}/vllm-omni}
MEGATRON_BRIDGE_REPO=${MEGATRON_BRIDGE_REPO:-${ASYNC_ROOT}/megatron-bridge}
RUNTIME_COMPAT_DIR="${REPO_ROOT}/tests/special_e2e/runtime_compat"
cd "${VERL_ROOT}"

CONDA_ENV=${CONDA_ENV:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async}
if [[ ! -f "${CONDA_ENV}/bin/activate" ]]; then
  echo "[error] CONDA_ENV not found: ${CONDA_ENV}" >&2
  exit 1
fi
source "${CONDA_ENV}/bin/activate"

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-verl_omni,verl_omni.models.transformers.qwen3_omni_thinker}
export VERL_OMNI_SKIP_MODELS=${VERL_OMNI_SKIP_MODELS:-1}
export VERL_OMNI_SKIP_PIPELINES=${VERL_OMNI_SKIP_PIPELINES:-1}
export VERL_OMNI_SKIP_REWARD_LOOP=${VERL_OMNI_SKIP_REWARD_LOOP:-1}
export VERL_OMNI_SKIP_TRAINER=${VERL_OMNI_SKIP_TRAINER:-0}
export VERL_OMNI_SKIP_ENGINES=${VERL_OMNI_SKIP_ENGINES:-0}
export VERL_FORCE_SHM_WEIGHT_TRANSFER=${VERL_FORCE_SHM_WEIGHT_TRANSFER:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VERL_PPO_LOGGING_LEVEL=${VERL_PPO_LOGGING_LEVEL:-INFO}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export CPATH=/usr/include${CPATH:+:$CPATH}
# sitecustomize in this local-only directory supplies the missing package
# version metadata for the optional nvidia-resiliency-ext checkpoint API.  It
# must be first so both this driver and Ray workers import it at startup.
export PYTHONPATH="${RUNTIME_COMPAT_DIR}:${REPO_ROOT}:${VERL_ROOT}:${MEGATRON_BRIDGE_REPO}/src:${VLLM_OMNI_ROOT}:${PYTHONPATH:-}"

export CUDA_HOME=${VERL_CUDA_HOME:-/usr/local/cuda-12.6}
_clean_ld_parts=()
IFS=: read -ra _ld_parts <<<"${LD_LIBRARY_PATH:-}"
for _ld_part in "${_ld_parts[@]}"; do
  if [[ "${_ld_part}" == *"/nvidia/cu13/lib" ]]; then
    continue
  fi
  _clean_ld_parts+=("${_ld_part}")
done
LD_LIBRARY_PATH="$(IFS=:; echo "${_clean_ld_parts[*]}")"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
unset _clean_ld_parts _ld_parts _ld_part
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}

TE_ATTENTION_DEBUG=${TE_ATTENTION_DEBUG:-0}
if [[ "${TE_ATTENTION_DEBUG}" == "1" ]]; then
  export NVTE_DEBUG=${NVTE_DEBUG:-1}
  export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-2}
  export NVTE_PRINT_RANK=${NVTE_PRINT_RANK:-1}
fi

CACHE_ROOT=${CACHE_ROOT:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async_full_32gpu_cache}
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export TORCH_HOME=${TORCH_HOME:-${CACHE_ROOT}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${CACHE_ROOT}/uv}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

MODEL_PATH=${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate}
TRAIN_FILES=${TRAIN_FILES:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/train.parquet}
VAL_FILES=${VAL_FILES:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/test.parquet}
STAGE_CONFIG=${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp4_full_async_no_sleep_raw_logprobs.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[error] MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILES}" ]]; then
  echo "[error] TRAIN_FILES not found: ${TRAIN_FILES}" >&2
  exit 1
fi
if [[ ! -f "${VAL_FILES}" ]]; then
  echo "[error] VAL_FILES not found: ${VAL_FILES}" >&2
  exit 1
fi
if [[ ! -f "${STAGE_CONFIG}" ]]; then
  echo "[error] STAGE_CONFIG not found: ${STAGE_CONFIG}" >&2
  exit 1
fi

PROJECT_NAME=${PROJECT_NAME:-qwen3_omni_megatron_async_fullmodel}
RUN_ID_RAW="${RUN_ID:-${LUBAN_JOB_ID:-${AIP_JOB_ID:-${VC_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}}}"
RUN_ID="$(printf '%s' "${RUN_ID_RAW}" | tr -c 'A-Za-z0-9_.-' '_')"
EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fully_async_${RUN_ID}}
PROFILE_LABEL=${PROFILE_LABEL:-32-GPU fully-async resource split 1-step}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${EXP_NAME}}
CKPT_DIR=${CKPT_DIR:-${OUTPUT_ROOT}/ckpts/${EXP_NAME}}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}
RUN_CONFIG_DIR=${RUN_CONFIG_DIR:-${OUTPUT_ROOT}/config}
LOG_NODE_RANK_HINT=${DISTRIBUTED_NODE_RANK:-${NODE_RANK:-${RAY_NODE_RANK:-}}}
LOG_TASK_ROLE_HINT=${DISTRIBUTED_TASK_ROLE:-${VC_TASK_ROLE:-${ROLE_NAME:-${TASK_ROLE:-}}}}
LOG_TASK_INDEX_HINT=${VC_TASK_INDEX:-${VK_TASK_INDEX:-${TASK_INDEX:-}}}
if [[ -z "${LOG_NODE_RANK_HINT}" && -n "${LOG_TASK_ROLE_HINT}" && "${LOG_TASK_INDEX_HINT}" =~ ^[0-9]+$ ]]; then
  if [[ "${LOG_TASK_ROLE_HINT}" == "master" || "${LOG_TASK_ROLE_HINT}" == "head" || "${LOG_TASK_ROLE_HINT}" == "chief" ]]; then
    LOG_NODE_RANK_HINT="${LOG_TASK_INDEX_HINT}"
  else
    LOG_NODE_RANK_HINT=$((LOG_TASK_INDEX_HINT + 1))
  fi
fi
if [[ -z "${LOG_NODE_RANK_HINT}" ]]; then
  LOG_HOST_TAG="$(hostname -s 2>/dev/null || hostname)"
  LOG_HOST_TAG="$(printf '%s' "${LOG_HOST_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"
  LOG_FILE=${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.host${LOG_HOST_TAG}.pid$$.log}
elif [[ "${LOG_NODE_RANK_HINT}" == "0" ]]; then
  LOG_FILE=${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}
else
  LOG_FILE=${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.node${LOG_NODE_RANK_HINT}.log}
fi
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
if [[ "${VERL_OMNI_LOG_TEE_INITIALIZED:-0}" != "1" ]]; then
  export VERL_OMNI_LOG_TEE_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

fail() {
  echo "[error] $*" >&2
  exit 1
}

persist_run_config_snapshot() {
  if [[ "${NODE_RANK}" != "0" ]]; then
    return
  fi

  local snapshot_tmp checksum_tmp source_path target_name name arg arg_index
  local -a snapshot_files
  mkdir -p "${RUN_CONFIG_DIR}"

  for source_path in "${SUBMITTED_WRAPPER_PATH:-}" "${BASH_SOURCE[0]}" "${STAGE_CONFIG}"; do
    if [[ -z "${source_path}" || ! -f "${source_path}" ]]; then
      continue
    fi
    if [[ "${source_path}" == "${SUBMITTED_WRAPPER_PATH:-}" ]]; then
      target_name=submitted_wrapper.sh
    elif [[ "${source_path}" == "${BASH_SOURCE[0]}" ]]; then
      target_name=base_launcher.sh
    else
      target_name=stage_config.yaml
    fi
    snapshot_tmp="${RUN_CONFIG_DIR}/.${target_name}.tmp.$$"
    cp -- "${source_path}" "${snapshot_tmp}"
    mv -f -- "${snapshot_tmp}" "${RUN_CONFIG_DIR}/${target_name}"
  done

  snapshot_tmp="${RUN_CONFIG_DIR}/.resolved_config.env.tmp.$$"
  {
    for name in RUN_ID EXP_NAME PROFILE_LABEL MODEL_PATH TRAIN_FILES VAL_FILES STAGE_CONFIG \
      OUTPUT_ROOT LOG_DIR LOG_FILE LOG_NODE_RANK_HINT TENSORBOARD_DIR CKPT_DIR CACHE_ROOT RAY_TMPDIR \
      RAY_CONTROL_ROOT RAY_HEAD_PORT_FILE NNODES GPUS_PER_NODE TRAIN_NNODES TRAIN_GPUS_PER_NODE \
      ROLLOUT_NNODES ROLLOUT_GPUS_PER_NODE TOTAL_EPOCHS TOTAL_TRAINING_STEPS \
      TOTAL_ROLLOUT_STEPS TEST_FREQ VAL_MAX_SAMPLES MAX_PROMPT_LENGTH \
      MAX_RESPONSE_LENGTH MAX_MODEL_LEN PPO_MINI_BATCH_SIZE N_RESP_PER_PROMPT \
      ROLLOUT_MAX_NUM_SEQS ROLLOUT_MAX_NUM_BATCHED_TOKENS \
      ROLLOUT_GPU_MEMORY_UTILIZATION ASYNC_MAX_QUEUE_SIZE \
      ASYNC_MAX_CONCURRENT_SAMPLES VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL \
      VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL; do
      printf '%s=%q\n' "${name}" "${!name:-}"
    done
    arg_index=0
    for arg in "$@"; do
      printf 'LAUNCH_ARG_%03d=%q\n' "${arg_index}" "${arg}"
      arg_index=$((arg_index + 1))
    done
  } > "${snapshot_tmp}"
  mv -f -- "${snapshot_tmp}" "${RUN_CONFIG_DIR}/resolved_config.env"

  snapshot_files=(base_launcher.sh stage_config.yaml resolved_config.env)
  if [[ -f "${RUN_CONFIG_DIR}/submitted_wrapper.sh" ]]; then
    snapshot_files=(submitted_wrapper.sh "${snapshot_files[@]}")
  fi
  checksum_tmp="${RUN_CONFIG_DIR}/.SHA256SUMS.tmp.$$"
  (
    cd "${RUN_CONFIG_DIR}"
    sha256sum "${snapshot_files[@]}" > "${checksum_tmp}"
  )
  mv -f -- "${checksum_tmp}" "${RUN_CONFIG_DIR}/SHA256SUMS"
  echo "[info] Persisted run configuration snapshot: ${RUN_CONFIG_DIR}"
}

is_true() {
  local value
  value="${1,,}"
  [[ "${value}" == "true" || "${value}" == "1" || "${value}" == "yes" ]]
}

require_nonnegative_int() {
  local name value
  name=$1
  value="${!name:-}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    fail "${name} must be a non-negative integer, got '${value}'"
  fi
}

require_positive_int() {
  local name value
  name=$1
  require_nonnegative_int "${name}"
  value="${!name}"
  if (( value <= 0 )); then
    fail "${name} must be a positive integer, got ${value}"
  fi
}

require_port() {
  local name value
  name=$1
  require_positive_int "${name}"
  value="${!name}"
  if (( value > 65535 )); then
    fail "${name} must be <= 65535, got ${value}"
  fi
}

require_unit_interval_float() {
  local name value
  name=$1
  value="${!name:-}"
  if [[ ! "${value}" =~ ^(0(\.[0-9]+)?|1(\.0+)?|\.[0-9]+)$ ]]; then
    fail "${name} must be in (0, 1], got '${value}'"
  fi
  if [[ "${value}" =~ ^0(\.0+)?$ ]]; then
    fail "${name} must be greater than 0, got ${value}"
  fi
}

ensure_divisible() {
  local dividend divisor message
  dividend=$1
  divisor=$2
  message=$3
  if (( divisor == 0 || dividend % divisor != 0 )); then
    fail "${message}: ${dividend} is not divisible by ${divisor}"
  fi
}

ensure_port_outside_worker_range() {
  local name value
  name=$1
  value="${!name}"
  if (( value >= RAY_MIN_WORKER_PORT && value <= RAY_MAX_WORKER_PORT )); then
    fail "${name}=${value} overlaps Ray worker port range ${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT}"
  fi
}

stable_hash_mod() {
  local seed mod checksum
  seed=$1
  mod=$2
  checksum=$(printf '%s' "${seed}" | cksum | awk '{print $1}')
  echo $((checksum % mod))
}

port_block_is_free() {
  local start span
  start=$1
  span=$2
  python3 - "${start}" "${span}" <<'PY'
import socket
import sys

start = int(sys.argv[1])
span = int(sys.argv[2])
sockets = []
try:
    for port in range(start, start + span):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))
        sockets.append(sock)
except OSError:
    sys.exit(1)
finally:
    for sock in sockets:
        sock.close()
sys.exit(0)
PY
}

port_block_free_count() {
  local start span
  start=$1
  span=$2
  python3 - "${start}" "${span}" <<'PY'
import socket
import sys

start = int(sys.argv[1])
span = int(sys.argv[2])
free = 0
for port in range(start, start + span):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))
    except OSError:
        pass
    else:
        free += 1
    finally:
        sock.close()
print(free)
PY
}

claim_ray_worker_port_block() {
  local pool_start pool_end span preferred_index seed min_free blocks lock_root safe_seed i index start end free_count lock_file fd
  pool_start=$1
  pool_end=$2
  span=$3
  preferred_index=$4
  seed=$5
  min_free=$6
  blocks=$(( (pool_end - pool_start + 1) / span ))
  lock_root="${RAY_WORKER_PORT_LOCK_DIR:-/tmp/verl_omni_ray_worker_ports_${USER:-unknown}}"
  safe_seed="$(printf '%s' "${seed}" | tr -c 'A-Za-z0-9_.-' '_')"

  command -v flock >/dev/null 2>&1 || fail "flock is required for Ray worker port block locking"
  mkdir -p "${lock_root}"

  for ((i = 0; i < blocks; i++)); do
    index=$(((preferred_index + i) % blocks))
    start=$((pool_start + index * span))
    end=$((start + span - 1))
    if (( end > pool_end )); then
      continue
    fi
    free_count=$(port_block_free_count "${start}" "${span}")
    if (( free_count < min_free )); then
      continue
    fi

    lock_file="${lock_root}/${safe_seed}.block${index}.lock"
    exec {fd}>"${lock_file}"
    if flock -n "${fd}"; then
      printf 'pid=%s host=%s range=%s-%s free_count=%s min_free=%s\n' "$$" "$(hostname -f 2>/dev/null || hostname)" "${start}" "${end}" "${free_count}" "${min_free}" >&"${fd}" || true
      RAY_WORKER_PORT_LOCK_FD=${fd}
      RAY_WORKER_PORT_BLOCK_INDEX=${index}
      RAY_WORKER_PORT_BASE=${start}
      RAY_MIN_WORKER_PORT=${start}
      RAY_MAX_WORKER_PORT=${end}
      return 0
    fi
    eval "exec ${fd}>&-"
  done

  fail "No Ray worker port block in ${pool_start}-${pool_end} with at least ${min_free}/${span} free ports"
}

claim_ray_component_port_block() {
  local pool_start pool_end span preferred_index seed blocks lock_root safe_seed i index start end lock_file fd
  pool_start=$1
  pool_end=$2
  span=$3
  preferred_index=$4
  seed=$5
  blocks=$(( (pool_end - pool_start + 1) / span ))
  lock_root="${RAY_COMPONENT_PORT_LOCK_DIR:-/tmp/verl_omni_ray_component_ports_${USER:-unknown}}"
  safe_seed="$(printf '%s' "${seed}" | tr -c 'A-Za-z0-9_.-' '_')"

  command -v flock >/dev/null 2>&1 || fail "flock is required for Ray component port block locking"
  mkdir -p "${lock_root}"

  for ((i = 0; i < blocks; i++)); do
    index=$(((preferred_index + i) % blocks))
    start=$((pool_start + index * span))
    end=$((start + span - 1))
    if (( end > pool_end )); then
      continue
    fi
    if ! port_block_is_free "${start}" "${span}"; then
      continue
    fi

    lock_file="${lock_root}/${safe_seed}.block${index}.lock"
    exec {fd}>"${lock_file}"
    if flock -n "${fd}"; then
      printf 'pid=%s host=%s range=%s-%s\n' "$$" "$(hostname -f 2>/dev/null || hostname)" "${start}" "${end}" >&"${fd}" || true
      RAY_COMPONENT_PORT_LOCK_FD=${fd}
      RAY_COMPONENT_PORT_BLOCK_INDEX=${index}
      RAY_COMPONENT_PORT_BASE=${start}
      return 0
    fi
    eval "exec ${fd}>&-"
  done

  fail "No free local Ray component port block in ${pool_start}-${pool_end} with span=${span}"
}

negotiate_ray_head_port() {
  local selected_file wait_seconds poll_seconds safe_seed preferred_port pool_start pool_end pool_size i port
  local lock_root lock_file fd tmp_file

  selected_file=$1
  wait_seconds=$2
  poll_seconds=${RAY_HEAD_PORT_FILE_POLL_SECONDS}
  safe_seed="$(printf '%s' "${RAY_PORT_SEED}" | tr -c 'A-Za-z0-9_.-' '_')"

  if [[ "${NODE_RANK}" == "0" ]]; then
    command -v flock >/dev/null 2>&1 || fail "flock is required for Ray head port locking"
    mkdir -p "$(dirname "${selected_file}")"
    lock_root="${RAY_HEAD_PORT_LOCK_DIR:-/tmp/verl_omni_ray_head_ports_${USER:-unknown}}"
    mkdir -p "${lock_root}"

    preferred_port=${RAY_PORT}
    pool_start=${RAY_HEAD_PORT_POOL_START}
    pool_end=${RAY_HEAD_PORT_POOL_END}
    pool_size=$((pool_end - pool_start + 1))
    for ((i = 0; i < pool_size; i++)); do
      port=$((pool_start + ((preferred_port - pool_start + i) % pool_size)))
      if (( port >= RAY_MIN_WORKER_PORT && port <= RAY_MAX_WORKER_PORT )); then
        continue
      fi
      if is_true "${RAY_INCLUDE_DASHBOARD}" && (( port == RAY_DASHBOARD_PORT )); then
        continue
      fi
      if ! port_block_is_free "${port}" 1; then
        continue
      fi

      lock_file="${lock_root}/port${port}.lock"
      exec {fd}>"${lock_file}"
      if flock -n "${fd}"; then
        if ! port_block_is_free "${port}" 1; then
          eval "exec ${fd}>&-"
          continue
        fi
        printf 'pid=%s host=%s seed=%s port=%s\n' "$$" "$(hostname -f 2>/dev/null || hostname)" "${safe_seed}" "${port}" >&"${fd}" || true
        RAY_HEAD_PORT_LOCK_FD=${fd}
        RAY_PORT=${port}
        tmp_file="${selected_file}.tmp.$$"
        printf '%s\n' "${RAY_PORT}" > "${tmp_file}"
        mv "${tmp_file}" "${selected_file}"
        if (( RAY_PORT != preferred_port )); then
          echo "[info] Ray head port ${preferred_port} unavailable; selected ${RAY_PORT}"
        fi
        return 0
      fi
      eval "exec ${fd}>&-"
    done
    fail "No free Ray head port in ${RAY_HEAD_PORT_POOL_START}-${RAY_HEAD_PORT_POOL_END}"
  fi

  echo "[info] Waiting for Ray head port file ${selected_file}"
  for ((i = 0; i < wait_seconds; i += poll_seconds)); do
    if [[ -s "${selected_file}" ]]; then
      RAY_PORT="$(tr -d '[:space:]' < "${selected_file}")"
      if [[ ! "${RAY_PORT}" =~ ^[0-9]+$ ]]; then
        fail "Invalid Ray head port file ${selected_file}: ${RAY_PORT}"
      fi
      echo "[info] Ray worker rank ${NODE_RANK} using negotiated head port ${RAY_PORT}"
      return 0
    fi
    sleep "${poll_seconds}"
  done
  fail "Timed out waiting ${wait_seconds}s for Ray head port file ${selected_file}"
}

select_next_ray_head_port() {
  local selected_file failed_port pool_start pool_end pool_size i port tmp_file
  selected_file=$1
  failed_port=$2
  pool_start=${RAY_HEAD_PORT_POOL_START}
  pool_end=${RAY_HEAD_PORT_POOL_END}
  pool_size=$((pool_end - pool_start + 1))

  for ((i = 1; i <= pool_size; i++)); do
    port=$((pool_start + ((failed_port - pool_start + i) % pool_size)))
    if (( port >= RAY_MIN_WORKER_PORT && port <= RAY_MAX_WORKER_PORT )); then
      continue
    fi
    if is_true "${RAY_INCLUDE_DASHBOARD}" && (( port == RAY_DASHBOARD_PORT )); then
      continue
    fi
    if ! port_block_is_free "${port}" 1; then
      continue
    fi

    RAY_PORT=${port}
    RAY_ADDRESS="${RAY_HEAD_HOST}:${RAY_PORT}"
    tmp_file="${selected_file}.tmp.$$"
    printf '%s\n' "${RAY_PORT}" > "${tmp_file}"
    mv "${tmp_file}" "${selected_file}"
    echo "[warn] Ray head port ${failed_port} failed; retrying with ${RAY_PORT}"
    return 0
  done

  fail "No replacement Ray head port available after ${failed_port} failed"
}

refresh_ray_head_address_from_file() {
  local selected_file port
  selected_file=$1
  if [[ ! -s "${selected_file}" ]]; then
    return 0
  fi
  port="$(tr -d '[:space:]' < "${selected_file}")"
  if [[ ! "${port}" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if [[ "${port}" != "${RAY_PORT}" ]]; then
    RAY_PORT="${port}"
    RAY_ADDRESS="${RAY_HEAD_HOST}:${RAY_PORT}"
    echo "[info] Ray worker rank ${NODE_RANK} refreshed head address to ${RAY_ADDRESS}"
  fi
}

check_megatron_topology() {
  local label tp pp cp ep etp model_size attention_dp expert_model_size expert_dp
  label=$1
  tp=$2
  pp=$3
  cp=$4
  ep=$5
  etp=$6

  model_size=$((tp * pp * cp))
  ensure_divisible "${TRAIN_WORLD_GPUS}" "${model_size}" \
    "${label} world size check: TRAIN_WORLD_GPUS must be divisible by TP*PP*CP"
  attention_dp=$((TRAIN_WORLD_GPUS / model_size))

  expert_model_size=$((etp * ep * pp))
  ensure_divisible "${TRAIN_WORLD_GPUS}" "${expert_model_size}" \
    "${label} expert world size check: TRAIN_WORLD_GPUS must be divisible by ETP*EP*PP"
  expert_dp=$((TRAIN_WORLD_GPUS / expert_model_size))

  if (( ep > 1 && tp > 1 )) && ! is_true "${SEQUENCE_PARALLEL}"; then
    fail "${label} EP=${ep} with TP=${tp} requires SEQUENCE_PARALLEL=True in Megatron Core"
  fi

  echo "[info] ${label} topology: train_world=${TRAIN_WORLD_GPUS} attention_dp=${attention_dp} expert_dp=${expert_dp} tp/pp/cp/ep/etp=${tp}/${pp}/${cp}/${ep}/${etp}"
}

check_stage_config_alignment() {
  local stage_rollout_tp stage_devices stage_device_count stage_max_model_len
  stage_rollout_tp="$(awk '/^[[:space:]]*tensor_parallel_size:[[:space:]]*[0-9]+[[:space:]]*$/ {print $2; exit}' "${STAGE_CONFIG}")"
  if [[ -n "${stage_rollout_tp}" ]]; then
    if [[ ! "${stage_rollout_tp}" =~ ^[0-9]+$ ]]; then
      fail "stage config tensor_parallel_size must be an integer, got '${stage_rollout_tp}'"
    fi
    if (( stage_rollout_tp != ROLLOUT_TP )); then
      fail "ROLLOUT_TP=${ROLLOUT_TP} does not match ${STAGE_CONFIG} tensor_parallel_size=${stage_rollout_tp}"
    fi
  fi

  stage_devices="$(awk -F'"' '/^[[:space:]]*devices:[[:space:]]*"/ {print $2; exit}' "${STAGE_CONFIG}")"
  if [[ -n "${stage_devices}" ]]; then
    stage_device_count="$(awk -F',' '{print NF}' <<<"${stage_devices}")"
    if (( stage_device_count != ROLLOUT_TP )); then
      fail "ROLLOUT_TP=${ROLLOUT_TP} does not match ${STAGE_CONFIG} runtime.devices='${stage_devices}' (${stage_device_count} devices)"
    fi
  fi

  stage_max_model_len="$(awk '/^[[:space:]]*max_model_len:[[:space:]]*[0-9]+[[:space:]]*$/ {print $2; exit}' "${STAGE_CONFIG}")"
  if [[ -n "${stage_max_model_len}" && "${stage_max_model_len}" =~ ^[0-9]+$ ]]; then
    if (( stage_max_model_len < MAX_MODEL_LEN )); then
      fail "stage max_model_len=${stage_max_model_len} is smaller than MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    fi
  fi
}

validate_static_config() {
  local name
  for name in NNODES GPUS_PER_NODE TRAIN_NNODES TRAIN_GPUS_PER_NODE ROLLOUT_NNODES \
    ROLLOUT_GPUS_PER_NODE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH PPO_MINI_BATCH_SIZE \
    PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU N_RESP_PER_PROMPT \
    TOTAL_EPOCHS TOTAL_TRAINING_STEPS REQUIRE_BATCHES TOTAL_ROLLOUT_STEPS ACTOR_TP ACTOR_PP \
    ACTOR_CP ACTOR_EP ACTOR_ETP REF_TP REF_PP REF_CP REF_EP REF_ETP ROLLOUT_TP \
    ROLLOUT_DP ROLLOUT_AGENT_NUM_WORKERS ROLLOUT_MAX_NUM_SEQS \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS STAGE_INIT_TIMEOUT INIT_TIMEOUT \
    VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT RAY_NODE_CPUS \
    RAY_START_WAIT_SECONDS RAY_WORKER_JOIN_ATTEMPT_TIMEOUT \
    RAY_WORKER_PORT_SPAN RAY_WORKER_PORT_MIN_FREE; do
    require_positive_int "${name}"
  done
  require_nonnegative_int LR_WARMUP_STEPS
  require_unit_interval_float ROLLOUT_GPU_MEMORY_UTILIZATION

  WORLD_GPUS=$((NNODES * GPUS_PER_NODE))
  TRAIN_WORLD_GPUS=$((TRAIN_NNODES * TRAIN_GPUS_PER_NODE))
  ROLLOUT_WORLD_GPUS=$((ROLLOUT_NNODES * ROLLOUT_GPUS_PER_NODE))
  if (( TRAIN_WORLD_GPUS + ROLLOUT_WORLD_GPUS != WORLD_GPUS )); then
    fail "resource split mismatch: train ${TRAIN_WORLD_GPUS} + rollout ${ROLLOUT_WORLD_GPUS} != Ray world ${WORLD_GPUS}"
  fi
  if (( TRAIN_NNODES != NNODES || ROLLOUT_NNODES != NNODES )); then
    fail "this launcher expects same-node split; set TRAIN_NNODES and ROLLOUT_NNODES equal to NNODES"
  fi
  if (( TRAIN_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE != GPUS_PER_NODE )); then
    fail "per-node split mismatch: train ${TRAIN_GPUS_PER_NODE} + rollout ${ROLLOUT_GPUS_PER_NODE} != GPUS_PER_NODE=${GPUS_PER_NODE}"
  fi

  if [[ -z "${MAX_MODEL_LEN}" ]]; then
    MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
  fi
  require_positive_int MAX_MODEL_LEN

  if (( LR_WARMUP_STEPS >= TOTAL_TRAINING_STEPS )); then
    fail "LR_WARMUP_STEPS=${LR_WARMUP_STEPS} must be smaller than TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
  fi
  ensure_divisible "${PPO_MINI_BATCH_SIZE}" "${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    "PPO batch check: PPO_MINI_BATCH_SIZE must be divisible by PPO_MICRO_BATCH_SIZE_PER_GPU"
  if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
    fail "ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} must be >= MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  fi
  if (( ROLLOUT_MAX_NUM_SEQS < N_RESP_PER_PROMPT )); then
    fail "ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS} must be >= N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
  fi
  if (( ROLLOUT_TP > ROLLOUT_GPUS_PER_NODE )); then
    fail "ROLLOUT_TP=${ROLLOUT_TP} must be <= ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE}"
  fi
  if (( ROLLOUT_DP != 1 )); then
    fail "ROLLOUT_DP=${ROLLOUT_DP} is not supported for this vLLM-Omni AR async launcher; use standalone TP${ROLLOUT_TP} replicas with ROLLOUT_DP=1"
  fi
  ensure_divisible "${ROLLOUT_WORLD_GPUS}" "$((ROLLOUT_TP * ROLLOUT_DP))" \
    "rollout replica check: ROLLOUT_WORLD_GPUS must be divisible by ROLLOUT_TP*ROLLOUT_DP"
  if is_true "${SEQUENCE_PARALLEL}" && [[ "${ATTENTION_BACKEND}" == "local" ]]; then
    fail "ATTENTION_BACKEND=local selects the torch-norm path, which cannot run with sequence parallel"
  fi
  case "${MOE_TOKEN_DISPATCHER_TYPE}" in
    alltoall|allgather|flex) ;;
    *) fail "MOE_TOKEN_DISPATCHER_TYPE must be alltoall, allgather, or flex, got ${MOE_TOKEN_DISPATCHER_TYPE}" ;;
  esac
  if ! is_true "${ROLLOUT_CALCULATE_LOG_PROBS}"; then
    fail "fully_async rollouter requires ROLLOUT_CALCULATE_LOG_PROBS=True"
  fi
  case "${ROLLOUT_LOGPROBS_MODE}" in
    raw_logprobs|processed_logprobs) ;;
    *) fail "ROLLOUT_LOGPROBS_MODE must be raw_logprobs or processed_logprobs, got ${ROLLOUT_LOGPROBS_MODE}" ;;
  esac
  case "${OMNI_LB_POLICY}" in
    random|round-robin|least-queue-length) ;;
    *) fail "OMNI_LB_POLICY must be random, round-robin, or least-queue-length, got ${OMNI_LB_POLICY}" ;;
  esac
  if grep -Eq '^[[:space:]]*logprobs_mode:' "${STAGE_CONFIG}" \
    && ! grep -Eq "^[[:space:]]*logprobs_mode:[[:space:]]*${ROLLOUT_LOGPROBS_MODE}([[:space:]]*#.*)?$" "${STAGE_CONFIG}"; then
    fail "STAGE_CONFIG=${STAGE_CONFIG} has a per-stage logprobs_mode that does not match ROLLOUT_LOGPROBS_MODE=${ROLLOUT_LOGPROBS_MODE}; per-stage YAML overrides top-level vLLM-Omni args"
  fi
  if is_true "${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-0}" && is_true "${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}"; then
    fail "cannot require both raw and processed rollout logprobs"
  fi
  if is_true "${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-0}" && [[ "${ROLLOUT_LOGPROBS_MODE}" != "raw_logprobs" ]]; then
    fail "raw logprob parity requires ROLLOUT_LOGPROBS_MODE=raw_logprobs, got ${ROLLOUT_LOGPROBS_MODE}"
  fi
  if is_true "${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}" && [[ "${ROLLOUT_LOGPROBS_MODE}" != "processed_logprobs" ]]; then
    fail "processed rollout logprob run requires ROLLOUT_LOGPROBS_MODE=processed_logprobs, got ${ROLLOUT_LOGPROBS_MODE}"
  fi

  require_port RAY_PORT
  if is_true "${RAY_INCLUDE_DASHBOARD}"; then
    require_port RAY_DASHBOARD_PORT
  fi
  require_port RAY_MIN_WORKER_PORT
  require_port RAY_MAX_WORKER_PORT
  require_port RAY_HEAD_PORT_POOL_START
  require_port RAY_HEAD_PORT_POOL_END
  require_positive_int RAY_HEAD_PORT_FILE_WAIT_SECONDS
  require_positive_int RAY_HEAD_PORT_FILE_POLL_SECONDS
  require_port RAY_COMPONENT_PORT_POOL_START
  require_port RAY_COMPONENT_PORT_POOL_END
  require_positive_int RAY_COMPONENT_PORT_SPAN
  require_port RAY_NODE_MANAGER_PORT
  require_port RAY_OBJECT_MANAGER_PORT
  require_port RAY_DASHBOARD_AGENT_LISTEN_PORT
  require_port RAY_DASHBOARD_AGENT_GRPC_PORT
  require_port RAY_RUNTIME_ENV_AGENT_PORT
  require_port RAY_METRICS_EXPORT_PORT
  require_port RAY_CLIENT_SERVER_PORT
  require_port RAY_WORKER_PORT_POOL_START
  require_port RAY_WORKER_PORT_POOL_END
  require_port VERL_OMNI_VLLM_PORT_POOL_START
  require_port VERL_OMNI_VLLM_PORT_POOL_END
  require_positive_int VERL_OMNI_VLLM_PORT_STRIDE
  require_positive_int VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET
  require_nonnegative_int VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD
  require_nonnegative_int VERL_OMNI_VLLM_STAGE_CORE_PORT_SPREAD
  require_positive_int VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL
  require_positive_int VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP
  require_positive_int VERL_OMNI_VLLM_PORT_ACTOR_SLOTS
  require_port VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START
  require_port VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END
  require_port VERL_OMNI_MASTER_ZMQ_PORT_POOL_START
  require_port VERL_OMNI_MASTER_ZMQ_PORT_POOL_END
  require_positive_int VERL_OMNI_MASTER_ZMQ_PORT_SPAN
  require_port VERL_OMNI_MASTER_ZMQ_PORT_BASE
  require_port VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START
  require_port VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END
  require_port VERL_OMNI_VLLM_DIST_MASTER_PORT
  if (( RAY_MIN_WORKER_PORT > RAY_MAX_WORKER_PORT )); then
    fail "RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT} must be <= RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT}"
  fi
  if (( RAY_HEAD_PORT_POOL_START > RAY_HEAD_PORT_POOL_END )); then
    fail "RAY_HEAD_PORT_POOL_START=${RAY_HEAD_PORT_POOL_START} must be <= RAY_HEAD_PORT_POOL_END=${RAY_HEAD_PORT_POOL_END}"
  fi
  if (( RAY_WORKER_PORT_POOL_START > RAY_WORKER_PORT_POOL_END )); then
    fail "RAY_WORKER_PORT_POOL_START=${RAY_WORKER_PORT_POOL_START} must be <= RAY_WORKER_PORT_POOL_END=${RAY_WORKER_PORT_POOL_END}"
  fi
  if (( RAY_WORKER_PORT_MIN_FREE > RAY_WORKER_PORT_SPAN )); then
    fail "RAY_WORKER_PORT_MIN_FREE=${RAY_WORKER_PORT_MIN_FREE} must be <= RAY_WORKER_PORT_SPAN=${RAY_WORKER_PORT_SPAN}"
  fi
  if (( RAY_COMPONENT_PORT_POOL_START > RAY_COMPONENT_PORT_POOL_END )); then
    fail "RAY_COMPONENT_PORT_POOL_START=${RAY_COMPONENT_PORT_POOL_START} must be <= RAY_COMPONENT_PORT_POOL_END=${RAY_COMPONENT_PORT_POOL_END}"
  fi
  if (( VERL_OMNI_VLLM_PORT_POOL_START > VERL_OMNI_VLLM_PORT_POOL_END )); then
    fail "VERL_OMNI_VLLM_PORT_POOL_START=${VERL_OMNI_VLLM_PORT_POOL_START} must be <= VERL_OMNI_VLLM_PORT_POOL_END=${VERL_OMNI_VLLM_PORT_POOL_END}"
  fi
  vllm_port_pool_size=$((VERL_OMNI_VLLM_PORT_POOL_END - VERL_OMNI_VLLM_PORT_POOL_START + 1))
  vllm_port_required_span=$((VERL_OMNI_VLLM_PORT_STRIDE * VERL_OMNI_VLLM_PORT_ACTOR_SLOTS))
  if (( vllm_port_pool_size < vllm_port_required_span )); then
    fail "vLLM port pool ${VERL_OMNI_VLLM_PORT_POOL_START}-${VERL_OMNI_VLLM_PORT_POOL_END} is too small for actor_slots=${VERL_OMNI_VLLM_PORT_ACTOR_SLOTS} stride=${VERL_OMNI_VLLM_PORT_STRIDE}"
  fi
  vllm_stage_core_required_tail=$((VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET + VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD + VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP + VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL))
  if (( VERL_OMNI_VLLM_PORT_STRIDE <= vllm_stage_core_required_tail )); then
    fail "VERL_OMNI_VLLM_PORT_STRIDE=${VERL_OMNI_VLLM_PORT_STRIDE} leaves too little stage-core tail for offset=${VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET}, guard=${VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD}, direct_gap=${VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP}, min_tail=${VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL}"
  fi
  if (( VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START > VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END )); then
    fail "VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START=${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START} must be <= VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END=${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END}"
  fi
  if (( VERL_OMNI_MASTER_ZMQ_PORT_POOL_START > VERL_OMNI_MASTER_ZMQ_PORT_POOL_END )); then
    fail "VERL_OMNI_MASTER_ZMQ_PORT_POOL_START=${VERL_OMNI_MASTER_ZMQ_PORT_POOL_START} must be <= VERL_OMNI_MASTER_ZMQ_PORT_POOL_END=${VERL_OMNI_MASTER_ZMQ_PORT_POOL_END}"
  fi
  if (( VERL_OMNI_MASTER_ZMQ_PORT_BASE < VERL_OMNI_MASTER_ZMQ_PORT_POOL_START || VERL_OMNI_MASTER_ZMQ_PORT_BASE > VERL_OMNI_MASTER_ZMQ_PORT_POOL_END )); then
    fail "VERL_OMNI_MASTER_ZMQ_PORT_BASE=${VERL_OMNI_MASTER_ZMQ_PORT_BASE} must be within ${VERL_OMNI_MASTER_ZMQ_PORT_POOL_START}-${VERL_OMNI_MASTER_ZMQ_PORT_POOL_END}"
  fi
  if (( VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START > VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END )); then
    fail "VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START=${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START} must be <= VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END=${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END}"
  fi
  if (( VERL_OMNI_VLLM_DIST_MASTER_PORT < VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START || VERL_OMNI_VLLM_DIST_MASTER_PORT > VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END )); then
    fail "VERL_OMNI_VLLM_DIST_MASTER_PORT=${VERL_OMNI_VLLM_DIST_MASTER_PORT} must be within ${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START}-${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END}"
  fi
  if is_true "${RAY_INCLUDE_DASHBOARD}" && (( RAY_PORT == RAY_DASHBOARD_PORT )); then
    fail "RAY_PORT and RAY_DASHBOARD_PORT must be different, got ${RAY_PORT}"
  fi
  ensure_port_outside_worker_range RAY_PORT
  if is_true "${RAY_INCLUDE_DASHBOARD}"; then
    ensure_port_outside_worker_range RAY_DASHBOARD_PORT
  fi
  ensure_port_outside_worker_range RAY_NODE_MANAGER_PORT
  ensure_port_outside_worker_range RAY_OBJECT_MANAGER_PORT
  ensure_port_outside_worker_range RAY_DASHBOARD_AGENT_LISTEN_PORT
  ensure_port_outside_worker_range RAY_DASHBOARD_AGENT_GRPC_PORT
  ensure_port_outside_worker_range RAY_RUNTIME_ENV_AGENT_PORT
  ensure_port_outside_worker_range RAY_METRICS_EXPORT_PORT
  ensure_port_outside_worker_range RAY_CLIENT_SERVER_PORT

  check_megatron_topology actor "${ACTOR_TP}" "${ACTOR_PP}" "${ACTOR_CP}" "${ACTOR_EP}" "${ACTOR_ETP}"
  check_megatron_topology ref "${REF_TP}" "${REF_PP}" "${REF_CP}" "${REF_EP}" "${REF_ETP}"
  check_stage_config_alignment
}

NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TRAIN_NNODES=${TRAIN_NNODES:-${NNODES}}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-${VERL_OMNI_TRAIN_GPUS_PER_NODE:-4}}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-${VERL_OMNI_ROLLOUT_NNODES:-${NNODES}}}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-${VERL_OMNI_ROLLOUT_GPUS_PER_NODE:-4}}

export VERL_OMNI_FORCE_STANDALONE_ROLLOUT=${VERL_OMNI_FORCE_STANDALONE_ROLLOUT:-1}
export VERL_OMNI_RESOURCE_SPLIT_IMPL=${VERL_OMNI_RESOURCE_SPLIT_IMPL:-standalone_rollout}
export VERL_OMNI_TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE}"
export VERL_OMNI_ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE}"
export VERL_OMNI_ROLLOUT_NNODES="${ROLLOUT_NNODES}"

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-4}
IGNORE_EOS=${IGNORE_EOS:-True}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TOTAL_TRAINING_STEPS))}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}

ACTOR_TP=${ACTOR_TP:-2}
ACTOR_PP=${ACTOR_PP:-2}
ACTOR_CP=${ACTOR_CP:-1}
ACTOR_EP=${ACTOR_EP:-4}
ACTOR_ETP=${ACTOR_ETP:-1}
REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_CP=${REF_CP:-${ACTOR_CP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_ETP=${REF_ETP:-${ACTOR_ETP}}
ROLLOUT_TP=${ROLLOUT_TP:-4}
# Default rollout topology is four independent TP4 standalone replicas on the
# 16 rollout GPUs.  Using vLLM internal DP4 for Qwen3-Omni-MoE pulls in vLLM
# 0.22's DP coordinator and has proven fragile on Luban multi-node bringup.
ROLLOUT_DP=${ROLLOUT_DP:-1}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS=${VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS:-model}
VERL_OMNI_QWEN3_OMNI_SP_SCATTER_POSITION_IDS=${VERL_OMNI_QWEN3_OMNI_SP_SCATTER_POSITION_IDS:-0}
VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL=${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-}
VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-2}
VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS:-2}
VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS:-16}
VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT:-2}
VERL_OMNI_MEGATRON_LOGPROB_COMPONENT_AUDIT=${VERL_OMNI_MEGATRON_LOGPROB_COMPONENT_AUDIT:-0}
VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT=${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT:-0}
VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS=${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS:-1,4,6,16,32,48}
VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT=${VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT:-0}
VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT_LAYERS=${VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT_LAYERS:-1}
VERL_OMNI_MEGATRON_MOE_REPLAY_METADATA_AUDIT=${VERL_OMNI_MEGATRON_MOE_REPLAY_METADATA_AUDIT:-0}
VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE=${VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE:-0}
VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS=${VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS:-1}
OFFLOAD=${OFFLOAD:-True}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-True}
ROLLOUT_LOGPROBS_MODE=${ROLLOUT_LOGPROBS_MODE:-raw_logprobs}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.9}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_CORR_BYPASS_MODE=${ROLLOUT_CORR_BYPASS_MODE:-False}
OMNI_LB_POLICY=${OMNI_LB_POLICY:-random}
STAGE_INIT_TIMEOUT=${STAGE_INIT_TIMEOUT:-1800}
INIT_TIMEOUT=${INIT_TIMEOUT:-${STAGE_INIT_TIMEOUT}}
export VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT=${VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT:-${STAGE_INIT_TIMEOUT}}

MASKED_SOFTMAX_FUSION=${MASKED_SOFTMAX_FUSION:-False}
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-False}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
GRADIENT_ACCUMULATION_FUSION=${GRADIENT_ACCUMULATION_FUSION:-False}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-unfused}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-True}

RAY_TMPDIR=${RAY_TMPDIR:-/tmp/verl_ray_${USER}_qwen3_omni_async_32gpu}
RAY_AUTOSTART=${RAY_AUTOSTART:-1}
RAY_PORT_SEED=${RAY_PORT_SEED:-${LUBAN_JOB_ID:-${AIP_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-${APP_ID:-${K8S_APP_ID:-${RUN_ID_RAW}}}}}}}}
RAY_WORKER_PORT_SEED=${RAY_WORKER_PORT_SEED:-${RAY_PORT_SEED}}
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-0}
RAY_PORT_WAS_SET=${RAY_PORT+x}
RAY_HEAD_PORT_POOL_START=${RAY_HEAD_PORT_POOL_START:-20000}
RAY_HEAD_PORT_POOL_END=${RAY_HEAD_PORT_POOL_END:-29999}
RAY_HEAD_PORT_POOL_SIZE=$((RAY_HEAD_PORT_POOL_END - RAY_HEAD_PORT_POOL_START + 1))
if (( RAY_HEAD_PORT_POOL_SIZE <= 0 )); then
  fail "RAY head port pool ${RAY_HEAD_PORT_POOL_START}-${RAY_HEAD_PORT_POOL_END} is empty"
fi
RAY_HEAD_PORT_OFFSET=$(stable_hash_mod "${RAY_PORT_SEED}" "${RAY_HEAD_PORT_POOL_SIZE}")
RAY_PORT=${RAY_PORT:-$((RAY_HEAD_PORT_POOL_START + RAY_HEAD_PORT_OFFSET))}
if is_true "${RAY_INCLUDE_DASHBOARD}"; then
  RAY_DASHBOARD_PORT_POOL_START=${RAY_DASHBOARD_PORT_POOL_START:-10000}
  RAY_DASHBOARD_PORT_POOL_END=${RAY_DASHBOARD_PORT_POOL_END:-19999}
  RAY_DASHBOARD_PORT_POOL_SIZE=$((RAY_DASHBOARD_PORT_POOL_END - RAY_DASHBOARD_PORT_POOL_START + 1))
  if (( RAY_DASHBOARD_PORT_POOL_SIZE <= 0 )); then
    fail "Ray dashboard port pool ${RAY_DASHBOARD_PORT_POOL_START}-${RAY_DASHBOARD_PORT_POOL_END} is empty"
  fi
  RAY_DASHBOARD_PORT_OFFSET=$(stable_hash_mod "${RAY_PORT_SEED}_dashboard" "${RAY_DASHBOARD_PORT_POOL_SIZE}")
  RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-$((RAY_DASHBOARD_PORT_POOL_START + RAY_DASHBOARD_PORT_OFFSET))}
else
  RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-0}
fi
RAY_WORKER_PORT_POOL_START=${RAY_WORKER_PORT_POOL_START:-30000}
RAY_WORKER_PORT_POOL_END=${RAY_WORKER_PORT_POOL_END:-59999}
RAY_WORKER_PORT_SPAN=${RAY_WORKER_PORT_SPAN:-256}
# Ray assigns CoreWorker gRPC listeners from this range. A partially occupied
# range can pass startup and then fail nondeterministically when a driver picks
# an occupied port, so only claim completely free blocks by default.
RAY_WORKER_PORT_MIN_FREE=${RAY_WORKER_PORT_MIN_FREE:-${RAY_WORKER_PORT_SPAN}}
RAY_WORKER_PORT_LOCAL_PROBE=${RAY_WORKER_PORT_LOCAL_PROBE:-1}
RAY_MIN_WORKER_PORT_WAS_SET=${RAY_MIN_WORKER_PORT+x}
RAY_MAX_WORKER_PORT_WAS_SET=${RAY_MAX_WORKER_PORT+x}
require_positive_int RAY_WORKER_PORT_SPAN
RAY_WORKER_PORT_POOL_SIZE=$((RAY_WORKER_PORT_POOL_END - RAY_WORKER_PORT_POOL_START + 1))
RAY_WORKER_PORT_BLOCKS=$((RAY_WORKER_PORT_POOL_SIZE / RAY_WORKER_PORT_SPAN))
if (( RAY_WORKER_PORT_BLOCKS <= 0 )); then
  fail "RAY_WORKER_PORT_SPAN=${RAY_WORKER_PORT_SPAN} is too large for worker port pool"
fi
RAY_WORKER_PORT_PREFERRED_BLOCK_INDEX=$(stable_hash_mod "${RAY_WORKER_PORT_SEED}" "${RAY_WORKER_PORT_BLOCKS}")
RAY_WORKER_PORT_NODE_RANK_HINT=${RAY_WORKER_PORT_NODE_RANK_HINT:-${DISTRIBUTED_NODE_RANK:-${NODE_RANK:-${RAY_NODE_RANK:-${VC_TASK_INDEX:-${VK_TASK_INDEX:-0}}}}}}
if [[ ! "${RAY_WORKER_PORT_NODE_RANK_HINT}" =~ ^[0-9]+$ ]]; then
  RAY_WORKER_PORT_NODE_RANK_HINT=0
fi
RAY_WORKER_PORT_BLOCK_INDEX=$(((RAY_WORKER_PORT_PREFERRED_BLOCK_INDEX + RAY_WORKER_PORT_NODE_RANK_HINT) % RAY_WORKER_PORT_BLOCKS))
RAY_WORKER_PORT_BASE=$((RAY_WORKER_PORT_POOL_START + RAY_WORKER_PORT_BLOCK_INDEX * RAY_WORKER_PORT_SPAN))
if [[ -z "${RAY_MIN_WORKER_PORT_WAS_SET}" && -z "${RAY_MAX_WORKER_PORT_WAS_SET}" && "${RAY_WORKER_PORT_LOCAL_PROBE}" == "1" ]]; then
  claim_ray_worker_port_block \
    "${RAY_WORKER_PORT_POOL_START}" \
    "${RAY_WORKER_PORT_POOL_END}" \
    "${RAY_WORKER_PORT_SPAN}" \
    "${RAY_WORKER_PORT_PREFERRED_BLOCK_INDEX}" \
    "${RAY_WORKER_PORT_SEED}" \
    "${RAY_WORKER_PORT_MIN_FREE}"
else
  RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-${RAY_WORKER_PORT_BASE}}
  RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-$((RAY_WORKER_PORT_BASE + RAY_WORKER_PORT_SPAN - 1))}
fi
RAY_COMPONENT_PORT_POOL_START=${RAY_COMPONENT_PORT_POOL_START:-12000}
RAY_COMPONENT_PORT_POOL_END=${RAY_COMPONENT_PORT_POOL_END:-19999}
RAY_COMPONENT_PORT_SPAN=${RAY_COMPONENT_PORT_SPAN:-16}
RAY_COMPONENT_PORT_LOCAL_PROBE=${RAY_COMPONENT_PORT_LOCAL_PROBE:-1}
require_positive_int RAY_COMPONENT_PORT_SPAN
RAY_COMPONENT_PORT_POOL_SIZE=$((RAY_COMPONENT_PORT_POOL_END - RAY_COMPONENT_PORT_POOL_START + 1))
RAY_COMPONENT_PORT_BLOCKS=$((RAY_COMPONENT_PORT_POOL_SIZE / RAY_COMPONENT_PORT_SPAN))
if (( RAY_COMPONENT_PORT_BLOCKS <= 0 )); then
  fail "RAY_COMPONENT_PORT_SPAN=${RAY_COMPONENT_PORT_SPAN} is too large for component port pool"
fi
RAY_COMPONENT_PORT_PREFERRED_BLOCK_INDEX=$(stable_hash_mod "${RAY_WORKER_PORT_SEED}_ray_components" "${RAY_COMPONENT_PORT_BLOCKS}")
RAY_COMPONENT_PORT_NODE_RANK_HINT=${RAY_COMPONENT_PORT_NODE_RANK_HINT:-${RAY_WORKER_PORT_NODE_RANK_HINT}}
if [[ ! "${RAY_COMPONENT_PORT_NODE_RANK_HINT}" =~ ^[0-9]+$ ]]; then
  RAY_COMPONENT_PORT_NODE_RANK_HINT=0
fi
RAY_COMPONENT_PORT_BLOCK_INDEX=$(((RAY_COMPONENT_PORT_PREFERRED_BLOCK_INDEX + RAY_COMPONENT_PORT_NODE_RANK_HINT) % RAY_COMPONENT_PORT_BLOCKS))
RAY_COMPONENT_PORT_BASE=${RAY_COMPONENT_PORT_BASE:-$((RAY_COMPONENT_PORT_POOL_START + RAY_COMPONENT_PORT_BLOCK_INDEX * RAY_COMPONENT_PORT_SPAN))}
if [[ "${RAY_COMPONENT_PORT_LOCAL_PROBE}" == "1" ]]; then
  claim_ray_component_port_block \
    "${RAY_COMPONENT_PORT_POOL_START}" \
    "${RAY_COMPONENT_PORT_POOL_END}" \
    "${RAY_COMPONENT_PORT_SPAN}" \
    "${RAY_COMPONENT_PORT_PREFERRED_BLOCK_INDEX}" \
    "${RAY_WORKER_PORT_SEED}_ray_components_${RAY_COMPONENT_PORT_NODE_RANK_HINT}"
fi
RAY_NODE_MANAGER_PORT=${RAY_NODE_MANAGER_PORT:-${RAY_COMPONENT_PORT_BASE}}
RAY_OBJECT_MANAGER_PORT=${RAY_OBJECT_MANAGER_PORT:-$((RAY_COMPONENT_PORT_BASE + 1))}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-$((RAY_COMPONENT_PORT_BASE + 2))}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-$((RAY_COMPONENT_PORT_BASE + 3))}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-$((RAY_COMPONENT_PORT_BASE + 4))}
RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-$((RAY_COMPONENT_PORT_BASE + 5))}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT:-$((RAY_COMPONENT_PORT_BASE + 6))}
export VERL_OMNI_VLLM_PORT_SEED=${VERL_OMNI_VLLM_PORT_SEED:-${RAY_WORKER_PORT_SEED}}
export VERL_OMNI_VLLM_PORT_POOL_START=${VERL_OMNI_VLLM_PORT_POOL_START:-61000}
export VERL_OMNI_VLLM_PORT_POOL_END=${VERL_OMNI_VLLM_PORT_POOL_END:-65099}
export VERL_OMNI_VLLM_PORT_STRIDE=${VERL_OMNI_VLLM_PORT_STRIDE:-512}
export VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET=${VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET:-128}
export VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD=${VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD:-64}
export VERL_OMNI_VLLM_STAGE_CORE_PORT_SPREAD=${VERL_OMNI_VLLM_STAGE_CORE_PORT_SPREAD:-128}
export VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL=${VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL:-64}
export VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP=${VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP:-32}
export VERL_OMNI_RESPECT_EXISTING_VLLM_PORT=${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT:-0}
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE=${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}
export VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START=${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START:-10000}
export VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END=${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END:-11999}
ROLLOUT_REPLICAS_FOR_VLLM_PORT=$(((ROLLOUT_NNODES * ROLLOUT_GPUS_PER_NODE) / (ROLLOUT_TP * ROLLOUT_DP)))
if (( ROLLOUT_REPLICAS_FOR_VLLM_PORT <= 0 )); then
  ROLLOUT_REPLICAS_FOR_VLLM_PORT=1
fi
VLLM_PORT_ACTOR_SLOTS_DEFAULT=${ROLLOUT_REPLICAS_FOR_VLLM_PORT}
export VERL_OMNI_VLLM_PORT_ACTOR_SLOTS=${VERL_OMNI_VLLM_PORT_ACTOR_SLOTS:-${VLLM_PORT_ACTOR_SLOTS_DEFAULT}}
export VERL_OMNI_MASTER_ZMQ_PORT_POOL_START=${VERL_OMNI_MASTER_ZMQ_PORT_POOL_START:-60000}
export VERL_OMNI_MASTER_ZMQ_PORT_POOL_END=${VERL_OMNI_MASTER_ZMQ_PORT_POOL_END:-60999}
export VERL_OMNI_MASTER_ZMQ_PORT_SPAN=${VERL_OMNI_MASTER_ZMQ_PORT_SPAN:-64}
OMNI_MASTER_ZMQ_USABLE=$((VERL_OMNI_MASTER_ZMQ_PORT_POOL_END - VERL_OMNI_MASTER_ZMQ_PORT_POOL_START - VERL_OMNI_MASTER_ZMQ_PORT_SPAN + 2))
if (( OMNI_MASTER_ZMQ_USABLE <= 0 )); then
  fail "Omni master ZMQ port pool ${VERL_OMNI_MASTER_ZMQ_PORT_POOL_START}-${VERL_OMNI_MASTER_ZMQ_PORT_POOL_END} is too small for span ${VERL_OMNI_MASTER_ZMQ_PORT_SPAN}"
fi
OMNI_MASTER_ZMQ_OFFSET=$(stable_hash_mod "${VERL_OMNI_VLLM_PORT_SEED}_omni_master" "${OMNI_MASTER_ZMQ_USABLE}")
export VERL_OMNI_MASTER_ZMQ_PORT_BASE=${VERL_OMNI_MASTER_ZMQ_PORT_BASE:-$((VERL_OMNI_MASTER_ZMQ_PORT_POOL_START + OMNI_MASTER_ZMQ_OFFSET))}
export VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START=${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START:-65100}
export VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END=${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END:-65535}
VLLM_DIST_MASTER_PORT_POOL_SIZE=$((VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END - VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START + 1))
if (( VLLM_DIST_MASTER_PORT_POOL_SIZE <= 0 )); then
  fail "vLLM distributed master port pool ${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START}-${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END} is empty"
fi
VLLM_DIST_MASTER_PORT_OFFSET=$(stable_hash_mod "${VERL_OMNI_VLLM_PORT_SEED}_vllm_dist_master" "${VLLM_DIST_MASTER_PORT_POOL_SIZE}")
export VERL_OMNI_VLLM_DIST_MASTER_PORT=${VERL_OMNI_VLLM_DIST_MASTER_PORT:-$((VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START + VLLM_DIST_MASTER_PORT_OFFSET))}
RAY_NODE_CPUS=${RAY_NODE_CPUS:-32}
RAY_WORKER_JOIN_ATTEMPT_TIMEOUT=${RAY_WORKER_JOIN_ATTEMPT_TIMEOUT:-60}
RAY_START_WAIT_SECONDS=${RAY_START_WAIT_SECONDS:-900}
RAY_HEAD_START_ATTEMPTS=${RAY_HEAD_START_ATTEMPTS:-16}
RAY_NODE_ROLE=${RAY_NODE_ROLE:-}
CONFIG_ONLY=${CONFIG_ONLY:-0}
RAY_ONLY=${RAY_ONLY:-0}
RAY_HEAD_PORT_NEGOTIATE=${RAY_HEAD_PORT_NEGOTIATE:-1}
RAY_HEAD_PORT_FILE_WAIT_SECONDS=${RAY_HEAD_PORT_FILE_WAIT_SECONDS:-300}
RAY_HEAD_PORT_FILE_POLL_SECONDS=${RAY_HEAD_PORT_FILE_POLL_SECONDS:-1}

export VERL_OMNI_SKIP_WEIGHT_UPDATE=${VERL_OMNI_SKIP_WEIGHT_UPDATE:-0}
export VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS=${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS:-0}
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME=${VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME:-1}
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP=${VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP:-1}
export VERL_OMNI_WAKE_TAGS=${VERL_OMNI_WAKE_TAGS:-kv_cache,weights}
export VERL_OMNI_SLEEP_LEVEL=${VERL_OMNI_SLEEP_LEVEL:-1}
export VERL_OMNI_WEIGHT_SYNC_DEBUG=${VERL_OMNI_WEIGHT_SYNC_DEBUG:-0}
export VERL_OMNI_WEIGHT_SYNC_TOPO_DEBUG=${VERL_OMNI_WEIGHT_SYNC_TOPO_DEBUG:-0}
export VERL_OMNI_WEIGHT_ROUTE_DEBUG=${VERL_OMNI_WEIGHT_ROUTE_DEBUG:-0}
export VERL_OMNI_WEIGHT_VALUE_DEBUG=${VERL_OMNI_WEIGHT_VALUE_DEBUG:-0}
export VERL_OMNI_PRE_IPC_LOCAL_COPY_DEBUG=${VERL_OMNI_PRE_IPC_LOCAL_COPY_DEBUG:-0}
export VERL_OMNI_PRE_LOAD_LOCAL_COPY_DEBUG=${VERL_OMNI_PRE_LOAD_LOCAL_COPY_DEBUG:-0}
export VERL_OMNI_WAKE_CUMEM_DEBUG=${VERL_OMNI_WAKE_CUMEM_DEBUG:-0}
export VERL_OMNI_PHASE_GENERATION_DEBUG=${VERL_OMNI_PHASE_GENERATION_DEBUG:-0}
export VERL_OMNI_PHASE_GENERATION_STRICT=${VERL_OMNI_PHASE_GENERATION_STRICT:-0}
export VERL_OMNI_PHASE_GENERATION_PHASES=${VERL_OMNI_PHASE_GENERATION_PHASES:-before_update_with_kv_cache,after_update_before_resume_kv_cache,after_resume_kv_cache}
export VERL_OMNI_PHASE_GENERATION_MAX_TOKENS=${VERL_OMNI_PHASE_GENERATION_MAX_TOKENS:-64}
export VERL_OMNI_PHASE_GENERATION_MIN_TOKENS=${VERL_OMNI_PHASE_GENERATION_MIN_TOKENS:-1}
export VERL_OMNI_PHASE_GENERATION_TEMPERATURE=${VERL_OMNI_PHASE_GENERATION_TEMPERATURE:-0}
export VERL_OMNI_PHASE_GENERATION_TOP_P=${VERL_OMNI_PHASE_GENERATION_TOP_P:-1}
export VERL_OMNI_PHASE_GENERATION_IGNORE_EOS=${VERL_OMNI_PHASE_GENERATION_IGNORE_EOS:-0}
export VERL_OMNI_GENERATION_DEBUG_LIMIT=${VERL_OMNI_GENERATION_DEBUG_LIMIT:-32}
export VERL_OMNI_LOGPROB_DEBUG_LIMIT=${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-0}
export VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT=${VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT:-0}
export VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL=${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL:-}
export VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS=${VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS:-0}
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT=${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-0}
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL=${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-}
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS=${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-4}
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL=${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL:-}
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS=${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS:-4}
export VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS=${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS:-16}
export VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL=${VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL:-}
export VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT=${VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT:-512}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-8}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=${VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD:-0}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS:-0}
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS=${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-1}
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS=${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}
export VERL_OMNI_STAGE_CORE_DIAG_DIR=${VERL_OMNI_STAGE_CORE_DIAG_DIR:-${LOG_DIR}/stage_core_crashes/${RUN_ID}}
mkdir -p "${VERL_OMNI_STAGE_CORE_DIAG_DIR}"

validate_static_config

TASK_ROLE=${DISTRIBUTED_TASK_ROLE:-${VC_TASK_ROLE:-${ROLE_NAME:-${TASK_ROLE:-}}}}
TASK_INDEX=${VC_TASK_INDEX:-${VK_TASK_INDEX:-${DISTRIBUTED_NODE_RANK:-${TASK_INDEX:-}}}}

detect_node_rank_from_name() {
  local name short
  name=$1
  short="${name%%.*}"

  if [[ "${short}" =~ -master-([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${short}" =~ -worker-([0-9]+)$ ]]; then
    echo $((BASH_REMATCH[1] + 1))
    return 0
  fi
  if [[ "${short}" =~ -([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

detect_node_rank_from_hostname() {
  local name
  for name in "${POD_NAME:-}" "${K8S_POD_NAME:-}" "${HOSTNAME:-}" "$(hostname -f 2>/dev/null || true)" "$(hostname -s 2>/dev/null || true)"; do
    if [[ -n "${name}" ]] && detect_node_rank_from_name "${name}"; then
      return 0
    fi
  done
  return 1
}

detect_integer_env() {
  local name value
  for name in "$@"; do
    value="${!name:-}"
    if [[ "${value}" =~ ^[0-9]+$ ]]; then
      echo "${value}"
      return 0
    fi
  done
  return 1
}

detect_node_rank() {
  if detect_integer_env NODE_RANK RAY_NODE_RANK DISTRIBUTED_NODE_RANK; then
    return 0
  elif [[ -n "${TASK_ROLE}" && "${TASK_INDEX}" =~ ^[0-9]+$ ]]; then
    if [[ "${TASK_ROLE}" == "master" || "${TASK_ROLE}" == "head" || "${TASK_ROLE}" == "chief" ]]; then
      echo "${TASK_INDEX}"
    else
      echo $((TASK_INDEX + 1))
    fi
  elif detect_node_rank_from_hostname; then
    return 0
  elif detect_integer_env GROUP_RANK ROLE_INDEX POD_INDEX INDEX TASK_INDEX \
      VC_TASK_INDEX VK_TASK_INDEX MATRIX_TASK_INDEX AIP_WORKER_INDEX; then
    return 0
  elif [[ -n "${RANK:-}" && -n "${LOCAL_RANK:-}" ]]; then
    echo $((RANK / GPUS_PER_NODE))
  else
    echo 0
  fi
}

detect_node_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}

resolve_host_ipv4() {
  local host=$1
  if [[ -z "${host}" ]]; then
    return 0
  fi
  python3 - "$host" <<'PY'
import re
import socket
import sys

host = sys.argv[1]
match = re.match(r"^(\d+)-(\d+)-(\d+)-(\d+)(?:\.|$)", host)
if match:
    print(".".join(match.groups()))
    raise SystemExit(0)

try:
    infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
except socket.gaierror:
    print(host)
    raise SystemExit(0)

seen = set()
for info in infos:
    ip = info[4][0]
    if ip not in seen:
        print(ip)
        break
    seen.add(ip)
else:
    print(host)
PY
}

NODE_RANK=$(detect_node_rank)
if [[ "${RAY_NODE_ROLE}" == "head" || "${RAY_NODE_ROLE}" == "master" ]]; then
  NODE_RANK=0
elif [[ "${RAY_NODE_ROLE}" == "worker" && "${NODE_RANK}" == "0" ]]; then
  echo "[warn] RAY_NODE_ROLE=worker but detected NODE_RANK=0. Set NODE_RANK/RAY_NODE_RANK explicitly if this is not worker-0." >&2
fi
RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS:-$(detect_node_ip)}
if [[ -z "${RAY_NODE_IP_ADDRESS}" ]]; then
  RAY_NODE_IP_ADDRESS=$(hostname -f)
fi

# Most long-running async jobs deliberately keep Ray worker pods alive while the
# rank-0 launcher owns the process lifetime. Offline probes are different: once
# rank 0 has finished scoring, those workers must exit so the platform can mark
# the job complete. Keep this opt-in because regular training wrappers may have
# post-launch work on rank 0.
RAY_WORKER_EXIT_ON_MASTER_DONE=${RAY_WORKER_EXIT_ON_MASTER_DONE:-0}
RAY_WORKER_EXIT_MARKER=${RAY_WORKER_EXIT_MARKER:-${LOG_DIR}/.ray_master_done_$(printf '%s' "${RAY_PORT_SEED}" | tr -c 'A-Za-z0-9_.-' '_').txt}
write_ray_worker_exit_marker() {
  local rc=$?
  if [[ "${NODE_RANK}" == "0" && "${RAY_WORKER_EXIT_ON_MASTER_DONE}" == "1" ]]; then
    mkdir -p "$(dirname "${RAY_WORKER_EXIT_MARKER}")"
    printf '%s\n' "${rc}" > "${RAY_WORKER_EXIT_MARKER}"
    echo "[info] Wrote Ray worker completion marker ${RAY_WORKER_EXIT_MARKER} (rc=${rc})"
  fi
}
if [[ "${RAY_WORKER_EXIT_ON_MASTER_DONE}" == "1" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    rm -f "${RAY_WORKER_EXIT_MARKER}"
    trap write_ray_worker_exit_marker EXIT
  fi
fi

RAY_HEAD_HOST=${RAY_HEAD_HOST:-${MASTER_ADDR:-${VC_MASTER_HOSTS:-${MASTER_HOST:-${TORCH_MASTER_ADDR:-}}}}}
if [[ "${RAY_HEAD_HOST}" == *,* ]]; then
  RAY_HEAD_HOST=${RAY_HEAD_HOST%%,*}
fi
if [[ -z "${RAY_HEAD_HOST}" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    RAY_HEAD_HOST="${RAY_NODE_IP_ADDRESS}"
  else
    echo "[error] RAY_HEAD_HOST/MASTER_ADDR is required on worker node rank ${NODE_RANK}" >&2
    exit 1
  fi
fi
RAY_HEAD_HOST=$(resolve_host_ipv4 "${RAY_HEAD_HOST}")
RAY_HEAD_PORT_FILE=${RAY_HEAD_PORT_FILE:-${LOG_DIR}/.ray_head_port_$(printf '%s' "${RAY_PORT_SEED}" | tr -c 'A-Za-z0-9_.-' '_').txt}
if [[ "${RAY_AUTOSTART}" == "1" && -z "${RAY_PORT_WAS_SET}" && "${RAY_HEAD_PORT_NEGOTIATE}" == "1" ]]; then
  negotiate_ray_head_port "${RAY_HEAD_PORT_FILE}" "${RAY_HEAD_PORT_FILE_WAIT_SECONDS}"
  require_port RAY_PORT
  ensure_port_outside_worker_range RAY_PORT
elif [[ "${RAY_AUTOSTART}" == "1" && "${NODE_RANK}" == "0" ]]; then
  mkdir -p "$(dirname "${RAY_HEAD_PORT_FILE}")"
  printf '%s\n' "${RAY_PORT}" > "${RAY_HEAD_PORT_FILE}"
fi
export RAY_ADDRESS=${RAY_ADDRESS:-${RAY_HEAD_HOST}:${RAY_PORT}}

start_ray_cluster() {
  local attempt failed_port
  if [[ "${RAY_AUTOSTART}" != "1" ]]; then
    echo "[info] RAY_AUTOSTART=0, using existing RAY_ADDRESS=${RAY_ADDRESS}"
    return
  fi

  ray stop --force >/dev/null 2>&1 || true
  rm -rf "${RAY_TMPDIR}" || true
  mkdir -p "${RAY_TMPDIR}"

  if [[ "${NODE_RANK}" == "0" ]]; then
    require_positive_int RAY_HEAD_START_ATTEMPTS
    for ((attempt = 1; attempt <= RAY_HEAD_START_ATTEMPTS; attempt++)); do
      echo "[info] Starting Ray head at ${RAY_NODE_IP_ADDRESS}:${RAY_PORT} (attempt ${attempt}/${RAY_HEAD_START_ATTEMPTS})"
      ray_head_args=(
        --head
        --node-ip-address="${RAY_NODE_IP_ADDRESS}" \
        --port="${RAY_PORT}" \
        --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
        --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
        --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
        --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
        --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
        --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
        --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
        --min-worker-port="${RAY_MIN_WORKER_PORT}" \
        --max-worker-port="${RAY_MAX_WORKER_PORT}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --num-cpus="${RAY_NODE_CPUS}" \
        --temp-dir="${RAY_TMPDIR}"
      )
      if is_true "${RAY_INCLUDE_DASHBOARD}"; then
        ray_head_args+=(--include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}")
      else
        ray_head_args+=(--include-dashboard=false)
      fi
      if ray start "${ray_head_args[@]}"; then
        return 0
      fi
      failed_port=${RAY_PORT}
      ray stop --force >/dev/null 2>&1 || true
      rm -rf "${RAY_TMPDIR}" || true
      mkdir -p "${RAY_TMPDIR}"
      if (( attempt == RAY_HEAD_START_ATTEMPTS )); then
        fail "Ray head failed to start after ${RAY_HEAD_START_ATTEMPTS} attempts; last port=${failed_port}"
      fi
      select_next_ray_head_port "${RAY_HEAD_PORT_FILE}" "${failed_port}"
    done
  else
    echo "[info] Starting Ray worker rank ${NODE_RANK}, joining ${RAY_ADDRESS}"
    command -v timeout >/dev/null 2>&1 || fail "timeout is required for bounded Ray worker join retries"
    until refresh_ray_head_address_from_file "${RAY_HEAD_PORT_FILE}" && timeout \
        --signal=TERM \
        --kill-after=10 \
        "${RAY_WORKER_JOIN_ATTEMPT_TIMEOUT}" \
        ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address="${RAY_NODE_IP_ADDRESS}" \
        --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
        --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
        --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
        --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
        --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
        --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
        --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
        --min-worker-port="${RAY_MIN_WORKER_PORT}" \
        --max-worker-port="${RAY_MAX_WORKER_PORT}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --num-cpus="${RAY_NODE_CPUS}" \
        --temp-dir="${RAY_TMPDIR}"; do
      echo "[warn] Ray worker join failed or exceeded ${RAY_WORKER_JOIN_ATTEMPT_TIMEOUT}s; refreshing negotiated head port and retrying in 10s"
      ray stop --force >/dev/null 2>&1 || true
      rm -rf "${RAY_TMPDIR}" || true
      mkdir -p "${RAY_TMPDIR}"
      sleep 10
    done
    if [[ "${RAY_WORKER_EXIT_ON_MASTER_DONE}" == "1" ]]; then
      echo "[info] Worker rank ${NODE_RANK} joined Ray. Waiting for master completion marker ${RAY_WORKER_EXIT_MARKER}."
      until [[ -s "${RAY_WORKER_EXIT_MARKER}" ]]; do
        sleep 5
      done
      read -r master_rc < "${RAY_WORKER_EXIT_MARKER}"
      echo "[info] Worker rank ${NODE_RANK} observed master completion (rc=${master_rc}); exiting."
      exit "${master_rc}"
    fi
    echo "[info] Worker rank ${NODE_RANK} joined Ray. Blocking here; training runs on rank 0."
    tail -f /dev/null
  fi
}

wait_for_ray_resources() {
  python3 -u - <<PY
import time
from ray._raylet import GcsClient
from ray.core.generated.gcs_pb2 import GcsNodeInfo

address = "${RAY_ADDRESS}"
expected_gpus = int("${NNODES}") * int("${GPUS_PER_NODE}")
expected_nodes = int("${NNODES}")
timeout = int("${RAY_START_WAIT_SECONDS}")
print(f"[info] Querying Ray GCS at {address} without starting a Ray driver", flush=True)
gcs_client = GcsClient(address=address)
start = time.time()
while True:
    node_infos = list(gcs_client.get_all_node_info(timeout=5).values())
    alive_nodes = [
        node for node in node_infos
        if node.state == GcsNodeInfo.GcsNodeState.ALIVE
    ]
    node_resources = [dict(node.resources_total) for node in alive_nodes]
    gpus = int(sum(resources.get("GPU", 0) for resources in node_resources))
    nodes = len(alive_nodes)
    print(
        f"[info] Ray resources: GPU={gpus}/{expected_gpus}, nodes={nodes}, "
        f"node_resources={node_resources}",
        flush=True,
    )
    print(
        "[info] Ray nodes: "
        + "; ".join(
            f"{node.node_manager_address} alive=True "
            f"resources={dict(node.resources_total)}"
            for node in alive_nodes
        ),
        flush=True,
    )
    if nodes >= expected_nodes and gpus >= expected_gpus:
        break
    if time.time() - start > timeout:
        raise SystemExit(
            "Timed out waiting for Ray resources: "
            f"GPU={gpus}/{expected_gpus}, nodes={nodes}/{expected_nodes}"
        )
    time.sleep(10)
PY
}

echo "[info] Qwen3-Omni Megatron + vLLM-Omni ${PROFILE_LABEL}"
echo "[info] CONDA_ENV=${CONDA_ENV}"
echo "[info] REPO_ROOT=${REPO_ROOT}"
echo "[info] VERL_ROOT=${VERL_ROOT}"
echo "[info] MEGATRON_BRIDGE_REPO=${MEGATRON_BRIDGE_REPO}"
echo "[info] VLLM_OMNI_ROOT=${VLLM_OMNI_ROOT}"
echo "[info] MODEL_PATH=${MODEL_PATH}"
echo "[info] TRAIN_FILES=${TRAIN_FILES}"
echo "[info] VAL_FILES=${VAL_FILES}"
echo "[info] STAGE_CONFIG=${STAGE_CONFIG}"
echo "[info] Ray world: NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_GPUS=${WORLD_GPUS}"
echo "[info] Resource split: train=${TRAIN_NNODES}x${TRAIN_GPUS_PER_NODE} (${TRAIN_WORLD_GPUS}), rollout=${ROLLOUT_NNODES}x${ROLLOUT_GPUS_PER_NODE} (${ROLLOUT_WORLD_GPUS})"
echo "[info] Split contract: force_standalone=${VERL_OMNI_FORCE_STANDALONE_ROLLOUT} impl=${VERL_OMNI_RESOURCE_SPLIT_IMPL} train_gpn=${VERL_OMNI_TRAIN_GPUS_PER_NODE} rollout_gpn=${VERL_OMNI_ROLLOUT_GPUS_PER_NODE} rollout_nnodes=${VERL_OMNI_ROLLOUT_NNODES}"
echo "[info] TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS} REQUIRE_BATCHES=${REQUIRE_BATCHES}"
echo "[info] actor tp/pp/cp/ep/etp=${ACTOR_TP}/${ACTOR_PP}/${ACTOR_CP}/${ACTOR_EP}/${ACTOR_ETP}"
echo "[info] ref tp/pp/cp/ep/etp=${REF_TP}/${REF_PP}/${REF_CP}/${REF_EP}/${REF_ETP}"
echo "[info] rollout tp=${ROLLOUT_TP} dp=${ROLLOUT_DP} replicas=$((ROLLOUT_WORLD_GPUS / (ROLLOUT_TP * ROLLOUT_DP))) n=${N_RESP_PER_PROMPT} corr_bypass=${ROLLOUT_CORR_BYPASS_MODE}"
echo "[info] vLLM-Omni LB policy: ${OMNI_LB_POLICY}"
echo "[info] rollout logprobs: calculate=${ROLLOUT_CALCULATE_LOG_PROBS} mode=${ROLLOUT_LOGPROBS_MODE} require_raw=${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS} require_processed=${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS} temperature=${ROLLOUT_TEMPERATURE} top_p=${ROLLOUT_TOP_P} top_k=${ROLLOUT_TOP_K}"
echo "[info] sequence_parallel=${SEQUENCE_PARALLEL} attention_backend=${ATTENTION_BACKEND} moe_dispatcher=${MOE_TOKEN_DISPATCHER_TYPE}"
echo "[info] rollout free_cache_engine=False weight_update_skip=${VERL_OMNI_SKIP_WEIGHT_UPDATE} phase_generation_debug=${VERL_OMNI_PHASE_GENERATION_DEBUG}"
echo "[info] HOSTNAME=$(hostname -f 2>/dev/null || hostname)"
echo "[info] NODE_RANK=${NODE_RANK} RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS}"
echo "[info] RAY_AUTOSTART=${RAY_AUTOSTART} RAY_ADDRESS=${RAY_ADDRESS} CONFIG_ONLY=${CONFIG_ONLY} RAY_ONLY=${RAY_ONLY}"
echo "[info] Ray ports: head=${RAY_PORT} head_pool=${RAY_HEAD_PORT_POOL_START}-${RAY_HEAD_PORT_POOL_END} head_negotiation=${RAY_HEAD_PORT_NEGOTIATE} head_file=${RAY_HEAD_PORT_FILE} seed=${RAY_PORT_SEED} dashboard_enabled=${RAY_INCLUDE_DASHBOARD} dashboard=${RAY_DASHBOARD_PORT} worker_range=${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT} worker_pool=${RAY_WORKER_PORT_POOL_START}-${RAY_WORKER_PORT_POOL_END} worker_seed=${RAY_WORKER_PORT_SEED} worker_span=${RAY_WORKER_PORT_SPAN} worker_min_free=${RAY_WORKER_PORT_MIN_FREE} worker_preferred_block=${RAY_WORKER_PORT_PREFERRED_BLOCK_INDEX} worker_block=${RAY_WORKER_PORT_BLOCK_INDEX} worker_node_rank_hint=${RAY_WORKER_PORT_NODE_RANK_HINT} worker_local_probe=${RAY_WORKER_PORT_LOCAL_PROBE}"
echo "[info] Ray component ports: pool=${RAY_COMPONENT_PORT_POOL_START}-${RAY_COMPONENT_PORT_POOL_END} span=${RAY_COMPONENT_PORT_SPAN} preferred_block=${RAY_COMPONENT_PORT_PREFERRED_BLOCK_INDEX} block=${RAY_COMPONENT_PORT_BLOCK_INDEX} base=${RAY_COMPONENT_PORT_BASE} node_manager=${RAY_NODE_MANAGER_PORT} object_manager=${RAY_OBJECT_MANAGER_PORT} dashboard_agent_http=${RAY_DASHBOARD_AGENT_LISTEN_PORT} dashboard_agent_grpc=${RAY_DASHBOARD_AGENT_GRPC_PORT} runtime_env_agent=${RAY_RUNTIME_ENV_AGENT_PORT} metrics=${RAY_METRICS_EXPORT_PORT} client=${RAY_CLIENT_SERVER_PORT} local_probe=${RAY_COMPONENT_PORT_LOCAL_PROBE}"
echo "[info] vLLM internal port pool: seed=${VERL_OMNI_VLLM_PORT_SEED} pool=${VERL_OMNI_VLLM_PORT_POOL_START}-${VERL_OMNI_VLLM_PORT_POOL_END} stride=${VERL_OMNI_VLLM_PORT_STRIDE} stage_core_offset=${VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET} stage_core_guard=${VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD} stage_core_spread=${VERL_OMNI_VLLM_STAGE_CORE_PORT_SPREAD} stage_core_min_tail=${VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL} stage_core_direct_gap=${VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP} actor_slots=${VERL_OMNI_VLLM_PORT_ACTOR_SLOTS}"
echo "[info] vLLM existing VLLM_PORT policy: respect_existing=${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT}"
echo "[info] vLLM stage-core TCPStore first-port override: use_master_port=${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE} worker_master_pool=${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START}-${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END}"
if (( ROLLOUT_DP > 1 )); then
  echo "[info] vLLM distributed rendezvous port override: port=${VERL_OMNI_VLLM_DIST_MASTER_PORT} pool=${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_START}-${VERL_OMNI_VLLM_DIST_MASTER_PORT_POOL_END}; addr is owned by rollout server[0]"
else
  echo "[info] vLLM distributed rendezvous inactive: rollout_dp=${ROLLOUT_DP}; using standalone TP${ROLLOUT_TP} replicas=${ROLLOUT_REPLICAS_FOR_VLLM_PORT}"
fi
echo "[info] Omni master ZMQ port pool: base=${VERL_OMNI_MASTER_ZMQ_PORT_BASE} pool=${VERL_OMNI_MASTER_ZMQ_PORT_POOL_START}-${VERL_OMNI_MASTER_ZMQ_PORT_POOL_END} span=${VERL_OMNI_MASTER_ZMQ_PORT_SPAN}"
echo "[info] vLLM-Omni timeouts: stage_init=${STAGE_INIT_TIMEOUT}s init=${INIT_TIMEOUT}s vllm_startup_handshake=${VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT}s"
echo "[info] Stage core crash diagnostics: ${VERL_OMNI_STAGE_CORE_DIAG_DIR}"
echo "[info] LOG_FILE=${LOG_FILE}"
echo "[info] TENSORBOARD_DIR=${TENSORBOARD_DIR}"

for env_name in MASTER_ADDR MASTER_PORT VC_MASTER_HOSTS MASTER_HOST TORCH_MASTER_ADDR \
  DISTRIBUTED_PYTORCH_PORT LUBAN_AVAILABLE_PORT_0 LUBAN_AVAILABLE_PORT_1 \
  DISTRIBUTED_TASK_ROLE VC_TASK_ROLE ROLE_NAME TASK_ROLE TASK_INDEX POD_NAME K8S_POD_NAME \
  DISTRIBUTED_NODE_RANK NODE_RANK RAY_NODE_RANK RAY_NODE_ROLE GROUP_RANK ROLE_INDEX \
  POD_INDEX INDEX VC_TASK_INDEX VK_TASK_INDEX MATRIX_TASK_INDEX AIP_WORKER_INDEX RANK LOCAL_RANK WORLD_SIZE; do
  echo "[info] env ${env_name}=${!env_name:-}"
done

persist_run_config_snapshot "$@"

if [[ "${CONFIG_ONLY}" == "1" ]]; then
  echo "[info] CONFIG_ONLY=1, static configuration preflight passed. Stop before Ray startup."
  exit 0
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  ln -sfn "${LOG_FILE}" "${LOG_DIR}/async_full_32gpu.latest.log"
fi

start_ray_cluster
wait_for_ray_resources

if [[ "${RAY_ONLY}" == "1" ]]; then
  echo "[info] RAY_ONLY=1, Ray resource preflight passed. Stop before model/training launch."
  exit 0
fi

set +e
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name=fully_async_ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode="${ROLLOUT_CORR_BYPASS_MODE}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.dataloader_num_workers=0 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.gen_batch_size=1 \
    data.train_batch_size=0 \
    data.val_max_samples=2 \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora.rank=0 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.lr_decay_steps="${TOTAL_TRAINING_STEPS}" \
    actor_rollout_ref.actor.optim.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP}" \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ACTOR_ETP}" \
    actor_rollout_ref.actor.megatron.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion="${MASKED_SOFTMAX_FUSION}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion="${MOE_PERMUTE_FUSION}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="${MOE_TOKEN_DISPATCHER_TYPE}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion="${GRADIENT_ACCUMULATION_FUSION}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend="${ATTENTION_BACKEND}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=24 \
    actor_rollout_ref.actor.megatron.param_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.megatron.optimizer_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.megatron.grad_offload="${OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.ignore_eos="${IGNORE_EOS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}" \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.calculate_log_probs="${ROLLOUT_CALCULATE_LOG_PROBS}" \
    actor_rollout_ref.rollout.logprobs_mode="${ROLLOUT_LOGPROBS_MODE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout="${STAGE_INIT_TIMEOUT}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout="${INIT_TIMEOUT}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.omni_lb_policy="${OMNI_LB_POLICY}" \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
    actor_rollout_ref.ref.megatron.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${REF_TP}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${REF_PP}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${REF_CP}" \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size="${REF_EP}" \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${REF_ETP}" \
    actor_rollout_ref.ref.megatron.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.masked_softmax_fusion="${MASKED_SOFTMAX_FUSION}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.moe_permute_fusion="${MOE_PERMUTE_FUSION}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.moe_token_dispatcher_type="${MOE_TOKEN_DISPATCHER_TYPE}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion="${GRADIENT_ACCUMULATION_FUSION}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend="${ATTENTION_BACKEND}" \
    actor_rollout_ref.ref.megatron.param_offload="${OFFLOAD}" \
    reward.num_workers=1 \
    reward.reward_manager.name=naive \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${TRAIN_GPUS_PER_NODE}" \
    trainer.nnodes="${TRAIN_NNODES}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    rollout.nnodes="${ROLLOUT_NNODES}" \
    rollout.n_gpus_per_node="${ROLLOUT_GPUS_PER_NODE}" \
    rollout.n="${N_RESP_PER_PROMPT}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    async_training.staleness_threshold=0 \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout=True \
    async_training.use_trainer_do_validate=False \
    ++ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\"${PYTHONPATH}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONNOUSERSITE=\"${PYTHONNOUSERSITE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"${VLLM_USE_V1}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DISABLE_COMPILE_CACHE=\"${VLLM_DISABLE_COMPILE_CACHE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.RAY_MIN_WORKER_PORT=\"${RAY_MIN_WORKER_PORT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.RAY_MAX_WORKER_PORT=\"${RAY_MAX_WORKER_PORT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=\"${VERL_USE_EXTERNAL_MODULES}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_FORCE_STANDALONE_ROLLOUT=\"${VERL_OMNI_FORCE_STANDALONE_ROLLOUT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_RESOURCE_SPLIT_IMPL=\"${VERL_OMNI_RESOURCE_SPLIT_IMPL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_TRAIN_GPUS_PER_NODE=\"${VERL_OMNI_TRAIN_GPUS_PER_NODE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_ROLLOUT_GPUS_PER_NODE=\"${VERL_OMNI_ROLLOUT_GPUS_PER_NODE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_ROLLOUT_NNODES=\"${VERL_OMNI_ROLLOUT_NNODES}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS=\"${VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_QWEN3_OMNI_SP_SCATTER_POSITION_IDS=\"${VERL_OMNI_QWEN3_OMNI_SP_SCATTER_POSITION_IDS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL=\"${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT=\"${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS=\"${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS=\"${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT=\"${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_LOGPROB_COMPONENT_AUDIT=\"${VERL_OMNI_MEGATRON_LOGPROB_COMPONENT_AUDIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT=\"${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS=\"${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT=\"${VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT_LAYERS=\"${VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT_LAYERS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MEGATRON_MOE_REPLAY_METADATA_AUDIT=\"${VERL_OMNI_MEGATRON_MOE_REPLAY_METADATA_AUDIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE=\"${VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS=\"${VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE_LAYERS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_PORT_SEED=\"${VERL_OMNI_VLLM_PORT_SEED}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_PORT_POOL_START=\"${VERL_OMNI_VLLM_PORT_POOL_START}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_PORT_POOL_END=\"${VERL_OMNI_VLLM_PORT_POOL_END}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_PORT_STRIDE=\"${VERL_OMNI_VLLM_PORT_STRIDE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET=\"${VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD=\"${VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_STAGE_CORE_PORT_SPREAD=\"${VERL_OMNI_VLLM_STAGE_CORE_PORT_SPREAD}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL=\"${VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP=\"${VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_RESPECT_EXISTING_VLLM_PORT=\"${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE=\"${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START=\"${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END=\"${VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_PORT_ACTOR_SLOTS=\"${VERL_OMNI_VLLM_PORT_ACTOR_SLOTS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MASTER_ZMQ_PORT_BASE=\"${VERL_OMNI_MASTER_ZMQ_PORT_BASE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_DIST_MASTER_PORT=\"${VERL_OMNI_VLLM_DIST_MASTER_PORT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=\"${VERL_LOGGING_LEVEL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_PPO_LOGGING_LEVEL=\"${VERL_PPO_LOGGING_LEVEL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=\"${TENSORBOARD_DIR}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_MODELS=\"${VERL_OMNI_SKIP_MODELS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_PIPELINES=\"${VERL_OMNI_SKIP_PIPELINES}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_REWARD_LOOP=\"${VERL_OMNI_SKIP_REWARD_LOOP}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_TRAINER=\"${VERL_OMNI_SKIP_TRAINER}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_ENGINES=\"${VERL_OMNI_SKIP_ENGINES}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_WEIGHT_UPDATE=\"${VERL_OMNI_SKIP_WEIGHT_UPDATE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS=\"${VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME=\"${VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP=\"${VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_WAKE_TAGS=\"${VERL_OMNI_WAKE_TAGS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_SLEEP_LEVEL=\"${VERL_OMNI_SLEEP_LEVEL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_WEIGHT_SYNC_DEBUG=\"${VERL_OMNI_WEIGHT_SYNC_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_WEIGHT_SYNC_TOPO_DEBUG=\"${VERL_OMNI_WEIGHT_SYNC_TOPO_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_WEIGHT_ROUTE_DEBUG=\"${VERL_OMNI_WEIGHT_ROUTE_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_WEIGHT_VALUE_DEBUG=\"${VERL_OMNI_WEIGHT_VALUE_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PRE_IPC_LOCAL_COPY_DEBUG=\"${VERL_OMNI_PRE_IPC_LOCAL_COPY_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PRE_LOAD_LOCAL_COPY_DEBUG=\"${VERL_OMNI_PRE_LOAD_LOCAL_COPY_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_WAKE_CUMEM_DEBUG=\"${VERL_OMNI_WAKE_CUMEM_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_DEBUG=\"${VERL_OMNI_PHASE_GENERATION_DEBUG}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_STRICT=\"${VERL_OMNI_PHASE_GENERATION_STRICT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_PHASES=\"${VERL_OMNI_PHASE_GENERATION_PHASES}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_MAX_TOKENS=\"${VERL_OMNI_PHASE_GENERATION_MAX_TOKENS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_MIN_TOKENS=\"${VERL_OMNI_PHASE_GENERATION_MIN_TOKENS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_TEMPERATURE=\"${VERL_OMNI_PHASE_GENERATION_TEMPERATURE}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_TOP_P=\"${VERL_OMNI_PHASE_GENERATION_TOP_P}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_PHASE_GENERATION_IGNORE_EOS=\"${VERL_OMNI_PHASE_GENERATION_IGNORE_EOS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_GENERATION_DEBUG_LIMIT=\"${VERL_OMNI_GENERATION_DEBUG_LIMIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_LOGPROB_DEBUG_LIMIT=\"${VERL_OMNI_LOGPROB_DEBUG_LIMIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT=\"${VERL_OMNI_LOGPROB_PARITY_DEBUG_LIMIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL=\"${VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS=\"${VERL_OMNI_VLLM_LOGPROB_DEBUG_PER_PROCESS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT=\"${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL=\"${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS=\"${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL=\"${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS=\"${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS=\"${VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL=\"${VERL_OMNI_IMAGE_BINDING_AUDIT_JSONL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT=\"${VERL_OMNI_IMAGE_BINDING_AUDIT_LIMIT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=\"${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=\"${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=\"${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=\"${VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=\"${VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS=\"${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS=\"${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_STAGE_CORE_DIAG_DIR=\"${VERL_OMNI_STAGE_CORE_DIAG_DIR}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FORCE_SHM_WEIGHT_TRANSFER=\"${VERL_FORCE_SHM_WEIGHT_TRANSFER}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=\"${CUDA_DEVICE_MAX_CONNECTIONS}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=\"${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.HYDRA_FULL_ERROR=\"${HYDRA_FULL_ERROR}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.NVTE_DEBUG=\"${NVTE_DEBUG:-0}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.NVTE_DEBUG_LEVEL=\"${NVTE_DEBUG_LEVEL:-0}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.NVTE_PRINT_RANK=\"${NVTE_PRINT_RANK:-0}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.NVTE_ALLOW_NONDETERMINISTIC_ALGO=\"${NVTE_ALLOW_NONDETERMINISTIC_ALGO}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME=\"${CUDA_HOME}\" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" \
    "$@"
python_rc=$?
set -e
if (( python_rc != 0 )); then
  echo "[error] fully_async_main exited with rc=${python_rc}" >&2
  exit "${python_rc}"
fi
echo "[info] fully_async_main completed successfully" | tee -a "${LOG_FILE}"
