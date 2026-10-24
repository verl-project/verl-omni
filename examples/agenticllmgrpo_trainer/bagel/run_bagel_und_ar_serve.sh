#!/usr/bin/env bash
# Standalone Bagel UND AR serve for the Hermes spike (fail-closed GPU preflight).
#
# Default: physical GPUs 0-3 -> logical 0=Thinker, 1=DiT, the same cards the trainer
# uses. Stop this server before training — the colocated UND AR shares GPU 0.
# Then prove Hermes: BAGEL_UND_URL=http://127.0.0.1:8094 python3 .../spike_und_hermes.py
# Do not point the spike at bagel_single_stage / GEN. Do not use Qwen for UND.
set -euo pipefail
# Shell tracing is opt-in: ``DEBUG=1`` adds the ``+ ...`` lines, a normal launch stays quiet.
if [[ "${DEBUG:-}" == "1" ]]; then
  set -x
fi

_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
_PY="${_REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${_PY}" ]]; then
  _PY="$(command -v python3)"
fi
# Use this repo's vllm-omni, not whichever checkout's venv comes first on PATH.
_VLLM_OMNI="${_REPO_ROOT}/.venv/bin/vllm-omni"
if [[ ! -x "${_VLLM_OMNI}" ]]; then
  _VLLM_OMNI="$(command -v vllm-omni)"
fi

# flashinfer JIT needs nvcc; this box's toolkits live under /cm/shared, not /usr/local/cuda.
# It also has to be a *recent* nvcc, which a first-found-wins loop gets wrong: flashinfer
# 0.6.18 builds with ``--compress-mode=size`` (nvcc 12.8+) and pointing at 12.4 aborts with
#   nvcc fatal   : Unknown option '--compress-mode=size'
#   ninja: build stopped: subcommand failed.
#   RuntimeError: Engine core initialization failed
# ``/cm/shared/apps/cuda-latest`` is stale (``current`` -> 12.4.1) while 13.0 -- the toolkit
# this venv's ``torch 2.13.0+cu130`` is built for -- sits at ``cuda13.0/toolkit/13.0`` with no
# ``current`` symlink. Keep this in sync with ``run_agentic_bagel_rpco_lora.sh``: enumerate
# every installed toolkit and keep the newest nvcc that accepts the flag flashinfer passes.
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

# flashinfer's JIT shell-outs to ``ninja`` to build its sampling kernels. The trainer
# recipe gets this from the repo venv; without it the engine dies deep in warm-up with
# ``FileNotFoundError: 'ninja'`` before the port ever opens.
if [[ ":${PATH}:" != *":${_REPO_ROOT}/.venv/bin:"* ]]; then
  export PATH="${_REPO_ROOT}/.venv/bin:${PATH}"
fi

# The node driver tops out at CUDA 12.2 while this venv's torch is cu13, so a bare shell
# dies before the port opens with:
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12020)
# from ``torch.cuda.get_device_name()`` during ``vllm_omni.diffusion.envs`` import (which
# runs at ``--help`` time, not at engine init). ``run_agentic_bagel_rpco_lora.sh`` already
# carries the fix and it is why the trainer boots on these cards: prepend the cuda-compat
# shim (a real forward-compat libcuda, so cu13 userspace runs on the 535 kernel module) and
# the cu13 runtime, and let torch answer ``is_available()`` from NVML. Keep these in sync
# with the recipe -- this script is the gate for the same engine.
_cuda_compat="${_REPO_ROOT}/.venv/cuda-compat"
_cu13_lib="${_REPO_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "${_cuda_compat}" ]]; then
  export LD_LIBRARY_PATH="${_cuda_compat}:${_cu13_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
fi

export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}"
# Match run_agentic_bagel_rpco_lora.sh's 0,1,2,3 default; honour the caller, never hard-assign.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8094}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
KV_ARENA_RATIO="${KV_ARENA_RATIO:-0.35}"
# Refuse to start unless the first card is at least this % free (set 0 to co-tenant).
GPU_MIN_FREE_PCT="${GPU_MIN_FREE_PCT:-80}"
DEPLOY_CONFIG="${BAGEL_UND_DEPLOY_CONFIG:-${_SCRIPT_DIR}/bagel_corl_deploy_ar.yaml}"

_FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
_GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${_FIRST_GPU}" | head -1 | tr -d '[:space:]')"
_GPU_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${_FIRST_GPU}" | head -1 | tr -d '[:space:]')"
if [[ -z "${_GPU_TOTAL_MIB}" || ! "${_GPU_TOTAL_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU ${_FIRST_GPU} memory.total via nvidia-smi." >&2
  exit 1
fi
if [[ -z "${_GPU_FREE_MIB}" || ! "${_GPU_FREE_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU ${_FIRST_GPU} memory.free via nvidia-smi." >&2
  exit 1
fi

# Preflight: the KV arena below is a fixed byte reservation, so a card still owned by a
# previous engine would OOM deep inside vLLM. Fail here instead and name the squatter.
_GPU_USED_MIB=$(( _GPU_TOTAL_MIB - _GPU_FREE_MIB ))
_GPU_FREE_PCT=$(( _GPU_FREE_MIB * 100 / _GPU_TOTAL_MIB ))
if (( _GPU_FREE_PCT < GPU_MIN_FREE_PCT )); then
  echo "[ERROR] GPU ${_FIRST_GPU} is already occupied: ${_GPU_FREE_MIB} MiB free of ${_GPU_TOTAL_MIB} MiB (${_GPU_FREE_PCT}%, need >= ${GPU_MIN_FREE_PCT}%)." >&2
  echo "        ${_GPU_USED_MIB} MiB is held by:" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${_FIRST_GPU}" 2>/dev/null | sed 's/^/          /' >&2 || true
  echo "        A leftover 'vllm-omni serve' from an earlier run is the usual cause. Free it and retry:" >&2
  echo "          pkill -f 'vllm-omni serve'" >&2
  echo "        Or override the guard with GPU_MIN_FREE_PCT=0 if you really intend to co-tenant." >&2
  exit 1
fi

# Size the KV arena from free (not total) memory so a shared card cannot be overcommitted.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-$("${_PY}" -c "print(int(${_GPU_FREE_MIB} * 1024 * 1024 * float('${KV_ARENA_RATIO}')))")}"

# Two visible cards → split Thinker / DiT. One card → both on logical 0.
_N_VISIBLE=1
IFS=',' read -ra _ids <<< "${CUDA_VISIBLE_DEVICES}"
_N_VISIBLE="${#_ids[@]}"
if [[ -z "${STAGE_OVERRIDES:-}" ]]; then
  if (( _N_VISIBLE >= 2 )); then
    STAGE_OVERRIDES="$("${_PY}" -c "
import json
print(json.dumps({
  '0': {
    'devices': '0',
    'gpu_memory_utilization': float('${GPU_MEM_UTIL}'),
    'kv_cache_memory_bytes': int('${KV_CACHE_MEMORY_BYTES}'),
  },
  '1': {'devices': '1'},
}))
")"
  else
    STAGE_OVERRIDES="$("${_PY}" -c "
import json
print(json.dumps({
  '0': {
    'devices': '0',
    'gpu_memory_utilization': float('${GPU_MEM_UTIL}'),
    'kv_cache_memory_bytes': int('${KV_CACHE_MEMORY_BYTES}'),
  },
  '1': {'devices': '0'},
}))
")"
  fi
fi

echo "[INFO] UND AR serve model=${BAGEL_MODEL_PATH}"
echo "[INFO] deploy=${DEPLOY_CONFIG} host=${HOST} port=${PORT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} STAGE_OVERRIDES=${STAGE_OVERRIDES}"
echo "[INFO] spike: BAGEL_UND_URL=http://${HOST}:${PORT} python3 ${_SCRIPT_DIR}/spike_und_hermes.py --model-path \"\$BAGEL_MODEL_PATH\""

exec "${_VLLM_OMNI}" serve "${BAGEL_MODEL_PATH}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --deploy-config "${DEPLOY_CONFIG}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --stage-overrides "${STAGE_OVERRIDES}" \
  "$@"
