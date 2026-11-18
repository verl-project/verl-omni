#!/usr/bin/env bash
# Shared setup for the model-specific NPU e2e launchers.

NPU_E2E_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${NPU_E2E_REPO_ROOT}"

NUM_NPUS=${NUM_NPUS:-8}
if [[ ! "${NUM_NPUS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_NPUS < 2 || NUM_NPUS % 2 != 0 )); then
    echo "NUM_NPUS must be an even integer of at least 2." >&2
    exit 2
fi
export NUM_NPUS

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES=0
    for ((npu_index = 1; npu_index < NUM_NPUS; npu_index++)); do
        ASCEND_RT_VISIBLE_DEVICES+=",${npu_index}"
    done
fi
export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export DEVICE_NAME=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES-1}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHNPU_PRECOMPILE_THREADS="${TORCHNPU_PRECOMPILE_THREADS:-2}"
if [[ ! "${TORCHNPU_PRECOMPILE_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TORCHNPU_PRECOMPILE_THREADS must be a positive integer." >&2
    exit 2
fi
# vllm-ascend rejects FRACTAL_NZ weights on wake_up in RL scenarios; keep NZ
# off for every NPU e2e entry (same guard the Qwen3-Omni launcher uses).
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"

set +u
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

python3 - <<'PY'
import os

expected = int(os.environ["NUM_NPUS"])
for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
    devices = os.environ[name].split(",")
    runtime = name == "ASCEND_RT_VISIBLE_DEVICES"
    valid_size = len(devices) == expected if runtime else len(devices) >= expected
    if (
        not valid_size
        or any(not device.isdecimal() for device in devices)
        or len({int(device) for device in devices}) != len(devices)
    ):
        count = "exactly" if runtime else "at least"
        raise SystemExit(f"{name} must list {count} {expected} distinct NPU IDs: {devices}")

import torch
import torch_npu  # noqa: F401

if not torch.npu.is_available() or torch.npu.device_count() != expected:
    raise SystemExit(f"Expected {expected} visible NPUs, found {torch.npu.device_count()}.")
print(f"NPU e2e: {expected} devices; torch {torch.__version__}; torch-npu {torch_npu.__version__}")
PY

export ATTN_BACKEND="${ATTN_BACKEND:-native}"
export ROLLOUT_ATTN_BACKEND="${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}"

NPU_RUNTIME_ARGS_TEXT=$(python3 "${NPU_E2E_REPO_ROOT}/tests/special_e2e/npu_runtime.py")
NPU_RUNTIME_ARGS=()
while IFS= read -r override; do
    NPU_RUNTIME_ARGS+=("${override}")
done <<< "${NPU_RUNTIME_ARGS_TEXT}"
unset NPU_RUNTIME_ARGS_TEXT
