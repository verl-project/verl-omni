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
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
    if (
        len(devices) != expected
        or any(not device.isdecimal() for device in devices)
        or len({int(device) for device in devices}) != expected
    ):
        raise SystemExit(f"{name} must list exactly {expected} distinct NPU IDs: {devices}")

import torch
import torch_npu  # noqa: F401

if not torch.npu.is_available() or torch.npu.device_count() != expected:
    raise SystemExit(f"Expected {expected} visible NPUs, found {torch.npu.device_count()}.")
print(f"NPU e2e: {expected} devices; torch {torch.__version__}; torch-npu {torch_npu.__version__}")
PY

NPU_VISIBLE_DEVICES_JSON="\"${ASCEND_VISIBLE_DEVICES}\""
export NPU_VISIBLE_DEVICES_JSON
NPU_RUNTIME_ARGS=(
    "trainer.device=npu"
    "++ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES_JSON}"
    '++ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="1"'
    '++ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"'
    '++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE="1"'
    '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ="0"'
    '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0'
    # vllm-ascend defaults weight_nz_mode=1 and wake_up rejects FRACTAL_NZ in RL
    # scenarios; reward-model vLLM servers sleep/wake too, so guard them as well.
    '++reward.reward_model.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0'
)
