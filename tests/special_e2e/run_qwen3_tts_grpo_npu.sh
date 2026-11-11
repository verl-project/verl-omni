#!/usr/bin/env bash
# Ascend port of GPU omni entry 3: two full-parameter GRPO updates.
# Requires pinned qwen-tts, audio helpers, SoX and the cached tiny source.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

NUM_NPUS=${NUM_NPUS:-8}
if [[ ! "${NUM_NPUS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_NPUS < 2 || NUM_NPUS % 2 != 0 )); then
    echo "NUM_NPUS must be an even integer of at least 2." >&2
    exit 2
fi
export NUM_NPUS NUM_GPUS="${NUM_NPUS}" DEVICE_NAME=npu
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES=0
    for ((npu_index = 1; npu_index < NUM_NPUS; npu_index++)); do
        ASCEND_RT_VISIBLE_DEVICES+=",${npu_index}"
    done
fi
export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES-1}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"

set +u
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

python3 - <<'PY'
import os

import torch
import torch_npu  # noqa: F401

expected = int(os.environ["NUM_NPUS"])
for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
    ids = os.environ[name].split(",")
    size_valid = len(ids) == expected if name == "ASCEND_RT_VISIBLE_DEVICES" else len(ids) >= expected
    if not size_valid or any(not item.isdecimal() for item in ids) or len({int(item) for item in ids}) != len(ids):
        raise SystemExit(f"Invalid {name} for {expected} NPUs: {ids}")
if not torch.npu.is_available() or torch.npu.device_count() != expected:
    raise SystemExit(f"Expected {expected} visible NPUs, found {torch.npu.device_count()}.")
PY

runtime_args=(
    "trainer.device=npu"
    "++ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_VISIBLE_DEVICES=\"${ASCEND_VISIBLE_DEVICES}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=\"${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES}\""
    '++ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"'
    '++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE="1"'
    '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ="0"'
    '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0'
)
if [[ -n "${RAY_MASTER_PORT_RANGE:-}" ]]; then
    runtime_args+=("trainer.ray_master_port_range=[${RAY_MASTER_PORT_RANGE}]")
fi
if [[ "${NPU_SMOKE_SKIP_GLOBAL_CLEANUP:-0}" == 1 ]]; then
    runtime_args+=("+ray_kwargs.ray_init.address=local")
fi

export WORK_DIR="${WORK_DIR:-${HOME}/data/qwen3_tts_grpo_npu/$(date +%Y%m%d_%H%M%S)_$$}"
unset MODEL_PATH TRAIN_FILES VAL_FILES

mini_batch_size=$(( (4 + NUM_NPUS - 1) / NUM_NPUS * NUM_NPUS ))

exec bash "${SCRIPT_DIR}/run_qwen3_tts_grpo_smoke.sh" \
    "${runtime_args[@]}" \
    "data.train_batch_size=${mini_batch_size}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}" \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.attention_config.backend=null' \
    "$@"
