#!/usr/bin/env bash
# NPU launcher for the L4 SD3.5 OCR LoRA v1 convergence test.
#
# Sets up the Ascend environment and delegates to the shared device-aware
# test script (run_sd35_medium_ocr_lora_v1.sh) with DEVICE=npu.
#
# Requirements mirror the GPU job, on Ascend NPUs instead:
#   - At least NUM_GPUS_ACTOR_ROLLOUT + NUM_GPUS_REWARD NPUs (default 4+2=6;
#     8 are exposed by default via ASCEND_RT_VISIBLE_DEVICES)
#   - CANN installed under /usr/local/Ascend (override via ASCEND_HOME_PATH)
#   - stabilityai/stable-diffusion-3.5-medium (MODEL_PATH)
#   - Qwen/Qwen2.5-VL-3B-Instruct (REWARD_MODEL_PATH)
#   - $WORKSPACE/data/ocr/sd3/{train,test}.parquet
#     (python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py --output_dir ...)
#
# Run:
#   bash tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1_npu.sh
#
# Any extra arguments are forwarded as Hydra overrides to the trainer.
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DEVICE=npu
export DEVICE_NAME=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
export VERL_DATAPROTO_SERIALIZATION_METHOD=${VERL_DATAPROTO_SERIALIZATION_METHOD:-numpy}
# Keep NPU artifacts separate from the GPU run by default.
export L4_TEST_CASE=${L4_TEST_CASE:-sd35_medium_ocr_lora_v1_npu}

NUM_NPUS=${NUM_NPUS:-8}
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    npu_list=""
    for ((i = 0; i < NUM_NPUS; i++)); do
        npu_list="${npu_list:+${npu_list},}${i}"
    done
    export ASCEND_RT_VISIBLE_DEVICES="${npu_list}"
fi

echo "===== Source Ascend environment ====="
set +u
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi
if [ -f "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"
fi
set -u
echo "===== Ascend environment after source ====="
ascend_env="$(
    {
        env | LC_ALL=C sort | grep -F "/usr/local/Ascend/" || true
        env | LC_ALL=C sort | grep -E "^(ASCEND|ATB|HCCL|NPU|DEVICE_|RANK_TABLE|RAY_EXPERIMENTAL_NOSET_ASCEND)" || true
    } | LC_ALL=C sort -u
)"
if [[ -n "${ascend_env}" ]]; then
    printf "%s\n" "${ascend_env}"
else
    echo "No Ascend/NPU environment entries were found."
fi
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info || true
fi
echo "======================================"

bash "${SCRIPT_DIR}/run_sd35_medium_ocr_lora_v1.sh" "$@"

echo "L4 SD3.5 OCR LoRA v1 NPU precision check passed."
