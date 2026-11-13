#!/usr/bin/env bash
# Shared L3 nightly runner. Case scripts export L3_* / TRAIN_BATCH_SIZE and
# pass the v1 trainer Hydra overrides as arguments.
#
# Training always goes through tests/nightly/run.py, which starts
# verl_omni.trainer.main_diffusion_v1. Do not point a case at the v0 trainer.
set -xeuo pipefail

NIGHTLY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${NIGHTLY_DIR}/../.." && pwd)

: "${L3_TEST_CASE:?set L3_TEST_CASE}"
: "${L3_ALGORITHM:?set L3_ALGORITHM}"
: "${L3_CASE_LABEL:?set L3_CASE_LABEL}"
: "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}"

L3_MODEL=${L3_MODEL:-qwen-image}
L3_DATA_SOURCE=${L3_DATA_SOURCE:-${L3_TEST_CASE}}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${HOME}/models/tiny-random/qwen3-vl}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-20}
DEBUG_DUMP_STEPS=${DEBUG_DUMP_STEPS:-1,2}
PERF_SKIP_STEPS=${PERF_SKIP_STEPS:-2}
PERF_THRESHOLD=${PERF_THRESHOLD:-0.15}
PRECISION_ATOL=${PRECISION_ATOL:-1e-3}
PRECISION_MEAN_ATOL=${PRECISION_MEAN_ATOL:-1e-4}
PRECISION_RMSE_ATOL=${PRECISION_RMSE_ATOL:-1e-3}
PRECISION_P99_ATOL=${PRECISION_P99_ATOL:-2e-3}
PRECISION_MAX_FRAC_ABS_OVER_ATOL=${PRECISION_MAX_FRAC_ABS_OVER_ATOL:-2e-2}
PRECISION_MIN_COS_SIM=${PRECISION_MIN_COS_SIM:-0.99}
BOOTSTRAP_MISSING_BASELINE=${BOOTSTRAP_MISSING_BASELINE:-1}
L3_REFRESH_BASELINE=${L3_REFRESH_BASELINE:-0}
NIGHTLY_DETERMINISTIC_SEED=${NIGHTLY_DETERMINISTIC_SEED:-42}

DATA_DIR=${DATA_DIR:-${HOME}/data/${L3_TEST_CASE}}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/debug_dumps}
CURRENT_DUMP_DIR=${CURRENT_DUMP_DIR:-${OUTPUT_ROOT}/current/${L3_TEST_CASE}}
BASELINE_DUMP_DIR=${BASELINE_DUMP_DIR:-${OUTPUT_ROOT}/baseline/${L3_TEST_CASE}}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs/${L3_TEST_CASE}}
CONSOLE_LOG=${CONSOLE_LOG:-${LOG_DIR}/${L3_TEST_CASE}.log}
DEBUG_METRICS_JSONL=${DEBUG_METRICS_JSONL:-${CURRENT_DUMP_DIR}/metrics.jsonl}
CURRENT_METRICS_JSON=${CURRENT_METRICS_JSON:-${CURRENT_DUMP_DIR}/metrics.json}
BASELINE_METRICS_JSON=${BASELINE_METRICS_JSON:-${BASELINE_DUMP_DIR}/metrics.json}
DUMP_COMPARE_JSON=${DUMP_COMPARE_JSON:-${CURRENT_DUMP_DIR}/dump_compare.json}
ENV_METADATA_JSON=${ENV_METADATA_JSON:-${LOG_DIR}/env_metadata.json}

# The GPU smoke image may contain the ``kernels`` Python package while its
# installed Torch/CUDA combination has no compatible Hub FA3 build variant.
# Select the portable native/SDPA pair explicitly for these E2E tests so the
# production engine can keep its fail-fast attention-backend behavior.
ATTN_BACKEND=${ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}
echo "[NIGHTLY] case=${L3_TEST_CASE} diffusion attention: ATTN_BACKEND=${ATTN_BACKEND} ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND}"

export NIGHTLY_DETERMINISTIC_SEED
export PYTHONHASHSEED=${PYTHONHASHSEED:-${NIGHTLY_DETERMINISTIC_SEED}}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export GENRM_OCR_TEMPERATURE=${GENRM_OCR_TEMPERATURE:-0.0}
export GENRM_OCR_TOP_P=${GENRM_OCR_TOP_P:-1.0}
export GENRM_OCR_MAX_TOKENS=${GENRM_OCR_MAX_TOKENS:-32}
export GENRM_OCR_SEED=${GENRM_OCR_SEED:-${NIGHTLY_DETERMINISTIC_SEED}}
export MODEL_PATH TOKENIZER_PATH REWARD_MODEL_PATH
export TRAIN_FILES VAL_FILES ATTN_BACKEND ROLLOUT_ATTN_BACKEND

rm -rf "${CURRENT_DUMP_DIR}"
mkdir -p "${CURRENT_DUMP_DIR}" "${BASELINE_DUMP_DIR}" "${LOG_DIR}"

python3 "${NIGHTLY_DIR}/env_metadata.py" \
    --output "${ENV_METADATA_JSON}" \
    --attn-backend "${ATTN_BACKEND}" \
    --rollout-attn-backend "${ROLLOUT_ATTN_BACKEND}"

python3 "${NIGHTLY_DIR}/create_single_sample_data.py" \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${TRAIN_BATCH_SIZE}" \
    --val_size 4 \
    --data_source "${L3_DATA_SOURCE}"

export DEBUG_DUMP_ENABLED=1
export DEBUG_DUMP_DIR="${CURRENT_DUMP_DIR}"
export DEBUG_DUMP_STEPS
export DEBUG_METRICS_JSONL

python3 "${NIGHTLY_DIR}/run.py" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    "$@" 2>&1 | tee "${CONSOLE_LOG}"

# Replace only this case's baseline after a successful run. Sibling case
# directories under OUTPUT_ROOT/baseline stay in place.
if [ "${L3_REFRESH_BASELINE}" = "1" ]; then
    BOOTSTRAP_MISSING_BASELINE=1
    rm -rf "${BASELINE_DUMP_DIR}"
    mkdir -p "${BASELINE_DUMP_DIR}"
fi

BOOTSTRAP_ARGS=()
if [ "${BOOTSTRAP_MISSING_BASELINE}" = "1" ]; then
    BOOTSTRAP_ARGS+=(--bootstrap-missing)
fi

METRICS_STATUS=0
python3 "${NIGHTLY_DIR}/collect_metrics.py" \
    --metrics-jsonl "${DEBUG_METRICS_JSONL}" \
    --log-file "${CONSOLE_LOG}" \
    --baseline "${BASELINE_METRICS_JSON}" \
    --output "${CURRENT_METRICS_JSON}" \
    --model "${L3_MODEL}" \
    --algorithm "${L3_ALGORITHM}" \
    --perf-skip-steps "${PERF_SKIP_STEPS}" \
    --threshold "${PERF_THRESHOLD}" \
    "${BOOTSTRAP_ARGS[@]}" || METRICS_STATUS=$?

DUMP_STATUS=0
python3 "${NIGHTLY_DIR}/compare_dumps.py" \
    --baseline "${BASELINE_DUMP_DIR}" \
    --current "${CURRENT_DUMP_DIR}" \
    --output "${DUMP_COMPARE_JSON}" \
    --atol "${PRECISION_ATOL}" \
    --mean-atol "${PRECISION_MEAN_ATOL}" \
    --rmse-atol "${PRECISION_RMSE_ATOL}" \
    --p99-atol "${PRECISION_P99_ATOL}" \
    --max-frac-abs-over-atol "${PRECISION_MAX_FRAC_ABS_OVER_ATOL}" \
    --min-cos-sim "${PRECISION_MIN_COS_SIM}" \
    "${BOOTSTRAP_ARGS[@]}" || DUMP_STATUS=$?

if [ "${METRICS_STATUS}" -ne 0 ] || [ "${DUMP_STATUS}" -ne 0 ]; then
    echo "${L3_CASE_LABEL} nightly regression failed: metrics_status=${METRICS_STATUS}, dump_status=${DUMP_STATUS}."
    exit 1
fi

echo "${L3_CASE_LABEL} nightly regression passed."
