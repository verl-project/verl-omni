#!/usr/bin/env bash
# Qwen3-TTS full-parameter GRPO e2e smoke: tiny model, two updates.
# This validates execution only; the in-process CPU reward is not a quality reward.

set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:${CPATH}}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-2}"
[[ "${NUM_GPUS}" =~ ^[0-9]+$ && "${NUM_GPUS}" -ge 2 ]] || {
    echo "Qwen3-TTS smoke requires at least two GPUs" >&2
    exit 2
}
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_REPO="${MODEL_REPO:-optimum-intel-internal-testing/tiny-random-qwen3-tts}"
MODEL_REVISION="${MODEL_REVISION:-6374d605b31381cac6f9577f5e742af2f76ba79c}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/qwen3_tts_grpo_smoke_${USER:-user}_$$}"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output}"
mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" -c \
    'from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration; import onnxruntime, soundfile, librosa, sox' \
    || { echo "Qwen3-TTS smoke dependencies must be installed by gpu-smoke-prepare" >&2; exit 2; }

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    SOURCE_MODEL_PATH="$("${PYTHON_BIN}" -c \
        'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' \
        "${MODEL_REPO}" "${MODEL_REVISION}")"
    # The published tiny checkpoint has four codebooks; the production actor
    # contract has 16, so rebuild its small architecture with random 16-codebook weights.
    MODEL_PATH="${WORK_DIR}/tiny-random-qwen3-tts-16-codebooks"
    "${PYTHON_BIN}" tests/special_e2e/build_qwen3_tts_tiny_random.py \
        --source-model-path "${SOURCE_MODEL_PATH}" \
        --output-dir "${MODEL_PATH}"
fi
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Invalid MODEL_PATH: ${MODEL_PATH}" >&2; exit 2; }

"${PYTHON_BIN}" tests/special_e2e/create_dummy_qwen3_tts_grpo_data.py \
    --output-dir "${DATA_DIR}" \
    --model-config "${MODEL_PATH}/config.json"

MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${DATA_DIR}/train.parquet" \
VAL_FILE="${DATA_DIR}/validation.parquet" \
SPK_EMBED_PATH="${DATA_DIR}/speaker.json" \
SCORER_URL="http://unused.invalid/score" \
OUTPUT_DIR="${OUTPUT_DIR}" \
NUM_GPUS="${NUM_GPUS}" \
TOTAL_TRAINING_STEPS=2 \
TEST_FREQ=-1 \
SAVE_FREQ=-1 \
RESUME_MODE=disable \
PYTHON_BIN="${PYTHON_BIN}" \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    "reward.custom_reward_function.path=${REPO_ROOT}/tests/special_e2e/qwen3_tts_dummy_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.log_val_generations=0 \
    "$@"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/train.log" <<'PY'
import re
import sys
from pathlib import Path

steps = [int(step) for step in re.findall(r"training/global_step:(\d+)(?!\d)", Path(sys.argv[1]).read_text())]
if not steps or max(steps) != 2:
    raise SystemExit(f"Expected training/global_step to reach exactly 2, observed {steps!r}")
PY
echo "Qwen3-TTS GRPO e2e smoke passed; artifacts: ${WORK_DIR}"
