#!/usr/bin/env bash
# Launch L4 convergence case(s).
#
# This script is the single source of truth for:
#   - the case list (name -> case script), in run order
#   - default local weight snapshots (override via MODEL_PATH/REWARD_MODEL_PATH)
#
# Examples:
#   bash tests/convergence/run_l4_convergence.sh --list
#   bash tests/convergence/run_l4_convergence.sh sd35_medium_ocr_lora_v1
#   bash tests/convergence/run_l4_convergence.sh --device npu sd35_medium_ocr_lora_v1
#   bash tests/convergence/run_l4_convergence.sh --all
#   bash tests/convergence/run_l4_convergence.sh sd35_medium_ocr_lora_v1 -- trainer.total_training_steps=10
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

OUTPUT_ROOT=${OUTPUT_ROOT:-tests/convergence/outputs/l4_convergence}

# L4 cases run on the 8-accelerator CI runner (L20x8 / linux-aarch64-a2b4-8).
REQUIRED_GPUS=8

# Registry: case names in run order, each mapped to its case script.
CASE_ORDER=(sd35_medium_ocr_lora_v1)
declare -A CASE_SCRIPTS=(
    [sd35_medium_ocr_lora_v1]="tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh"
)

# Local weight snapshots baked into the runner/home cache; exported so every
# case script picks them up. When a snapshot is absent (e.g. a fresh runner),
# fall back to the HF id so weights download on first run. Override by
# exporting MODEL_PATH/REWARD_MODEL_PATH beforehand.
MODEL_SNAPSHOT=${MODEL_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80}
REWARD_SNAPSHOT=${REWARD_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d "${MODEL_SNAPSHOT}" ]]; then MODEL_PATH="${MODEL_SNAPSHOT}"; else MODEL_PATH="stabilityai/stable-diffusion-3.5-medium"; fi
    export MODEL_PATH
fi
if [[ -z "${REWARD_MODEL_PATH:-}" ]]; then
    if [[ -d "${REWARD_SNAPSHOT}" ]]; then REWARD_MODEL_PATH="${REWARD_SNAPSHOT}"; else REWARD_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"; fi
    export REWARD_MODEL_PATH
fi

# Raw OCR splits (one prompt per line). Cases derive their parquet dir from
# this: OCR_DATA_DIR=${OCR_RAW_DIR}/sd3.
export OCR_RAW_DIR=${OCR_RAW_DIR:-$HOME/.cache/huggingface/hub/verl-omni}

usage() {
    cat <<EOF
Usage: $(basename "$0") [--all | CASE ...] [OPTIONS] [-- HYDRA_OVERRIDE ...]

Options:
  --all             run every registered case (default when no CASE is given)
  --list            list registered case names and exit
  --device DEVICE   export DEVICE=gpu|cuda|npu|auto before running
                    (default: each case script auto-detects the accelerator)
  --dry-run         print the resolved case plan without running anything
  -h, --help        show this help

Environment:
  MODEL_PATH, REWARD_MODEL_PATH   default to the local weight snapshots when
                                  present, else the HF model ids
  OCR_RAW_DIR                     raw OCR train.txt/test.txt dir (default:
                                  ~/.cache/huggingface/hub/verl-omni); generated
                                  parquet goes to ${OCR_RAW_DIR}/sd3

Arguments after -- are forwarded to the case script as Hydra overrides.
Only allowed when exactly one case is selected.

Registered cases:
$(printf '  %s\n' "${CASE_ORDER[@]}")
EOF
}

fail() { echo "[FAIL] $*" >&2; exit 1; }

is_case() {
    local known
    for known in "${CASE_ORDER[@]}"; do
        [[ "${known}" == "$1" ]] && return 0
    done
    return 1
}

LIST_ONLY=0
DRY_RUN=0
SELECT_ALL=0
DEVICE_ARG=""
CASES=()
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --all)
            SELECT_ALL=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --device)
            [[ $# -ge 2 ]] || fail "--device requires a value"
            DEVICE_ARG="$2"
            shift 2
            ;;
        --device=*)
            DEVICE_ARG="${1#*=}"
            shift
            ;;
        --)
            shift
            EXTRA=("$@")
            break
            ;;
        -*)
            fail "Unknown option: $1 (see --help)"
            ;;
        *)
            CASES+=("$1")
            shift
            ;;
    esac
done

if [[ "${LIST_ONLY}" -eq 1 ]]; then
    printf '%s\n' "${CASE_ORDER[@]}"
    exit 0
fi

if [[ -n "${DEVICE_ARG}" ]]; then
    case "${DEVICE_ARG}" in
        gpu|cuda|npu|auto) export DEVICE="${DEVICE_ARG}" ;;
        *) fail "Unsupported --device ${DEVICE_ARG} (expected gpu|cuda|npu|auto)" ;;
    esac
fi

if [[ "${SELECT_ALL}" -eq 1 && ${#CASES[@]} -gt 0 ]]; then
    fail "--all cannot be combined with explicit case names"
fi

if [[ ${#EXTRA[@]} -gt 0 && ( "${SELECT_ALL}" -eq 1 || ${#CASES[@]} -ne 1 ) ]]; then
    fail "Hydra overrides after -- require exactly one selected case"
fi

for want in ${CASES[@]+"${CASES[@]}"}; do
    is_case "${want}" || fail "Unknown case '${want}' (known: ${CASE_ORDER[*]})"
done

# Select in registry order for deterministic runs.
SELECTED=()
for known in "${CASE_ORDER[@]}"; do
    for want in ${CASES[@]+"${CASES[@]}"}; do
        if [[ "${want}" == "${known}" ]]; then
            SELECTED+=("${known}")
            break
        fi
    done
done
if [[ "${SELECT_ALL}" -eq 1 || ${#CASES[@]} -eq 0 ]]; then
    SELECTED=("${CASE_ORDER[@]}")
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    fail "No L4 convergence cases selected."
fi

echo "L4 plan: ${SELECTED[*]} (${#SELECTED[@]} case(s), ${REQUIRED_GPUS} ${DEVICE:-auto} accelerators each)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "REWARD_MODEL_PATH=${REWARD_MODEL_PATH}"
echo "OCR_RAW_DIR=${OCR_RAW_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run — would execute:"
    for name in "${SELECTED[@]}"; do
        echo "  ${name}: ${CASE_SCRIPTS[${name}]}"
    done
    exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
ray stop --force || true

failed=0
for name in "${SELECTED[@]}"; do
    script="${CASE_SCRIPTS[${name}]}"
    echo "===== Starting L4 case ${name} (requires ${REQUIRED_GPUS} accelerators, DEVICE=${DEVICE:-auto}) ====="
    [[ -f "${script}" ]] || { echo "[FAIL] ${name}: missing script ${script}" >&2; failed=1; continue; }
    if bash "${script}" ${EXTRA[@]+"${EXTRA[@]}"}; then
        echo "[PASS] ${name}"
    else
        echo "[FAIL] ${name}"
        failed=1
    fi
done

ray stop --force || true

if [[ "${failed}" -ne 0 ]]; then
    echo "One or more L4 convergence cases failed." >&2
fi
exit "${failed}"
