#!/usr/bin/env bash
# Convert raw MMK12 parquet shards with the repository's shared image-math converter.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 RAW_MMK12_DIR OUTPUT_DIR [converter arguments...]" >&2
    exit 2
fi

RAW_MMK12_DIR=$1
OUTPUT_DIR=$2
shift 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

exec python3 "${REPO_ROOT}/examples/gspo_trainer/data_process/mmk12.py" \
    --local_dataset_path "${RAW_MMK12_DIR}" \
    --local_save_dir "${OUTPUT_DIR}" \
    "$@"
