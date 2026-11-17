#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${NPU_SMOKE_PARALLEL:-0}" == 1 ]]; then
    exec bash "${SCRIPT_DIR}/run_npu_smoke_parallel.sh" "$@"
fi

"${SCRIPT_DIR}/run_npu_smoke_core.sh" "$@"
"${SCRIPT_DIR}/run_npu_smoke_e2e.sh" "$@"
