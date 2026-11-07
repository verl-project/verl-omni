#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_npu_smoke_core.sh" "$@"
"${SCRIPT_DIR}/run_npu_smoke_e2e.sh" "$@"
