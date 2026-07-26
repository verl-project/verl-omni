#!/usr/bin/env bash
# Run all verl-omni GPU smoke test groups.
#
# Usage:
#   bash tests/gpu_smoke/run_gpu_smoke_tests.sh
#
# To run locally on selected CUDA devices:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash tests/gpu_smoke/run_gpu_smoke_tests.sh
#
# To run a smaller resource group directly, execute one of:
#   bash tests/gpu_smoke/run_gpu_smoke_core.sh
#   bash tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh
#   bash tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh

set -euo pipefail

GPU_SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Run all verl-omni GPU smoke test groups.

Usage:
  bash tests/gpu_smoke/run_gpu_smoke_tests.sh

To run locally on selected CUDA devices:
  CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash tests/gpu_smoke/run_gpu_smoke_tests.sh

To run a smaller resource group directly, execute one of:
  bash tests/gpu_smoke/run_gpu_smoke_core.sh
  bash tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh
  bash tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh
EOF
    exit 0
fi

if [[ $# -gt 0 ]]; then
    echo "Unknown argument '$1'. Run individual group scripts to pass per-group options." >&2
    exit 2
fi

bash "${GPU_SMOKE_DIR}/run_gpu_smoke_core.sh"
bash "${GPU_SMOKE_DIR}/run_gpu_smoke_omni_e2e.sh"
bash "${GPU_SMOKE_DIR}/run_gpu_smoke_diffusion_e2e.sh"
