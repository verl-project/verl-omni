#!/usr/bin/env bash
set -euo pipefail

# Download every local model required by the LTX-2.3 OmniNFT recipe.
# HF_ENDPOINT and HF_TOKEN are passed to the Hugging Face CLI unchanged.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/outputs}"
REWARD_ROOT="${REWARD_ROOT:-${MODEL_ROOT}/omninft-rewards}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: Hugging Face CLI 'hf' is required; install huggingface_hub first" >&2
  exit 127
fi

mkdir -p "${MODEL_ROOT}"

echo "Downloading the LTX-2.3 base model -> ${MODEL_ROOT}"
hf download "diffusers/LTX-2.3-Diffusers" \
  --revision "8eee8edcf067e838b843f926ec4d4cc9b2be1aaf" \
  --cache-dir "${MODEL_ROOT}"

REWARD_ROOT="${REWARD_ROOT}" "${SCRIPT_DIR}/download_reward_models.sh"

test -d "${MODEL_ROOT}/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"

echo "All models are ready under ${MODEL_ROOT}"
