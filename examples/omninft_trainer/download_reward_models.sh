#!/usr/bin/env bash
set -euo pipefail

# Install dependencies and download every local asset used by the OmniNFT reward pipeline.
# HF_ENDPOINT and HF_TOKEN are passed to the Hugging Face CLI unchanged.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
REWARD_ROOT="${REWARD_ROOT:-${REPO_ROOT}/outputs/omninft-rewards}"
REFERENCE_ROOT="${REWARD_ROOT}/OmniNFT-reference"
REFERENCE_REVISION="fb9237f6e74edf0d0f2a683f4d975b79fde588fe"

if ! command -v pip >/dev/null 2>&1; then
  echo "error: 'pip' is required to install OmniNFT reward dependencies" >&2
  exit 127
fi

echo "Installing OmniNFT reward dependencies"
pip install \
  "audiobox_aesthetics==0.0.4" \
  "qwen-vl-utils==0.0.14" \
  "timm==1.0.27"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: the Hugging Face CLI ('hf') is required; install huggingface_hub first" >&2
  exit 127
fi

mkdir -p "${REWARD_ROOT}"

download_repo() {
  local repo="$1"
  local revision="$2"
  local destination="$3"

  echo "Downloading ${repo}@${revision} -> ${destination}"
  mkdir -p "${destination}"
  hf download "${repo}" \
    --revision "${revision}" \
    --local-dir "${destination}"
}

download_file() {
  local repo="$1"
  local file="$2"
  local revision="$3"

  echo "Downloading ${repo}:${file}@${revision} -> ${REWARD_ROOT}"
  hf download "${repo}" "${file}" \
    --revision "${revision}" \
    --local-dir "${REWARD_ROOT}"
}

download_repo \
  "KlingTeam/VideoReward" \
  "4f26600130683e6f1de9f5d463887f28e8ef995c" \
  "${REWARD_ROOT}/VideoReward"

download_repo \
  "MizzenAI/HPSv3" \
  "4f81e3e09edd82fe3c5f636444c721b592a735ca" \
  "${REWARD_ROOT}/HPSv3"

download_repo \
  "facebook/audiobox-aesthetics" \
  "9b1dd8e5df9af7216e836a98974fe3b82c56ded6" \
  "${REWARD_ROOT}/audiobox-aesthetics"

download_repo \
  "laion/clap-htsat-unfused" \
  "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a" \
  "${REWARD_ROOT}/checkpoints/clap-htsat-unfused"

download_file \
  "zghhui/OmniNFT-Reward-Series" \
  "synchformer/synchformer_state_dict.pth" \
  "9e30061a1392d03bafdcf717e80a385ddf411b4d"

download_repo \
  "Qwen/Qwen2-VL-2B-Instruct" \
  "895c3a49bc3fa70a340399125c650a463535e71c" \
  "${REWARD_ROOT}/Qwen2-VL-2B-Instruct"

download_repo \
  "Qwen/Qwen2-VL-7B-Instruct" \
  "eed13092ef92e448dd6875b2a00151bd3f7db0ac" \
  "${REWARD_ROOT}/Qwen2-VL-7B-Instruct"

if [[ -e "${REFERENCE_ROOT}" ]]; then
  echo "Using existing OmniNFT reference source at ${REFERENCE_ROOT}"
else
  echo "Cloning OmniNFT reference source -> ${REFERENCE_ROOT}"
  git clone "https://github.com/zghhui/OmniNFT.git" "${REFERENCE_ROOT}"
  git -C "${REFERENCE_ROOT}" checkout --detach "${REFERENCE_REVISION}"
fi

for source_file in synchformer.py divided_224_16x4.yaml; do
  if [[ ! -f "${REFERENCE_ROOT}/flow_grpo/audio_video_align/synchformer/${source_file}" ]]; then
    echo "error: ${REFERENCE_ROOT} is missing DeSync source file ${source_file}" >&2
    exit 1
  fi
done

verify_sha256() {
  local relative_path="$1"
  local expected="$2"
  local path="${REWARD_ROOT}/${relative_path}"

  if [[ ! -f "${path}" ]]; then
    echo "error: expected file is missing: ${path}" >&2
    exit 1
  fi

  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "error: SHA-256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
  echo "Verified ${relative_path}"
}

echo "Verifying core reward checkpoints"
verify_sha256 \
  "VideoReward/checkpoint-11352/model.pth" \
  "48375908e6112de9f0248402db156a23b480709a6960b091c598c6f4c88d21b9"
verify_sha256 \
  "VideoReward/checkpoint-11352/tokenizer/tokenizer.json" \
  "a05e80818b958b49531b8e1f758c4ae65f090fdec80e6d8408963f70ae3f737d"
verify_sha256 \
  "HPSv3/HPSv3.safetensors" \
  "a13d7ff5a07b7ffa0f7824e60d62e6ae144541ceefd5224b4c08fda7ab39f353"
verify_sha256 \
  "audiobox-aesthetics/checkpoint.pt" \
  "a4931a7a01c3e6733352e9d85371835f03bf9135f8b31e1583c23538811d4a32"
verify_sha256 \
  "audiobox-aesthetics/model.safetensors" \
  "a5a3c2412649cc2384ec525ffd5180ce6c4778f43bed6108e0a1303de04d014e"
verify_sha256 \
  "checkpoints/clap-htsat-unfused/pytorch_model.bin" \
  "1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb"
verify_sha256 \
  "synchformer/synchformer_state_dict.pth" \
  "8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c"

echo "All OmniNFT reward models and dependencies are ready under ${REWARD_ROOT}"
