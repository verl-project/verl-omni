#!/usr/bin/env bash

# Sync the pinned rollout stack onto an NPU CI base image.
#
# The registry image carries a full prebuilt stack (CANN, torch, torch-npu,
# triton-ascend, editable vllm / vllm-ascend / vllm-omni / verl, verl-omni
# metadata) at the generation it was built for. When a pin bump lands before
# the matching image tag is pushed, this script reinstalls the pinned
# generation on top of the previous image so npu_smoke keeps running the
# pinned stack instead of a missing tag.
#
# It mirrors docker/Dockerfile.npu's install stages (tests/special_sanity/
# test_pin_sync.py asserts the shared pins stay in sync), including the
# triton-ascend-last ordering. Deviations, all deliberate:
# - vllm-ascend is installed pure-python (COMPILE_CUSTOM_KERNELS=0,
#   --no-deps): the smoke suite runs tiny models through the native op path,
#   and an in-job AscendC build would cost most of the job timeout.
# - memfabric-hybrid/memcache-hybrid (vllm-ascend kv-offload lazy imports)
#   are not installed: no smoke test exercises kv offload.
# - fastapi is pinned to the window vllm-ascend's requirements state; the
#   base image may sit below its floor.
# - torch/torch-npu are not force-reinstalled at the end: nothing in the
#   pinned chain requires a torch version, and the base image already
#   carries torch 2.10.0+cpu with torch-npu installed by this script.
#
# TODO (mike): drop this script and its workflow step once the registry image
# at the current generation exists; the workflow then pins that tag directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_ROOT="${NPU_SYNC_SRC_ROOT:-/workspace/src}"

# Mirrors docker/Dockerfile.npu (test_pin_sync.py asserts the shared pins
# stay in sync). The a2 variant matches this CI pool's 910B4 devices.
TORCH_NPU_PIN="2.10.0.post7"
TRITON_ASCEND_PIN="3.6.0.post2"
FLA_NPU_PIN="flash-linear-attention-npu-a2==26.9.1"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
ASCEND_PIP_INDEX_URL="${ASCEND_PIP_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
PYTORCH_CPU_INDEX_URL="${PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu/}"

log() { echo "[sync_npu_stack] $*"; }

VLLM_OMNI_PIN="$(tr -d '[:space:]' < "${REPO_ROOT}/.github/vllm_omni_pin.txt")"
VLLM_ASCEND_PIN="$(tr -d '[:space:]' < "${REPO_ROOT}/.github/vllm_ascend_pin.txt")"
VERL_PIN="$(tr -d '[:space:]' < "${REPO_ROOT}/.github/verl_pin.txt")"
VLLM_TAG="$(sed -n 's/.*"vllm==\([0-9][0-9.]*\)".*/v\1/p' "${REPO_ROOT}/pyproject.toml" | head -n 1)"
[[ "${VLLM_TAG}" =~ ^v[0-9.]+$ ]] || { echo "cannot read the vllm pin from pyproject.toml" >&2; exit 1; }

log "pins: vllm ${VLLM_TAG}, vllm-ascend ${VLLM_ASCEND_PIN}, vllm-omni ${VLLM_OMNI_PIN}, verl ${VERL_PIN}"

# Ascend env scripts read unbound shell variables; suspend -u while sourcing.
set +u
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

pip_install() {
    python3 -m pip install "$@" \
        --extra-index-url "${ASCEND_PIP_INDEX_URL}" \
        --extra-index-url "${PYTORCH_CPU_INDEX_URL}"
}

# 1. torch-npu: the generation the pinned vllm-ascend requires.
log "installing torch-npu==${TORCH_NPU_PIN}"
pip_install "torch-npu==${TORCH_NPU_PIN}"

# 2. vllm at the pyproject tag, empty target (no CUDA kernels on NPU).
if [[ ! -d "${SRC_ROOT}/vllm/.git" ]]; then
    git clone --depth 1 -b "${VLLM_TAG}" https://github.com/vllm-project/vllm.git "${SRC_ROOT}/vllm"
else
    git -C "${SRC_ROOT}/vllm" fetch --depth 1 origin "refs/tags/${VLLM_TAG}:refs/tags/${VLLM_TAG}"
    git -C "${SRC_ROOT}/vllm" checkout --detach "${VLLM_TAG}"
fi
log "installing vllm ${VLLM_TAG} (empty target)"
(
    cd "${SRC_ROOT}/vllm"
    VLLM_TARGET_DEVICE="empty" pip_install --no-build-isolation -e ".[audio]"
    python3 -m pip uninstall -y triton || true
)

# 3. vllm-ascend at the pin, pure-python.
if [[ ! -d "${SRC_ROOT}/vllm-ascend/.git" ]]; then
    git clone https://github.com/vllm-project/vllm-ascend.git "${SRC_ROOT}/vllm-ascend"
fi
git -C "${SRC_ROOT}/vllm-ascend" fetch origin "${VLLM_ASCEND_PIN}" --depth 1
git -C "${SRC_ROOT}/vllm-ascend" checkout --detach "${VLLM_ASCEND_PIN}"
# Drop op binaries built for the previous generation before the pure-python install.
git -C "${SRC_ROOT}/vllm-ascend" clean -xffd
git -C "${SRC_ROOT}/vllm-ascend" submodule update --init --recursive
log "installing vllm-ascend ${VLLM_ASCEND_PIN} (pure-python)"
(
    cd "${SRC_ROOT}/vllm-ascend"
    COMPILE_CUSTOM_KERNELS=0 VLLM_BATCH_INVARIANT=1 \
        pip_install --no-build-isolation --no-deps -e .
    pip_install concurrent-log-handler
    # The pin's fastapi window; vllm accepts it and the image may sit below the floor.
    pip_install 'fastapi>=0.133.0,<0.137.0'
    # Last, like the Dockerfile: a triton pulled by any earlier step would
    # clobber triton-ascend's triton/ tree.
    python3 -m pip uninstall -y triton triton-ascend || true
    pip_install "triton-ascend==${TRITON_ASCEND_PIN}"
    # Kernel operators vllm-ascend's worker init imports on A2/A3/950.
    pip_install --no-deps "${FLA_NPU_PIN}"
)

# 4. vllm-omni at the pin, npu target.
if [[ ! -d "${SRC_ROOT}/vllm-omni/.git" ]]; then
    git clone https://github.com/vllm-project/vllm-omni.git "${SRC_ROOT}/vllm-omni"
fi
git -C "${SRC_ROOT}/vllm-omni" fetch origin "${VLLM_OMNI_PIN}" --depth 1
git -C "${SRC_ROOT}/vllm-omni" checkout --detach "${VLLM_OMNI_PIN}"
log "installing vllm-omni ${VLLM_OMNI_PIN} (npu target)"
(
    cd "${SRC_ROOT}/vllm-omni"
    VLLM_OMNI_TARGET_DEVICE=npu pip_install -e .
)

# 5. verl at the pin (the base image usually already carries it).
if [[ ! -d "${SRC_ROOT}/verl/.git" ]] || [[ "$(git -C "${SRC_ROOT}/verl" rev-parse HEAD)" != "${VERL_PIN}" ]]; then
    if [[ ! -d "${SRC_ROOT}/verl/.git" ]]; then
        git clone https://github.com/verl-project/verl.git "${SRC_ROOT}/verl"
    fi
    git -C "${SRC_ROOT}/verl" fetch origin "${VERL_PIN}" --depth 1
    git -C "${SRC_ROOT}/verl" checkout --detach "${VERL_PIN}"
    log "installing verl ${VERL_PIN}"
    (cd "${SRC_ROOT}/verl" && pip_install -e .)
else
    log "verl already at ${VERL_PIN}"
fi

# 6. The verl-omni checkout itself: metadata only, imports resolve to the job
#    workspace (see docs/contributing/npu_smoke_tests.md).
log "installing verl-omni checkout (no-deps)"
pip_install --no-deps -e "${REPO_ROOT}"
pip_install diffusers==0.41.0 Levenshtein

# 7. Transformers range + TransferQueue, mirroring the Dockerfile tail.
pip_install 'transformers>=5.16.1,<5.20.0' 'TransferQueue==0.1.9'
pip_install --upgrade onnxruntime

log "synced stack:"
python3 -m pip show vllm vllm-ascend vllm-omni verl verl-omni torch-npu triton-ascend transformers 2>/dev/null \
    | grep -E "^(Name|Version|Location):" || true
