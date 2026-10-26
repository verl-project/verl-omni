# Installation (Intel GPU)

Last updated: 10/20/2026

For NVIDIA GPU, see the {doc}`GPU installation guide <install>`. For Ascend NPU, see the {doc}`NPU installation guide <install_npu>`. For AMD GPU, see the {doc}`ROCm installation guide <install_rocm>`.

## Requirements

* **Python**: Version >= 3.11 (the image ships 3.12)
* **GPU**: Intel Arc Pro B-series; validated on Arc Pro B60 and B70
* **Host**: Linux with the Intel GPU kernel driver loaded, so the devices appear under `/dev/dri`

## Install

On Intel GPU the Docker image is the supported path. It builds on vLLM's official
XPU image, which already ships torch for XPU, oneCCL, the Intel GPU user-mode
driver and vLLM 0.30.0.

```bash
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
docker build -f docker/Dockerfile.xpu -t verl-omni:xpu .
```

Behind a proxy, add `--build-arg http_proxy=... --build-arg https_proxy=...`.
Build context is controlled by the repo-root [`.dockerignore`](https://github.com/verl-project/verl-omni/blob/main/.dockerignore).

### Build arguments

| Argument       | Default                                              | Purpose                                         |
| -------------- | ---------------------------------------------------- | ----------------------------------------------- |
| `BASE_IMAGE`   | `vllm/vllm-openai-xpu:v0.30.0`                       | torch for XPU, oneCCL, Intel GPU driver, vLLM   |
| `VERL_GIT_REF` | [`.github/verl_pin.txt`](../../.github/verl_pin.txt) | verl commit to install                          |

### Install into an existing vLLM XPU environment

The same steps work in any environment that already has `vllm==0.30.0+xpu`.
Pin the XPU builds first, so the resolver cannot replace them with CUDA or CPU wheels:

```bash
python3 - > /tmp/xpu-constraints.txt <<'PY'
from importlib.metadata import version, PackageNotFoundError
for pkg in ("torch", "torchvision", "torchaudio", "triton", "triton-xpu", "vllm", "vllm-xpu-kernels"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        pass
PY
export UV_CONSTRAINT=/tmp/xpu-constraints.txt VLLM_OMNI_TARGET_DEVICE=xpu
uv pip install --no-deps "verl @ git+https://github.com/verl-project/verl.git@$(cat .github/verl_pin.txt)"
uv pip install -e ".[xpu]"
```

## Dependencies

The `xpu` extra is required on Intel GPU. It installs
[`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin),
which registers the Intel platform with verl. Without it verl does not detect
the GPU and falls back to NCCL. It also installs `dpctl`, which Ray uses to
count Intel GPUs; without it Ray reports no GPU resources.

| Extra      | Install                         | When needed                    |
| ---------- | ------------------------------- | ------------------------------ |
| Intel GPU  | `uv pip install -e ".[xpu]"`    | Always on Intel GPU            |
| OCR reward | `uv pip install -e ".[ocr]"`    | Training with OCR-based reward |

## Run

```bash
export REPO=/path/to/verl-omni
export WORKSPACE=$HOME

mkdir -p \
  "$WORKSPACE/data" \
  "$WORKSPACE/checkpoints" \
  "$HOME/.cache/huggingface"

docker run -it --rm \
  --ipc=host \
  --device=/dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --group-add="$(getent group render | cut -d: -f3)" \
  -v "$REPO:/workspace/verl-omni" \
  -v "$WORKSPACE/data:$WORKSPACE/data" \
  -v "$WORKSPACE/checkpoints:$WORKSPACE/checkpoints" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e WORKSPACE="$WORKSPACE" \
  -e HF_HOME=/root/.cache/huggingface \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -w /workspace/verl-omni \
  verl-omni:xpu \
  bash
```

`--device=/dev/dri` passes the device nodes but not `/dev/dri/by-path`, which
oneCCL reads under Docker's default seccomp profile. Without that mount the
first FSDP collective fails with
`oneCCL: ze_fd_manager.cpp:144 init_device_fds: EXCEPTION: opendir failed: could not open device directory`.

To use a subset of GPUs, set `ZE_AFFINITY_MASK` (for example `-e ZE_AFFINITY_MASK=0,1`),
the Intel counterpart of `CUDA_VISIBLE_DEVICES`.

## Post-Installation Verification

```bash
xpu-smi discovery
python -c "import torch; print('torch', torch.__version__, '| XPU', torch.xpu.is_available(), torch.xpu.device_count())"
python -c "from verl.utils.device import get_device_name, get_nccl_backend; print('verl', get_device_name(), get_nccl_backend())"
python -c "from vllm.platforms import current_platform; print('vllm', current_platform.device_type)"
python -c "from vllm_omni.platforms import current_omni_platform; print('vllm-omni', current_omni_platform.device_type)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```

verl should report `xpu xccl`, and both vLLM and vLLM-Omni should report `xpu`.

## Required overrides

Intel GPU runs currently need the overrides below on top of a recipe. Each
script forwards extra Hydra overrides, so append them to the command.

### Sleep mode

Sleep mode is not supported on Intel GPU yet. vLLM-Omni's diffusion worker uses
vLLM's CUDA memory allocator for it, so with the default config the rollout
server fails at model load with `AssertionError: cumem allocator is not available`.
The fix, [vllm-project/vllm-omni#5485](https://github.com/vllm-project/vllm-omni/pull/5485),
is waiting to be merged. Until then, turn sleep mode off:

```bash
actor_rollout_ref.rollout.free_cache_engine=False \
+actor_rollout_ref.rollout.enable_sleep_mode=False
```

The rollout weights then stay on the GPU while the actor trains, which limits
the model size a colocated run can fit.

### Attention backend

Only the native/SDPA pair has been validated on Intel GPU. Recipes that select
FlashAttention (`_flash_3_varlen_hub` / `FLASH_ATTN_3_HUB`) need:

```bash
actor_rollout_ref.model.attn_backend=native \
actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA
```

## Current status

Validated with the tiny-random GPU smoke tests on Arc Pro B60 and B70: FlowGRPO
(FSDP and FSDP2, rollout TP 1 and 2, rule-based and vLLM-served reward models),
DiffusionNFT, online DPO, and BAGEL FlowGRPO. Full-size recipes have not been
validated yet.

Not yet supported on Intel GPU:

* The VeOmni engine backend.
* Omni-modality training (`main_omni`, for example Qwen3-Omni and Qwen3-TTS).

## Build Your Own Docker Image

* Intel GPU Dockerfile: [`docker/Dockerfile.xpu`](https://github.com/verl-project/verl-omni/blob/main/docker/Dockerfile.xpu)

## Example: FlowGRPO smoke test in Docker

This runs the repository's FlowGRPO smoke test with tiny random models: rollout
with vLLM-Omni, a rule-based reward plus an OCR reward served by vLLM, and two
LoRA training steps.

Inside the container from [Run](#run):

```bash
uv pip install -e ".[ocr]"
python3 tests/special_e2e/build_qwen_image_tiny_random.py --output-dir ~/models/tiny-random/Qwen-Image
hf download tiny-random/qwen3-vl --local-dir ~/models/tiny-random/qwen3-vl

NUM_GPUS=2 bash tests/special_e2e/run_flowgrpo_qwen_image.sh \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.enable_sleep_mode=False
```

The script prints per-step metrics and ends with
`FlowGRPO diffusion e2e test passed (training completed successfully).`
