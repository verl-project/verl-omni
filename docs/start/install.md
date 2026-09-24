# Installation

Last updated: 09/22/2026

For Ascend NPU, see the {doc}`NPU installation guide <install_npu>`. For AMD GPU, see the {doc}`ROCm installation guide <install_rocm>`.

## Requirements

* **Python**: Version >= 3.11
* **CUDA**: Version >= 12.8

## Install

```bash
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
```

1. Create a Python virtual environment

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

2. Install the platform backend

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
```

This installs `vllm` for the CUDA PyTorch stack and `kernels` for FA3 backend.

3. Install vLLM-Omni and VeRL-Omni

```bash
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train]"
```

This installs `vllm-omni`, then `verl` and `verl-omni`.

### Extras

| Extra       | Adds                                                          | When                     |
| ----------- | ------------------------------------------------------------- | ------------------------ |
| `gpu`       | `vllm==0.28.0`, `kernels==0.16.0`, `liger-kernel`, `pyzmq`, `qwen-vl-utils` | CUDA rollout + actor FA3 |
| `vllm-omni` | `vllm-omni==0.28.0rc1`                                        | Optional PyPI baseline only; CI/docs use the git pin above |
| `train`     | `verl` @ [`.github/verl_pin.txt`](../../.github/verl_pin.txt) | RL training              |
| `dev`       | `pytest`, `pre-commit`, `Levenshtein`, …                      | Local development / CI   |
| `ocr`       | `Levenshtein`                                                 | OCR reward (FlowGRPO)    |

## Optional Dependencies

| Extra                 | Install                                                   | When needed                             |
| --------------------- | --------------------------------------------------------- | --------------------------------------- |
| OCR reward            | `uv pip install -e ".[ocr]"`                              | FlowGRPO training with OCR-based reward |
| Dev tools             | `uv pip install -e ".[dev]"`                              | Linting and unit tests                  |
| VeOmni engine backend | See [Optional engine backends](#optional-engine-backends) | VeOmni instead of default FSDP2         |

### Flash Attention 3

The `gpu` extra pulls `kernels==0.16.0` for Diffusers actor FA3 (`attn_backend=_flash_3_varlen_hub`).
Defaults pair actor and rollout on the same Hub kernel backend:

```bash
actor_rollout_ref.model.attn_backend=_flash_3_varlen_hub
actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN_3_HUB
```

`FLASH_ATTN_3_HUB` is provided by vLLM-Omni (`kernels-community/flash-attn3`). The legacy
`FLASH_ATTN` rollout path still uses local FA packages (`fa3-fwd` / `flash-attn`).

If FA deps are missing or broken at runtime, requesting an FA2/FA3 backend fails fast instead of silently downgrading to native/SDPA. Fix the install or select `native` / `TORCH_SDPA` explicitly.

On older GPUs, prefer FA2 over the FA3 default — both use the same `kernels` Hub path, so nothing extra to install:

```bash
actor_rollout_ref.model.attn_backend=flash_varlen_hub
actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN_HUB
```

### Flash Attention 2 (omni trainer)

The omni trainer's actor is a transformers LLM; following verl's practice for LLM training, it defaults to `flash_attention_2`, which requires the local `flash-attn` package — see verl's [installation docs](https://verl.readthedocs.io/en/latest/start/install.html).

## Optional engine backends

VeRL-Omni defaults to **FSDP2** as the training engine for the policy and reference models. The diffusion trainer and Qwen3-Omni Thinker can alternatively use [**VeOmni**](https://github.com/ByteDance-Seed/VeOmni). The engine is selected at the Hydra command line — see [`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh) for a complete recipe.

### Installing VeOmni alongside vLLM 0.28.0

VeOmni 0.1.12's `gpu` extra pins `torch==2.11.0+cu130`, which conflicts with the `torch==2.13.0` pulled in by `vllm==0.28.0`. A plain `uv pip install veomni[gpu]==0.1.12` therefore fails dependency resolution.

Install the base VeOmni package without dependency resolution so the existing
torch/vLLM stack is preserved, then add its media runtime dependencies
(CI runs the image-only VeOmni smoke and pulls `librosa`/`soundfile`/`av`/`audioread`
from the `[audio]`/`[omni]` extras; `torchcodec` is only needed when the VeOmni
engine decodes video). This is the shared CI install; Qwen3-Omni Thinker also
needs the kernels listed below.

```bash
uv pip install veomni==0.1.12 --no-deps
uv pip install torchcodec librosa soundfile av audioread
```

Verify the engine is importable:

```bash
python -c "import veomni; print('veomni', veomni.__version__)"
python -c "from veomni.distributed.offloading import load_model_to_gpu, load_optimizer, offload_model_to_cpu, offload_optimizer; print('VeOmni offloading helpers OK')"
```

The two-GPU Thinker backend check and two-step V1 smoke pass with torch 2.13
and vLLM 0.28, including forward, backward, optimizer updates, EP weight export
and full-weight rollout synchronization. These checks use tiny random weights;
they do not validate the full 30B checkpoint or convergence. The base
import/offloading checks above do not exercise these training paths.
The complete `veomni[gpu]` extra belongs in a separate environment compatible
with its torch pin; on the vLLM stack, install the required kernels individually.

### Qwen3-Omni Thinker kernel dependencies

The [VeOmni GSPO launcher](../../examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh)
uses the following native VeOmni defaults on every training node:

| Selector | Required package |
| --- | --- |
| `flash_attention_2` | Local `flash-attn` (FA2), built for the installed torch/CUDA ABI |
| `liger_kernel` for CE, RMSNorm, SwiGLU and RoPE | `liger-kernel` (already included in verl-omni's `[gpu]` extra) |
| `fused_triton` MoE and `triton` load-balancing loss | `triton` from the installed torch stack; kernels ship in VeOmni |

`kernels` Hub attention and vLLM's internal attention package do not supply the
local `flash_attn` import used by VeOmni. The shared CI install and Docker
`veomni` target do not install local FA2; add it before using this recipe.
FA3, FA4 and Quack are not required by these defaults.

Start from the installed torch/vLLM environment and constrain its existing
packages, including torch, Triton and CUDA runtime packages, while adding the
missing dependencies:

```bash
uv pip freeze --exclude-editable > /tmp/verl-omni-thinker-constraints.txt
uv pip install -c /tmp/verl-omni-thinker-constraints.txt \
    liger-kernel ninja packaging psutil setuptools wheel
```

For FA2, use a wheel matching the installed torch, CUDA, Python and platform,
or build from source against that environment. A source build requires a CUDA
development toolkit (`nvcc`) compatible with the installed torch build:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
uv pip install -c /tmp/verl-omni-thinker-constraints.txt \
    --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn
```

These constraints prevent the added packages from replacing the installed
stack; an incompatibility should fail resolution rather than change torch.
The backend check passed on two GB200 GPUs with torch 2.13.0+cu130, locally
built flash-attn 2.8.3.post1, liger-kernel 0.8.3, Triton 3.7.1 and Transformers
5.14.1 through normal package initialization. This uses a tiny random
checkpoint, not the full 30B model. Do not install a torch 2.11 FA2 wheel into
a torch 2.13 environment.

Run these checks on each training node. They load the FA2 extension and resolve
VeOmni's native operator implementations, beyond merely importing `veomni`:

```bash
python - <<'PY'
import torch
import triton
from flash_attn import flash_attn_func, flash_attn_varlen_func
from veomni.arguments import OpsImplementationConfig
from veomni.ops import apply_ops_config

apply_ops_config(OpsImplementationConfig(moe_implementation="fused_triton"))
print("Thinker kernels imported:", torch.__version__, torch.version.cuda, triton.__version__)
PY
python -c "import vllm_omni; import verl_omni; print('Full package imports OK')"
```

Imports do not exercise GPU execution. Before treating the stack as validated,
run both the backend check and the complete V1 smoke through normal package
initialization, as described in the [omni integration guide](../contributing/integrating_an_omni_model.md#veomni-backend-optional).
The smoke allows 1800 seconds for rollout startup because a cold FlashInfer
kernel build on GB200 can exceed vLLM-Omni's default 600-second timeout.

## Post-Installation Verification

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import vllm_omni; print('vllm-omni OK')"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```

## Build Your Own Docker Image

CUDA Dockerfile: [`docker/Dockerfile.cuda`](https://github.com/verl-project/verl-omni/blob/main/docker/Dockerfile.cuda)

The CUDA image is intended for NVIDIA GPU training and rollout. The default CUDA base image uses **CUDA 13.0.2** on Ubuntu 22.04. You can override the CUDA version with `--build-arg CUDA_VERSION=...` if needed.

Build context is controlled by the repo-root [`.dockerignore`](https://github.com/verl-project/verl-omni/blob/main/.dockerignore); keep large local folders such as `.venv`, `data/`, and `checkpoints/` out of the context.

Ascend NPU images are documented in the {doc}`NPU installation guide <install_npu>`.

## CUDA Docker Image

### Prerequisites

* Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Build commands

From the repository root:

```bash
# Standard GPU training image (runtime target)
docker build -f docker/Dockerfile.cuda -t verl-omni:gpu .

# OCR reward (adds the `ocr` extra / Levenshtein)
docker build -f docker/Dockerfile.cuda --target ocr -t verl-omni:gpu-ocr .

# Local development tools (adds the `dev` extra)
docker build -f docker/Dockerfile.cuda --target dev -t verl-omni:gpu-dev .
```

The image bakes in `verl_omni` and its Python dependencies. Recipe scripts under `examples/` are **not** copied into the image — mount the repository at runtime.

### Launch with interactive session for development

Start an interactive shell with GPU access, shared memory for Ray/vLLM, and common host directories mounted:

```bash
export REPO=/path/to/verl-omni          # this repository
export WORKSPACE=$HOME                  # data, checkpoints, HF cache root

docker run --gpus all --shm-size=16g -it --rm \
  --name verl-omni-ocr \
  -v "$REPO:/workspace/verl-omni" \
  -v "$WORKSPACE/data:$WORKSPACE/data" \
  -v "$WORKSPACE/checkpoints:$WORKSPACE/checkpoints" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e WORKSPACE="$WORKSPACE" \
  -e HF_HOME=/root/.cache/huggingface \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -w /workspace/verl-omni \
  verl-omni:gpu-ocr \
  /bin/bash
```

Inside the container, confirm the installation using the same checks as [Post-Installation Verification](#post-installation-verification).

Notes:

* **`--shm-size=16g`** — Ray and vLLM use shared memory; larger shared memory is needed for training.
* **Mount the repo** — training recipes live in `examples/`; mounting `$REPO` lets you edit scripts locally and run them immediately in the container.
* **`WORKSPACE`** — example scripts read datasets and write checkpoints under this path. The default is `$HOME` inside the container, i.e. `/root` unless overridden.
* **Hugging Face cache** — mounting `~/.cache/huggingface` avoids re-downloading `Qwen/Qwen-Image` and reward models on every run.

## Example: Qwen-Image FlowGRPO training in Docker

This walkthrough follows the [FlowGRPO quickstart](flowgrpo_quickstart.md) using the OCR dataset and `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`.

Use the **`ocr` image target** (`verl-omni:gpu-ocr`) so the `Levenshtein` dependency is present.

### 1. Launch the interactive container

Use the CUDA launch command above.

### 2. Prepare the OCR dataset inside the container

```bash
export WORKSPACE=${WORKSPACE:-$HOME}
mkdir -p $WORKSPACE/data/ocr

# Obtain raw train.txt / test.txt from the Flow-GRPO repo:
# https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr
# Place them under $WORKSPACE/data/ocr/, then preprocess:

python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/qwen_image
```

### 3. Optional: Set W&B credentials

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

### 4. Run FlowGRPO training

The default OCR LoRA script uses 4 GPUs by default:

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh
```

The script launches `python3 -m verl_omni.trainer.main_diffusion` with FlowGRPO + `vllm_omni` rollout and OCR reward (`compute_score_ocr`). Checkpoints are written to:

```bash
checkpoints/flow_grpo/qwen_image_ocr_lora
```
