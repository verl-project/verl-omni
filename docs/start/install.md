# Installation

Last updated: 09/24/2026

For Ascend NPU, see the {doc}`NPU installation guide <install_npu>`. For AMD GPU, see the {doc}`ROCm installation guide <install_rocm>`.

## Requirements

* **Python**: Version >= 3.11
* **CUDA**: Version >= 13.0
* **NVIDIA driver**: 580+ natively; datacenter GPUs with older drivers (535+) can use [CUDA forward compatibility](#older-nvidia-drivers-cuda-forward-compatibility) instead.

### Older NVIDIA drivers (CUDA forward compatibility)

On datacenter GPUs with a pre-CUDA-13.0 driver (535+), install NVIDIA's
`cuda-compat` forward-compatibility package and point the loader at it
(see [NVIDIA's forward-compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)):

```bash
conda create -n verl-omni python=3.12 -c conda-forge
conda activate verl-omni
conda install -c conda-forge cuda-compat

# the compat libcuda must take precedence over the driver's own
export LD_LIBRARY_PATH=${CONDA_PREFIX}/cuda-compat:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${CONDA_PREFIX}/cuda-compat:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}
```

Set both exports in every shell and launcher that runs training or rollout
(e.g. in the training script) — without them CUDA initialization fails with
"the NVIDIA driver on your system is too old". Forward compatibility is a
**datacenter-only fallback** for clusters whose driver cannot be upgraded and is
**not guaranteed** across all driver branches and workloads; for production (and
consumer GPUs) prefer a **native 580+ driver**.

## Install

```bash
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
```

1. Create a Python virtual environment — on a pre-CUDA-13.0 driver, use the conda environment from the [forward-compatibility section](#older-nvidia-drivers-cuda-forward-compatibility) above instead

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

2. Install vLLM

```bash
uv pip install vllm==0.28.0 --torch-backend=auto
```

3. Install the rollout engine and training stack

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
```

In the cuda-compat environment, pass `--python "$CONDA_PREFIX/bin/python"` and use `--torch-backend=cu130` instead of `auto`.

### Extras

| Extra       | Adds                                                          | When                     |
| ----------- | ------------------------------------------------------------- | ------------------------ |
| `gpu`       | `kernels==0.16.0`, `liger-kernel`, `cupy-cuda13x` | CUDA rollout + actor FA3 |
| `omni`      | omni-trainer runtime (`librosa`, `torchaudio`, `av`, …)       | Omni-modality training   |
| `fa2`       | `flash-attn` (source build, needs a CUDA toolkit)             | Omni trainer FA2 default |
| `audio`     | `qwen-omni-utils`, `audioread`                                 | Audio parsing (omni data) |
| `dev`       | `pytest`, `pre-commit`, …                                     | Local development / CI   |
| `ocr`       | `Levenshtein`                                                 | OCR reward               |

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

The omni trainer's actor is a transformers LLM; following verl's practice for LLM training, it defaults to `flash_attention_2`, which requires the local `flash-attn` package:

```bash
uv pip install -e ".[fa2]"
```

### Optional engine backends

VeRL-Omni defaults to FSDP2; the diffusion trainer and Qwen3-Omni Thinker can alternatively use [VeOmni]({doc}`Optional engine backends <engine_backends>`).

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

The CUDA image is intended for NVIDIA GPU training and rollout. The base image is pinned to **CUDA 13.0.2** to match the cu130-pinned Python stack; changing it requires updating the torch backend and the pyproject pins together.

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
