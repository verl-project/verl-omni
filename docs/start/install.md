<!-- 2026-09-02, tianqi, split GPU/NPU install docs for issue #226; GPU extras list kernels==0.16.0 to match pyproject.toml -->
# Installation (NVIDIA GPU)

Last updated: 09/02/2026

For Ascend NPU, see the {doc}`NPU installation guide <install_npu>`.

## Requirements

* **Python**: Version >= 3.10
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
| `gpu`       | `vllm==0.27.0`, `kernels==0.16.0`, `liger-kernel`             | CUDA rollout + actor FA3 |
| `vllm-omni` | `vllm-omni==0.27.0rc1`                                        | Optional PyPI baseline only; CI/docs use the git pin above |
| `train`     | `verl` @ [`.github/verl_pin.txt`](../../.github/verl_pin.txt) | RL training              |
| `dev`       | `pytest`, `pre-commit`, `Levenshtein`, …                      | Local development / CI   |
| `ocr`       | `Levenshtein`                                                 | OCR reward (FlowGRPO)    |

## Optional Dependencies

| Extra                 | Install                                                   | When needed                             |
| --------------------- | --------------------------------------------------------- | --------------------------------------- |
| OCR reward            | `uv pip install -e ".[ocr]"`                              | FlowGRPO training with OCR-based reward |
| Multimodal training   | `pip install qwen-vl-utils math-verify`                   | Vision-language training (e.g. MMK12)   |
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

If FA3 deps are missing at runtime, training falls back to native/SDPA automatically.

## Optional engine backends

VeRL-Omni defaults to **FSDP2** as the training engine for the policy and reference models. The diffusion trainer can alternatively be switched to [**VeOmni**](https://github.com/ByteDance-Seed/VeOmni). The engine is selected at the Hydra command line — see [`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh) for a complete recipe.

### Installing VeOmni alongside vLLM 0.27.0

VeOmni 0.1.11's `gpu` extra pins `torch==2.9.1+cu129`, which may conflict with the torch version pulled in by `vllm==0.27.0`. A plain `uv pip install veomni[gpu,dit]==0.1.11` therefore fails dependency resolution.

VeOmni itself runs correctly on torch 2.11 — only the `[gpu]` extra's pin is too strict. Install it without dependency resolution so the existing torch/vllm stack is preserved, and add the small set of runtime extras that the verl-omni VeOmni engine actually needs:

```bash
uv pip install veomni==0.1.11 --no-deps
uv pip install torchcodec librosa soundfile av audioread
```

Verify the engine is importable:

```bash
python -c "import veomni; print('veomni', veomni.__version__)"
python -c "from veomni.distributed.offloading import load_model_to_gpu, load_optimizer, offload_model_to_cpu, offload_optimizer; print('VeOmni offloading helpers OK')"
```

If you want VeOmni's full `[gpu,dit]` extras (flash-attn variants, liger-kernel, cuda-python, etc.), install them in a separate environment not pinned to vllm 0.27.0; verl-omni does not need them.

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
<!-- end -->
