# Installation (ROCm)

Last updated: 09/10/2026

For NVIDIA GPU, see the {doc}`GPU installation guide <install>`. For Ascend NPU, see the {doc}`NPU installation guide <install_npu>`.

## Requirements

* **Python**: Version >= 3.11
* **ROCm**: Version >= 7.0
* **GPU**: CDNA3 (`gfx942`, MI300X / MI325X) or CDNA4 (`gfx950`, MI350X / MI355X)

## Install

On ROCm the Docker image is the supported path; there is no bare-metal `pip` install.

```bash
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
docker build -f docker/Dockerfile.rocm -t verl-omni:rocm .
```

Compiling vLLM dominates the build. It defaults to both current CDNA
generations; narrowing it to your card roughly halves the time:

```bash
docker build -f docker/Dockerfile.rocm -t verl-omni:rocm \
  --build-arg PYTORCH_ROCM_ARCH=gfx950 .
```

Build context is controlled by the repo-root [`.dockerignore`](https://github.com/verl-project/verl-omni/blob/main/.dockerignore); keep large local folders such as `.venv`, `data/`, and `checkpoints/` out of the context.

### Build arguments

| Argument            | Default                                        | Purpose                                     |
| ------------------- | ---------------------------------------------- | ------------------------------------------- |
| `BASE_IMAGE`        | `verlai/verl:rocm7.14_torch2.12_release_0724`  | ROCm PyTorch stack, vLLM checkout, hipcc    |
| `PYTORCH_ROCM_ARCH` | `gfx942;gfx950`                                | GPU architectures to compile vLLM kernels for |
| `VLLM_VERSION`      | `v0.28.0`                                      | vLLM tag built from source                  |
| `VERL_GIT_REF`      | [`.github/verl_pin.txt`](../../.github/verl_pin.txt) | verl commit to install                |

## Optional Dependencies

The base image already provides `qwen-vl-utils` for vision-language training and
`pytest`, `pytest-cov`, `pytest-asyncio`, and `pre-commit` for development, and
the build adds `math-verify` and `latex2sympy2_extended`. Only these need
installing inside the container:

| Extra      | Install                     | When needed                             |
| ---------- | --------------------------- | --------------------------------------- |
| OCR reward | `pip install Levenshtein`   | FlowGRPO training with OCR-based reward |
| Profiling  | `pip install py-spy`        | Sampling profiler for stack traces      |

`PIP_CONSTRAINT` is baked into the image, so these cannot pull a CUDA `torch` over the ROCm build.


## Run

```bash
export REPO=/path/to/verl-omni
export WORKSPACE=$HOME

mkdir -p \
  "$WORKSPACE/data" \
  "$WORKSPACE/checkpoints" \
  "$HOME/.cache/huggingface"

docker run -it --rm \
  --network=host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt=seccomp=unconfined \
  --privileged \
  --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$REPO:/workspace/verl-omni" \
  -v "$WORKSPACE/data:$WORKSPACE/data" \
  -v "$WORKSPACE/checkpoints:$WORKSPACE/checkpoints" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e WORKSPACE="$WORKSPACE" \
  -e HF_HOME=/root/.cache/huggingface \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -w /workspace/verl-omni \
  verl-omni:rocm \
  bash
```

## Post-Installation Verification

The image verifies `torch`, `vllm`, `verl`, and `vllm-omni` at build time, but
only the checks that do not touch a device. Importing `vllm_omni` or `verl_omni`
pulls in vLLM's cumem allocator (`vllm.device_allocator.cumem`), which asserts
that the GPU runtime library is loaded into the process — it is not until torch
has opened a device, so these cannot be checked during `docker build`. Run the
full set inside the container:

```bash
rocm-smi
python -c "import torch; print('torch', torch.__version__, '| HIP', torch.version.hip, '| GPU', torch.cuda.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "from importlib.metadata import version; import vllm_omni; print('vllm-omni', version('vllm-omni'))"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```

## Attention backend

Diffusion recipes default to FlashAttention 3
(`attn_backend=_flash_3_varlen_hub`), and the `kernels-community` hub repos
publish no ROCm build variant — the run fails at model init with
`FileNotFoundError: Cannot find a build variant for this system`. Nothing warns
first: `fa_available()` only checks that `kernels` is importable. Every
diffusion recipe on ROCm needs both overrides:

```bash
actor_rollout_ref.model.attn_backend=native \
actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA
```

Both are required together; `validate_attention_consistency` rejects a
mismatched pair. Omni recipes (`main_omni`) are unaffected — they select
attention via `attn_implementation`, which defaults to `flash_attention_2` and
works on ROCm using the base image's `flash_attn` build.

## Build Your Own Docker Image

* AMD GPU (ROCm) Dockerfile: [`docker/Dockerfile.rocm`](https://github.com/verl-project/verl-omni/blob/main/docker/Dockerfile.rocm)

NVIDIA GPU images are documented in the {doc}`GPU installation guide <install>`, and Ascend NPU images in the {doc}`NPU installation guide <install_npu>`.

## Example: Qwen-Image FlowGRPO training in Docker

This walkthrough uses the OCR dataset and `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`. 

### 1. Launch the interactive container

Use the `docker run` command from [Run](#run) above.

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

`Qwen/Qwen-Image` and the `Qwen/Qwen3-VL-8B-Instruct` reward model are fetched
from the Hub on first use; pre-download them with `hf download` to keep the
first step from timing out.

### 3. Optional: Set W&B credentials

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

Without a key the run aborts at logger init with
`wandb.errors.UsageError: No API key configured`. To run without W&B, append
`trainer.logger=[console]`.

### 4. Run FlowGRPO training

ROCm requires the native/SDPA attention pair described in
[Attention backend](#attention-backend); the script forwards extra Hydra
overrides, so append them to the command:

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh \
  actor_rollout_ref.model.attn_backend=native \
  actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA
```

Without those two overrides the run fails during model init with
`FileNotFoundError: Cannot find a build variant for this system in
kernels-community/flash-attn3`.

The script launches `python3 -m verl_omni.trainer.main_diffusion` with FlowGRPO
+ `vllm_omni` rollout and OCR reward (`compute_score_ocr`). Checkpoints are
written to:

```bash
checkpoints/flow_grpo/qwen_image_ocr_lora
```
