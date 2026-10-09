# Installation (NPU)

Last updated: 10/08/2026

For NVIDIA GPU, see the {doc}`GPU installation guide <install>`. For AMD GPU, see the {doc}`ROCm installation guide <install_rocm>`.

## Requirements

* **Python**: Version >= 3.11
* **CANN**: Version == 9.1.0

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

2. Install PyTorch

```bash
uv pip install --force-reinstall \
    torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu/
uv pip install --force-reinstall torch-npu==2.10.0.post4 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/
```

3. Install vLLM and vLLM-Ascend

```bash
source ${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}/set_env.sh
source ${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}/../nnal/atb/set_env.sh

git clone --depth 1 -b v0.28.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty uv pip install -e ".[audio]"
uv pip uninstall -y triton || true
cd ..

uv pip install "vllm-ascend @ git+https://github.com/vllm-project/vllm-ascend.git@$(cat .github/vllm_ascend_pin.txt)"
uv pip uninstall -y triton triton-ascend || true
uv pip install triton-ascend==3.2.2 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

4. Install vLLM-Omni and VeRL-Omni

```bash
VLLM_OMNI_TARGET_DEVICE=npu \
    uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e .
```

5. Reinstall PyTorch

```bash
uv pip install --force-reinstall \
    torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu/
uv pip install --force-reinstall torch-npu==2.10.0.post4 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/
```

### Extras

| Extra       | Adds                                                          | When                     |
| ----------- | ------------------------------------------------------------- | ------------------------ |
| `dev`       | `pytest`, `pre-commit`, …                                     | Local development / CI   |
| `ocr`       | `Levenshtein`                                                 | OCR reward               |

The CUDA `gpu` extra is not used on NPU. NPU recipes override the attention backend with `actor_rollout_ref.model.attn_backend=_native_npu`.

## Post-Installation Verification

```bash
python -c "from importlib.metadata import version; import torch, torch_npu; print('torch', torch.__version__, '| torch-npu', version('torch-npu'), '| NPU', torch.npu.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import triton; from triton.backends import backends; print('triton', triton.__version__, sorted(backends))"
python -c "from importlib.metadata import version; import vllm_ascend; print('vllm-ascend', version('vllm-ascend'))"
python -c "from importlib.metadata import version; import vllm_omni; print('vllm-omni', version('vllm-omni'))"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```

## Build Your Own Docker Image

* Ascend Atlas A2 NPU Dockerfile: [`docker/Dockerfile.a2.npu`](https://github.com/verl-project/verl-omni/blob/main/docker/Dockerfile.a2.npu)
* Ascend Atlas A3 NPU Dockerfile: [`docker/Dockerfile.a3.npu`](https://github.com/verl-project/verl-omni/blob/main/docker/Dockerfile.a3.npu)

The NPU images are split by Ascend hardware generation: `Dockerfile.a2.npu` is intended for Ascend 910B / Atlas A2, and `Dockerfile.a3.npu` is intended for Ascend Atlas A3. Both NPU images include CANN, `torch-npu`, `vllm-ascend`, and `vllm-omni`.

The `torch` and `torch-npu` versions in these Dockerfiles are the currently
validated image defaults, not a universal VeRL-Omni requirement. Each
Dockerfile installs the selected pair before the engine stack, then re-applies
the same pair after installing `vllm`, `vllm-ascend`, `vllm-omni`, and `verl`.
This prevents their dependency resolvers from leaving the final image with a
mixed PyTorch stack. When changing PyTorch versions, update both alignment
steps together and keep the pair compatible with the image's CANN version.

Build context is controlled by the repo-root [`.dockerignore`](https://github.com/verl-project/verl-omni/blob/main/.dockerignore); keep large local folders such as `.venv`, `data/`, and `checkpoints/` out of the context.

NVIDIA GPU images are documented in the {doc}`GPU installation guide <install>`.

## Ascend NPU Docker Image

### Prerequisites

The Ascend NPU Docker image expects the host machine to provide the Ascend driver and device files.

Before launching the container, make sure the host has:

* Ascend driver installed.
* CANN-compatible runtime environment.
* `npu-smi` available on the host.
* Ascend device nodes under `/dev`, such as `/dev/davinci0`, `/dev/davinci_manager`, `/dev/devmm_svm`, and `/dev/hisi_hdc`.
* Docker permission to pass NPU devices into the container.

The NPU container mounts the host driver directory:

```bash
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
```

This allows the containerized CANN / `torch-npu` runtime to use the host Ascend driver.

### Build commands

From the repository root, choose the Dockerfile that matches your Ascend hardware.

For Ascend Atlas A3:

```bash
docker build \
  -f docker/Dockerfile.a3.npu \
  -t verl-omni:npu-a3 \
  .
```

For Ascend Atlas A2 / 910B:

```bash
docker build \
  -f docker/Dockerfile.a2.npu \
  -t verl-omni:npu-a2 \
  .
```

When debugging dependency installation or making sure no old Docker layer is reused, add `--no-cache`:

```bash
# Atlas A3
docker build --no-cache \
  -f docker/Dockerfile.a3.npu \
  -t verl-omni:npu-a3 \
  .

# Atlas A2 / 910B
docker build --no-cache \
  -f docker/Dockerfile.a2.npu \
  -t verl-omni:npu-a2 \
  .
```

You may choose different image tags locally. If you do so, replace the image name in the `docker run` command accordingly.

### Launch on Ascend Atlas A3, 16 NPU

Use this command on a 16-card Ascend Atlas A3 machine:

```bash
DEVICES=""
for i in $(seq 0 15); do
  DEVICES="$DEVICES --device=/dev/davinci$i"
done

docker run -it --rm \
  --name verl_omni_16npu \
  --network host \
  --ipc host \
  $DEVICES \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /mnt/data:/mnt/data \
  verl-omni:npu-a3 \
  bash
```

### Launch on Ascend Atlas A2 / 910B, 8 NPU

Use this command on an 8-card Ascend Atlas A2 / 910B machine:

```bash
DEVICES=""
for i in $(seq 0 7); do
  DEVICES="$DEVICES --device=/dev/davinci$i"
done

docker run -it --rm \
  --name verl_omni_8npu \
  --network host \
  --ipc host \
  $DEVICES \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /home:/home \
  verl-omni:npu-a2 \
  bash
```

### Notes for NPU containers

* **`--network host`** — useful for Ray, distributed training, and multi-process communication.
* **`--ipc host`** — avoids shared-memory limitations during training and rollout.
* **`/dev/davinci*` devices** — expose Ascend NPU cards to the container.
* **`/dev/davinci_manager`**, **`/dev/devmm_svm`**, and **`/dev/hisi_hdc`** — required Ascend runtime device files.
* **`/usr/local/Ascend/driver`** — mounted read-only from the host so the container can use the installed Ascend driver.
* **`npu-smi`** — mounted from the host to inspect device status inside the container.
* **Atlas A3 16 NPU** — exposes `/dev/davinci0` through `/dev/davinci15`.
* **Atlas A2 / 910B 8 NPU** — exposes `/dev/davinci0` through `/dev/davinci7`.

Inside the container, confirm the NPU environment:

```bash
npu-smi info
python -c "from importlib.metadata import version; import torch, torch_npu; print('torch', torch.__version__, '| torch-npu', version('torch-npu'), '| NPU', torch.npu.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "from importlib.metadata import version; import vllm_ascend; print('vllm-ascend', version('vllm-ascend'))"
python -c "from importlib.metadata import version; import vllm_omni; print('vllm-omni', version('vllm-omni'))"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```

## Example: Qwen-Image FlowGRPO training in Docker

This walkthrough follows the [FlowGRPO NPU quickstart](flowgrpo_quickstart_npu.md) using the OCR dataset and `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh`.

Use the NPU image and the NPU-specific recipe options. NPU recipes should override the attention backend with:

```bash
actor_rollout_ref.model.attn_backend=_native_npu
```

### 1. Launch the interactive container

Use the Atlas A2 or A3 launch command above.

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

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh
```

If you run the training command manually, make sure to include the NPU-specific Hydra override:

```bash
actor_rollout_ref.model.attn_backend=_native_npu
```

The script launches `python3 -m verl_omni.trainer.main_diffusion` with FlowGRPO + `vllm_omni` rollout and OCR reward (`compute_score_ocr`). Checkpoints are written to:

```bash
checkpoints/flow_grpo/qwen_image_ocr_lora
```
