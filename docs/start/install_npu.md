# Installation (NPU)

Last updated: 10/20/2026

For NVIDIA GPU, see the {doc}`GPU installation guide <install>`. For AMD GPU, see the {doc}`ROCm installation guide <install_rocm>`. For Intel GPU, see the {doc}`Intel GPU installation guide <install_xpu>`.

## Requirements

* **Python**: Version >= 3.11
* **CANN**: Version == 9.1.0

## Install

Supported devices: Ascend A2, A3, and A5.

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

Install the CPU build first; otherwise later steps pull the CUDA build of `torch` from PyPI.

```bash
uv pip install --force-reinstall \
    torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu/ \
    --index-strategy unsafe-best-match
uv pip install --force-reinstall torch-npu==2.10.0.post4 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/ \
    --index-strategy unsafe-best-match
```

3. Install vLLM and vLLM-Ascend

Use `ASCEND_HOME_PATH` to select a custom CANN toolkit installation; otherwise,
the commands use the `latest` symlink. Load ATB from its own installation
directory rather than deriving its location from the toolkit path.

```bash
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh

cd ..
git clone --depth 1 -b v0.30.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty uv pip install -e ".[audio]"
uv pip uninstall -y triton || true
cd ..

# Clone instead of `git+https`: that path is under ~/.cache, which CANN's linker skips.
git clone https://github.com/vllm-project/vllm-ascend.git vllm-ascend
cd vllm-ascend
git checkout $(tr -d '[:space:]' < ../verl-omni/.github/vllm_ascend_pin.txt)
git submodule update --init --recursive
uv pip install -e . \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/ \
    --index-strategy unsafe-best-match
cd ..
uv pip install --reinstall --no-deps triton-ascend==3.2.2 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
cd verl-omni
```

4. Install vLLM-Omni and VeRL-Omni

```bash
VLLM_OMNI_TARGET_DEVICE=npu \
    uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e .
uv pip install --upgrade onnxruntime
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

## Docker

Use [`docker/Dockerfile.npu`](https://github.com/verl-project/verl-omni/blob/main/docker/Dockerfile.npu)
for all supported Ascend devices. The image includes CANN, `torch-npu`,
`vllm-ascend`, and `vllm-omni`.

### Build

Pass `BASE_IMAGE` and `SOC_VERSION` from the same row. The Dockerfile has a
single `FROM`, so the build pulls only that image. Omitting both build args
uses the A3 row.

| Device | `BASE_IMAGE` | `SOC_VERSION` |
| --- | --- | --- |
| A2 | `quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12` | `ascend910b1` |
| A3 | `quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12` | `ascend910_9391` |
| A5 | `quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12` | `ascend950dt_9582` |

Other CANN versions, operating systems, and Python versions are listed in the
[Ascend CANN container image tags](https://github.com/Ascend/cann-container-image/blob/main/supported_tags.md).
Use `quay.io/ascend/cann:<tag>` as `BASE_IMAGE`, and keep `SOC_VERSION` aligned
with the chip series in that tag.

Example for A5:

```bash
BASE_IMAGE=quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12
SOC_VERSION=ascend950dt_9582
IMAGE=verl-omni:npu-a5

docker build \
  -f docker/Dockerfile.npu \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg SOC_VERSION=$SOC_VERSION \
  -t $IMAGE \
  .
```

### Run

The host needs the Ascend driver and firmware compatible with CANN 9.1.0.
Use the command matching your machine, and adjust the `davinci` range to the
NPUs you want to expose.

A2:

```bash
DEVICES=""
for i in {0..7}; do DEVICES+=" --device=/dev/davinci$i"; done

docker run -it \
  --name verl_omni_npu \
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

A3:

```bash
DEVICES=""
for i in {0..15}; do DEVICES+=" --device=/dev/davinci$i"; done

docker run -it \
  --name verl_omni_npu \
  --network host \
  --ipc host \
  $DEVICES \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /home:/home \
  verl-omni:npu-a3 \
  bash
```

A5:

```bash
DEVICES=""
for i in {0..7}; do DEVICES+=" --device=/dev/davinci$i"; done

docker run -it \
  --name verl_omni_npu \
  --network host \
  --ipc host \
  $DEVICES \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/ummu \
  --device=/dev/uburma \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /home:/home \
  verl-omni:npu-a5 \
  bash
```

Inside the container, run the [post-installation checks](#post-installation-verification).

## Example: Qwen-Image FlowGRPO training in Docker

This walkthrough follows the [FlowGRPO NPU quickstart](flowgrpo_quickstart_npu.md) using the OCR dataset and `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh`.

Use the NPU image and the NPU-specific recipe options. NPU recipes should override the attention backend with:

```bash
actor_rollout_ref.model.attn_backend=_native_npu
```

### 1. Launch the interactive container

Use the Docker launch command above.

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
