# Installation (NPU)

Last updated: 10/22/2026

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
Keep the selected PyTorch stack constrained in subsequent installs. Store the
constraints under the active virtual environment so the path remains valid when
changing directories.

```bash
NPU_TORCH_CONSTRAINTS="$VIRTUAL_ENV/constraints-npu-torch.txt"
cat > "$NPU_TORCH_CONSTRAINTS" <<'EOF'
torch==2.10.0+cpu
torchvision==0.25.0+cpu
torchaudio==2.10.0+cpu
torch-npu==2.10.0.post4
EOF

uv pip install --force-reinstall \
    torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
    --constraint "$NPU_TORCH_CONSTRAINTS" \
    --extra-index-url https://download.pytorch.org/whl/cpu/ \
    --index-strategy unsafe-best-match
uv pip install --force-reinstall torch-npu==2.10.0.post4 \
    --constraint "$NPU_TORCH_CONSTRAINTS" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/ \
    --index-strategy unsafe-best-match
```

3. Install vLLM and vLLM-Ascend

Use `ASCEND_HOME_PATH` to select a custom CANN toolkit installation; otherwise,
the commands use the `latest` symlink. Load ATB from its own installation
directory rather than deriving its location from the toolkit path.

vLLM's isolated build environment installs its own build dependencies, including
PyTorch, even when CPU PyTorch is already installed. `VLLM_TARGET_DEVICE=empty`
does not disable build isolation. Install the build tools explicitly, then use
`--no-build-isolation` to build against the selected NPU-compatible PyTorch.
Runtime constraints alone do not constrain isolated build dependencies.
PyYAML is needed for torch-npu to load when the build imports PyTorch. The
packaging/setuptools-scm bounds keep the build tools compatible with runtime
dependencies that require packaging below 26.

```bash
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh

cd ..
git clone --depth 1 -b v0.30.0 https://github.com/vllm-project/vllm.git
cd vllm
uv pip install \
    'cmake>=3.26.1' ninja 'packaging>=24.2,<26' \
    'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0,<9' \
    'setuptools-rust>=1.9.0' wheel jinja2 pyyaml
VLLM_TARGET_DEVICE=empty uv pip install --no-build-isolation -e ".[audio]" \
    --constraint "$NPU_TORCH_CONSTRAINTS"
uv pip uninstall -y triton || true
cd ..

# Clone instead of `git+https`: that path is under ~/.cache, which CANN's linker skips.
git clone https://github.com/vllm-project/vllm-ascend.git vllm-ascend
cd vllm-ascend
git checkout $(tr -d '[:space:]' < ../verl-omni/.github/vllm_ascend_pin.txt)
git submodule update --init --recursive
uv pip install -e . \
    --constraint "$NPU_TORCH_CONSTRAINTS" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/ \
    --index-strategy unsafe-best-match
cd ..
uv pip install --reinstall --no-deps triton-ascend==3.2.2 \
    --constraint "$NPU_TORCH_CONSTRAINTS" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
cd verl-omni
```

4. Install vLLM-Omni and VeRL-Omni

Keep the NPU target set for both installs, since installing verl-omni can also
build its vllm-omni dependency.

```bash
export VLLM_OMNI_TARGET_DEVICE=npu
uv pip install \
    "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)" \
    --constraint "$NPU_TORCH_CONSTRAINTS"
uv pip install -e . --constraint "$NPU_TORCH_CONSTRAINTS"
uv pip install --upgrade onnxruntime --constraint "$NPU_TORCH_CONSTRAINTS"
```

### Extras

| Extra       | Adds                                                          | When                     |
| ----------- | ------------------------------------------------------------- | ------------------------ |
| `dev`       | `pytest`, `pre-commit`, …                                     | Local development / CI   |
| `ocr`       | `Levenshtein`                                                 | OCR reward               |

The CUDA `gpu` extra is not used on NPU. NPU recipes override the attention backend with `actor_rollout_ref.model.attn_backend=_native_npu`.

## Post-Installation Verification

```bash
python -c "import torch; assert torch.version.cuda is None, torch.version.cuda"
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

Set `NPU_DEVICE` to `a2`, `a3` (default), or `a5`. The Dockerfile selects the
base image and compiler target automatically.

```bash
NPU_DEVICE=a5

docker build \
  -f docker/Dockerfile.npu \
  --build-arg NPU_DEVICE=$NPU_DEVICE \
  -t verl-omni:npu-$NPU_DEVICE \
  .
```

### Run

The host needs the Ascend driver and firmware compatible with CANN 9.1.0.
Set `NPU_DEVICE` to the image you built and `NPU_DEVICE_IDS` to the devices
to expose, as listed by `npu-smi info`.

```bash
NPU_DEVICE=a5
NPU_DEVICE_IDS=(0 1 2 3 4 5 6 7)
DEVICES=()
for id in "${NPU_DEVICE_IDS[@]}"; do
  DEVICES+=(--device="/dev/davinci${id}")
done
for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [[ -e "$device" ]]; then
    DEVICES+=(--device="$device")
  fi
done

docker run -it --rm \
  --name verl_omni_npu \
  --network host \
  --ipc host \
  "${DEVICES[@]}" \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /mnt/data:/mnt/data \
  verl-omni:npu-$NPU_DEVICE \
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
