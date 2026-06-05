# Installation

Last updated: 06/05/2026

## Requirements

For NVIDIA GPU:
- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.8

For Ascend NPU:
- **Python**: Version >= 3.10
- **CANN**: Version >= 8.5.0

## Install

Follow the steps below in order to avoid dependency conflicts:

1. Create a Python virtual environment:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

2. Install `vllm` followed by `vllm-omni`.

For NVIDIA GPU:

```bash
uv pip install vllm==0.20.2
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@c7178d89bb7a70817f239febc84c3b21a714dae7"
```

For Ascend NPU:

```bash
uv pip install "vllm @ git+https://github.com/vllm-project/vllm.git@releases/v0.20.2"
uv pip install "vllm-ascend @ git+https://github.com/vllm-project/vllm-ascend.git@07f6fec2aa4404e1283c4cd6c0981aa878bc5be9"
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@c7178d89bb7a70817f239febc84c3b21a714dae7"
```

3. Install `verl` followed by `verl-omni` from source:

```bash
# Install verl
uv pip install "verl==0.8.0"

# Install verl-omni from source
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
uv pip install -e .
```

> Note: Install `vllm` and `vllm-omni` first, as they may override your existing PyTorch installation. Installing them before `verl` and `verl-omni` ensures a compatible, hardware-aware PyTorch version.

## Optional Dependencies

| Extra | Install | When needed |
|---|---|---|
| OCR reward | `uv pip install Levenshtein` | FlowGRPO training with OCR-based reward |

## Post-Installation Verification

For NVIDIA GPU:

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```

For Ascend NPU:

```bash
python -c "import torch; import torch_npu; print('torch', torch.__version__, '| NPU', torch.npu.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```
