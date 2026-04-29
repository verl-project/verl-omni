# Installation

Last updated: 04/23/2026

## Requirements

- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.8

## Install

Install in the following order to avoid dependency conflicts:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vllm, vllm-omni, then verl in order
uv pip install vllm==0.18.0
uv pip install vllm-omni==0.18
uv pip install git+https://github.com/verl-project/verl.git@a512e90fddcefb64baaa6384e9cf8571b6bfab0b

# Install verl-omni from source
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
uv pip install -e .
```

Note: Install vllm and vllm-omni first — they may override your existing PyTorch installation,
so installing them before verl and verl-omni ensures a compatible CUDA-aware torch version.

## Optional Dependencies

| Extra | Install | When needed |
|---|---|---|
| OCR reward | `pip install Levenshtein` | FlowGRPO training with OCR-based reward |

## Post-Installation Verification

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```
