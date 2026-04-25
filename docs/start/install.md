# Installation

Last updated: 04/23/2026

## Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.10 |
| CUDA | >= 12.1 |
| GPU | NVIDIA GPU (≥ 24 GB VRAM recommended) |

## Install

Install in this order to avoid dependency conflicts:

```bash
# 1. vLLM and vLLM-Omni rollout backend
pip install "vllm==0.18" "vllm-omni==0.18"

# 2. verl
pip install git+https://github.com/verl-project/verl.git@3eab8ccc6143c624e7f11c871896f941b3fec900

# 3. VeRL-Omni
pip install git+https://github.com/verl-project/verl-omni.git@main
```

Note: Install vLLM and vLLM-Omni first — they may override your existing PyTorch installation,
so installing them before verl and VeRL-Omni ensures a compatible CUDA-aware torch version.

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
