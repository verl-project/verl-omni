# Installation

Last updated: 04/30/2026

## Requirements

- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.8

## Install

Install in the following order to avoid dependency conflicts:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vllm, vllm-omni first
uv pip install vllm==0.18.0
uv pip install vllm-omni==0.18

# Install verl from source at the pinned commit (editable install)
git clone https://github.com/verl-project/verl.git
cd verl
git checkout a512e90fddcefb64baaa6384e9cf8571b6bfab0b
uv pip install -e .
cd ..

# Install verl-omni from source
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
uv pip install -e .
```

Notes:

- Install vllm and vllm-omni first — they may override your existing PyTorch installation,
  so installing them before verl and verl-omni ensures a compatible CUDA-aware torch version.
- verl is intentionally installed in editable mode (`uv pip install -e .` from a clone) rather
  than via `uv pip install git+…@<commit>`. The wheel built from the pinned commit is missing
  `verl/experimental/reward_loop/router/` because the upstream directory had no `__init__.py`
  at that commit and `setuptools`' default package discovery silently dropped it, which breaks
  the FlowGRPO trainer at runtime with
  `ModuleNotFoundError: No module named 'verl.experimental.reward_loop.router'`.
  An editable install exposes the source tree directly and side-steps the issue via PEP 420
  implicit namespace packages.
  This pin will be bumped past the upstream packaging fix
  ([verl-project/verl#5209](https://github.com/verl-project/verl/pull/5209)) once verl-omni is
  also adapted to the breaking `LLMServerClient` refactor in
  [verl-project/verl#6129](https://github.com/verl-project/verl/pull/6129) (tracked separately).

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
