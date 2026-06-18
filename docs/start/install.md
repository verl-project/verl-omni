# Installation

Last updated: 06/15/2026

## Requirements

For NVIDIA GPU:
- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.8

For Ascend NPU:
- **Python**: Version >= 3.10
- **CANN**: Version >= 8.5.0

## Install

```bash
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
```

1. Create a Python virtual environment:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

2. Install the platform backend.

For NVIDIA GPU:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
```

It will install `vllm` for the CUDA PyTorch stack and `kernels` for the actor FA3 backend.

For Ascend NPU:

```bash
uv pip install vllm==0.22.0
uv pip install "vllm-ascend @ git+https://github.com/vllm-project/vllm-ascend.git@bb4d0776eee8fc45c3484a45c971a7049f1a2bbf"
```

3. Install VeRL-Omni:

```bash
uv pip install -e ".[vllm-omni,train]"
```

It will install vllm-omni, verl, and verl-omni.

### Extras

| Extra | Adds | When |
|---|---|---|
| `gpu` | `vllm==0.22.0`, `kernels==0.14.1`, `liger-kernel` | CUDA rollout + actor FA3 |
| `vllm-omni` | `vllm-omni==0.22.0` | vLLM-Omni rollout |
| `train` | `verl==0.8.0` | RL training |
| `dev` | `pytest`, `pre-commit`, `Levenshtein`, … | Local development / CI |
| `ocr` | `Levenshtein` | OCR reward (FlowGRPO) |

## Optional Dependencies

| Extra | Install | When needed |
|---|---|---|
| OCR reward | `uv pip install -e ".[ocr]"` | FlowGRPO training with OCR-based reward |
| Dev tools | `uv pip install -e ".[dev]"` | Linting and unit tests |
| VeOmni engine backend | See [Optional engine backends](#optional-engine-backends) | VeOmni instead of default FSDP2 |

### Flash Attention 3

The `gpu` extra pulls `kernels==0.14.1` for the Diffusers **actor** FA3 backend. Rollout FA3 comes from `vllm-omni` (`fa3-fwd`), not from `kernels`.

If FA3 deps are missing at runtime, training falls back to native/SDPA automatically. NPU recipes override with `actor_rollout_ref.model.attn_backend=_native_npu`.

## Optional engine backends

VeRL-Omni defaults to **FSDP2** as the training engine for the policy and reference models. The diffusion trainer can alternatively be switched to [**VeOmni**](https://github.com/ByteDance-Seed/VeOmni). The engine is selected at the Hydra command line — see [`examples/flowgrpo_trainer/run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/run_qwen_image_ocr_veomni.sh) for a complete recipe.

### Installing VeOmni alongside vLLM 0.22.0

VeOmni 0.1.11's `gpu` extra pins `torch==2.9.1+cu129`, which may conflict with the torch version pulled in by `vllm==0.22.0`. A plain `uv pip install veomni[gpu,dit]==0.1.11` therefore fails dependency resolution.

VeOmni itself runs correctly on torch 2.11 — only the `[gpu]` extra's pin is too strict. Install it without dependency resolution so the existing torch/vllm stack is preserved, and add the small set of runtime extras that the verl-omni VeOmni engine actually needs:

```bash
uv pip install veomni==0.1.11 --no-deps
uv pip install torchcodec librosa soundfile av
```

Verify the engine is importable:

```bash
python -c "import veomni; print('veomni', veomni.__version__)"
python -c "from veomni.distributed.offloading import load_model_to_gpu, load_optimizer, offload_model_to_cpu, offload_optimizer; print('VeOmni offloading helpers OK')"
```

If you want VeOmni's full `[gpu,dit]` extras (flash-attn variants, liger-kernel, cuda-python, etc.), install them in a separate environment not pinned to vllm 0.22.0; verl-omni does not need them.

### Older CUDA drivers (CUDA 12.8, driver 535)

The `gpu` extra installs `vllm`, whose prebuilt wheels are compiled targeting CUDA >= 12.9 by default. On hosts with CUDA 12.8 / NVIDIA driver 535, importing the stack can fail with:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12080).
```

The recommended workaround is NVIDIA CUDA forward compatibility, using a conda/micromamba environment (the `cuda-compat` package is distributed on conda-forge): install the [`cuda-compat`](https://anaconda.org/conda-forge/cuda-compat) package, version >= 12.9.0, into the environment, then point the loader at it before running:

```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/cuda-compat:$LD_LIBRARY_PATH
```

If you followed the `uv venv` flow above rather than conda, replace `${CONDA_PREFIX}/cuda-compat` with the directory where `cuda-compat` was installed.

CUDA forward compatibility works only with the Proprietary (closed) NVIDIA kernel module, not the Open kernel module, and is limited to select datacenter/professional GPUs. This mirrors [vLLM's guidance for older CUDA drivers](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#running-on-systems-with-older-cuda-drivers).

If forward compatibility cannot be used, for example with the Open kernel module, a community-validated but unofficial alternative is to build vLLM from source against the local CUDA 12.8 toolkit. Set `CUDA_HOME` to the local CUDA install and pass `--no-binary vllm` / `--no-build-isolation` to the install command; see [issue #133](https://github.com/verl-project/verl-omni/issues/133) for the full reference script.

A maintainer-provided alternative is to use an NVIDIA Docker image with the appropriate driver compatibility.

## Post-Installation Verification

For NVIDIA GPU:

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import vllm_omni; print('vllm-omni OK')"
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
