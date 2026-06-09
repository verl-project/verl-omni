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
| Diffusers Flash Attention 3 backend | `uv pip install kernels==0.14.1` | Using `attn_backend="_flash_3_varlen_hub"` for faster attention |
| VeOmni engine backend | See [Optional engine backends](#optional-engine-backends) | Running the diffusion trainer with VeOmni instead of the default FSDP2 |

### Flash Attention 3 (`_flash_3_varlen_hub`)

Set `actor_rollout_ref.model.attn_backend="_flash_3_varlen_hub"` in your
training script to switch from the default `native` attention to a
Flash-Attention-3-based backend. This requires the `kernels` package:

```bash
uv pip install kernels==0.14.1
```

## Optional engine backends

VeRL-Omni defaults to **FSDP2** as the training engine for the policy and reference models. The diffusion trainer can alternatively be switched to [**VeOmni**](https://github.com/ByteDance-Seed/VeOmni). The engine is selected at the Hydra command line — see [`examples/flowgrpo_trainer/run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/run_qwen_image_ocr_veomni.sh) for a complete recipe.

### Installing VeOmni alongside vLLM 0.20.2

VeOmni 0.1.11's `gpu` extra pins `torch==2.9.1+cu129`, which conflicts with `vllm==0.20.2` (depends on `torch>=2.11`). A plain `uv pip install veomni[gpu,dit]==0.1.11` therefore fails dependency resolution.

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

If you want VeOmni's full `[gpu,dit]` extras (flash-attn variants, liger-kernel, cuda-python, etc.), install them in a separate environment not pinned to vllm 0.20.2; verl-omni does not need them.

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
