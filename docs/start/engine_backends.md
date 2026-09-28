# Optional Engine Backends

Last updated: 09/24/2026

VeRL-Omni defaults to **FSDP2** as the training engine for the policy and reference models. The diffusion trainer and Qwen3-Omni Thinker can alternatively use [**VeOmni**](https://github.com/ByteDance-Seed/VeOmni). The engine is selected at the Hydra command line — see [`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh) for a complete recipe.

## Installing VeOmni alongside vLLM 0.28.0

VeOmni 0.1.12's `gpu` extra pins `torch==2.11.0+cu130`, which conflicts with the `torch==2.13.0` pulled in by `vllm==0.28.0`. A plain `uv pip install veomni[gpu]==0.1.12` therefore fails dependency resolution.

Install the base VeOmni package without dependency resolution so the existing
torch/vLLM stack is preserved, then add its media runtime dependencies
(CI runs the image-only VeOmni smoke and pulls `librosa`/`soundfile`/`av`/`audioread`
from the `[audio]`/`[omni]` extras; `torchcodec` is only needed when the VeOmni
engine decodes video). This is the shared CI install; Qwen3-Omni Thinker also
needs the kernels listed below.

```bash
uv pip install veomni==0.1.12 --no-deps
uv pip install torchcodec librosa soundfile av audioread
```

Verify the engine is importable:

```bash
python -c "import veomni; print('veomni', veomni.__version__)"
python -c "from veomni.distributed.offloading import load_model_to_gpu, load_optimizer, offload_model_to_cpu, offload_optimizer; print('VeOmni offloading helpers OK')"
```

The two-GPU Thinker backend check and two-step V1 smoke pass with torch 2.13
and vLLM 0.28, including forward, backward, optimizer updates, EP weight export
and full-weight rollout synchronization. These checks use tiny random weights;
they do not validate the full 30B checkpoint or convergence. The base
import/offloading checks above do not exercise these training paths.
The complete `veomni[gpu]` extra belongs in a separate environment compatible
with its torch pin; on the vLLM stack, install the required kernels individually.

## Qwen3-Omni Thinker kernel dependencies

The [VeOmni GSPO launcher](../../examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh)
uses the following native VeOmni defaults on every training node:

| Selector | Required package |
| --- | --- |
| `flash_attention_2` | Local `flash-attn` (FA2), built for the installed torch/CUDA ABI |
| `liger_kernel` for CE, RMSNorm, SwiGLU and RoPE | `liger-kernel` |
| `fused_triton` MoE and `triton` load-balancing loss | `triton` from the installed torch stack; kernels ship in VeOmni |

`kernels` Hub attention and vLLM's internal attention package do not supply the
local `flash_attn` import used by VeOmni. The shared CI install and Docker
`veomni` target do not install local FA2; add it before using this recipe.
FA3, FA4 and Quack are not required by these defaults.

Start from the installed torch/vLLM environment and constrain its existing
packages, including torch, Triton and CUDA runtime packages, while adding the
missing dependencies:

```bash
uv pip freeze --exclude-editable > /tmp/verl-omni-thinker-constraints.txt
uv pip install -c /tmp/verl-omni-thinker-constraints.txt \
    liger-kernel ninja packaging psutil setuptools wheel
```

For FA2, use a wheel matching the installed torch, CUDA, Python and platform,
or build from source against that environment. A source build requires a CUDA
development toolkit (`nvcc`) compatible with the installed torch build:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
uv pip install -c /tmp/verl-omni-thinker-constraints.txt \
    --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn
```

These constraints prevent the added packages from replacing the installed
stack; an incompatibility should fail resolution rather than change torch.
The backend check passed on two GB200 GPUs with torch 2.13.0+cu130, locally
built flash-attn 2.8.3.post1, liger-kernel 0.8.3, Triton 3.7.1 and Transformers
5.14.1 through normal package initialization. This uses a tiny random
checkpoint, not the full 30B model. Do not install a torch 2.11 FA2 wheel into
a torch 2.13 environment.

Run these checks on each training node. They load the FA2 extension and resolve
VeOmni's native operator implementations, beyond merely importing `veomni`:

```bash
python - <<'PY'
import torch
import triton
from flash_attn import flash_attn_func, flash_attn_varlen_func
from veomni.arguments import OpsImplementationConfig
from veomni.ops import apply_ops_config

apply_ops_config(OpsImplementationConfig(moe_implementation="fused_triton"))
print("Thinker kernels imported:", torch.__version__, torch.version.cuda, triton.__version__)
PY
python -c "import vllm_omni; import verl_omni; print('Full package imports OK')"
```

Imports do not exercise GPU execution. Before treating the stack as validated,
run both the backend check and the complete V1 smoke through normal package
initialization, as described in the [omni integration guide](../contributing/integrating_an_omni_model.md#veomni-backend-optional).
The smoke allows 1800 seconds for rollout startup because a cold FlashInfer
kernel build on GB200 can exceed vLLM-Omni's default 600-second timeout.
