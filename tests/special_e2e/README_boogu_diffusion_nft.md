# BOOGU-Image DiffusionNFT training smoke test

`run_diffusionnft_boogu_image.sh` exercises online vLLM-Omni generation,
JPEG-compressibility rewards, DiffusionNFT loss, FSDP LoRA updates, and policy
weight synchronization. It builds a tiny random checkpoint using the real
BOOGU transformer, Qwen3VL encoder, and VAE classes. This validates integration;
it does not establish reward convergence or image quality for a pretrained model.

Install the pinned stack in `docs/start/install.md`, plus `boogu-image`. Keep
the repository's Diffusers and PyTorch versions when resolving BOOGU's optional
dependencies. On minimal Ubuntu containers, install `libgl1` and
`libglib2.0-0` so OpenCV can import in spawned rollout workers. A failed OpenCV
import can otherwise surface as an `EOFError` during engine startup.

When building a CUDA 13.0 image without a GPU, use an explicit
`uv --torch-backend=cu130` for the GPU installation step. The script checks CUDA
availability and OpenCV imports before starting Ray.

Only the source checkpoint's processor and scheduler files are needed:

```bash
hf download Boogu/Boogu-Image-0.1-Base --include 'processor/*' 'scheduler/*'
NUM_GPUS=2 TOTAL_TRAIN_STEPS=4 bash tests/special_e2e/run_diffusionnft_boogu_image.sh
MODE=edit NUM_GPUS=2 TOTAL_TRAIN_STEPS=2 bash tests/special_e2e/run_diffusionnft_boogu_image.sh
```

Each prompt has two sampled responses. Use at least two GPUs to exercise FSDP
sharding; `NUM_GPUS=1` is useful for diagnostics but uses FSDP's `NO_SHARD` mode.
Both runs must finish with finite losses and gradient norms. Inspect the logs
for actual optimizer steps; successful model loading alone is insufficient.

`SOURCE_MODEL` can point to a local source checkpoint. `MODEL_PATH` controls
the generated tiny checkpoint and `DATA_DIR` controls dummy parquet output.
Additional Hydra overrides can be passed as arguments to the shell script.
