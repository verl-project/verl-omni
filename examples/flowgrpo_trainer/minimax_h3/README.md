# MiniMax H3 FL2VA FlowGRPO

This recipe trains `MiniMaxAI/MiniMax-H3` FL2VA LoRA adapters with a
Diffusers + FSDP actor, vLLM-Omni rollout, joint audio-video CPS transitions,
and the CLAP plus ImageBind rewards on GPU and Ascend NPU. It covers the FL2VA
model family, including its T2VA path.

## Install model dependencies

Install the standard verl-omni environment first. MiniMax H3 currently requires
a newer Diffusers commit than the repository-wide baseline:

```bash
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@d6726f38a0c5ca6c06a8f227fb7bade3486ed98d"
```

## Prepare model

Download MiniMax H3 and point `MODEL_PATH` to the `FL2VA` directory:

```bash
huggingface-cli download MiniMaxAI/MiniMax-H3 \
  --local-dir "$HOME/models/MiniMax-H3"
export MODEL_PATH="$HOME/models/MiniMax-H3/FL2VA"
```

The scripts derive `ACTOR_CONFIG_PATH` as `$(dirname "$MODEL_PATH")/transformer`.
Override both variables when using a different local layout.

## Prepare data

## Install reward dependencies

CLAP uses the existing `transformers` and `torchaudio` dependencies. ImageBind
is optional software under the CC-BY-NC-SA 4.0 non-commercial license:

```bash
uv pip install 'git+https://github.com/facebookresearch/ImageBind.git'
uv pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'
```

Download the ImageBind checkpoint separately and set `IMAGEBIND_MODEL_PATH` to
its location. The scripts default to `.checkpoints/imagebind_huge.pth`. Review
the ImageBind license before enabling this reward in your environment.

## Launch

### GPU

```bash
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

The GPU recipe defaults to 8 GPUs, vLLM-Omni DiT and text-encoder tensor
parallel size 4, one reward worker, Actor `_flash_3_varlen_hub`, and rollout
`FLASH_ATTN_3_HUB`. CLAP and ImageBind run on `cuda:0` and `cuda:1`,
respectively. On GPUs without the required FA3 support, override the Actor
attention backend with `native` and the rollout backend with `TORCH_SDPA`.

### Ascend NPU

```bash
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_npu.sh
```

The NPU recipe defaults to 8 NPUs, vLLM-Omni DiT and text-encoder tensor
parallel size 4, one reward worker, Actor `_native_npu`, and rollout
`TORCH_SDPA`. CLAP and ImageBind run on `npu:0` and `npu:1`, respectively.
The script sources the Ascend toolkit and ATB environments from
`ASCEND_HOME_PATH`, which defaults to `/usr/local/Ascend/ascend-toolkit`.

Both launch scripts accept `WORKSPACE`, `MODEL_PATH`, `DATA_DIR`, `OUTPUT_DIR`,
`ACTOR_CONFIG_PATH`, `NUM_GPUS`, `ROLLOUT_TP`, `TEXT_ENCODER_TP`,
`REWARD_NUM_WORKERS`, `REWARD_DEVICE`, `CLAP_MODEL_PATH`,
`IMAGEBIND_MODEL_PATH`, and `TOTAL_TRAINING_STEPS` through environment
variables. The NPU script additionally accepts `ASCEND_HOME_PATH`. Extra Hydra
overrides can be appended to either command.

`DATA_DIR` must contain `train.parquet` and `test.parquet`. `NUM_GPUS` must be
divisible by `ROLLOUT_TP`, and `TEXT_ENCODER_TP` must not exceed `ROLLOUT_TP`.
H3 supports text-encoder TP sizes `1`, `2`, `4`, and `8`. The configured reward
devices must also be visible to the reward worker.

Both scripts use a training batch size of 8, a PPO mini-batch size of 8, a
per-device micro-batch size of 1, and 100 total training steps. The GPU recipe
generates eight rollouts per prompt, while the NPU recipe generates one. The
default `256x448`, 107-frame output is intended for integration debugging
rather than final-quality generation.

For each 50-step rollout, the recipe samples three contiguous SDE transitions
from the configured range `[0, 50]` (clamped to the available transitions).
Video and audio use separate FlowMatch sigma schedules, and their mean
transition log-probabilities contribute equally to FlowGRPO training.
