# MiniMax H3 T2VA FlowGRPO

Last updated: 08/23/2026

This recipe trains `MiniMaxAI/MiniMax-H3` LoRA adapters with FlowGRPO for
text-to-audio-video (T2VA) generation. The provided launchers configure a
Diffusers H3 Actor and vLLM-Omni rollout for joint video and audio generation,
with CLAP and ImageBind as the default alignment rewards.

The launchers support NVIDIA GPUs and Ascend NPUs. FL2VA and Ref2VA training
are not yet supported by the FlowGRPO adapters.

## Install

Follow the project [installation guide](../../../docs/start/install.md). In
particular, install the platform backend, the repository-pinned vLLM-Omni
revision, and the training dependencies in that order. Run the commands below
from the verl-omni repository root.

For NVIDIA GPU:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
```

For Ascend NPU:

```bash
uv pip install vllm==0.27.0
uv pip install "vllm-ascend @ git+https://github.com/vllm-project/vllm-ascend.git@$(cat .github/vllm_ascend_pin.txt)"
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
```

Install the tested Diffusers revision that provides
`MiniMaxH3Transformer3DModel`:

```bash
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@d6726f38a0c5ca6c06a8f227fb7bade3486ed98d"
```

## Prepare the checkpoint

Download the complete MiniMax H3 repository rather than only one subfolder:

```bash
export MODEL_ROOT="$HOME/models/MiniMax-H3"

huggingface-cli download MiniMaxAI/MiniMax-H3 \
  --local-dir "$MODEL_ROOT"
```

The recipe uses two representations from that download:

```text
MiniMax-H3/
|-- FL2VA/             # vLLM-Omni T2VA/FL2VA rollout pipeline
|   `-- transformer/
`-- transformer/       # Diffusers Actor weights and config
```

Set the corresponding paths before launching:

```bash
export MODEL_PATH="$MODEL_ROOT/FL2VA"
export ACTOR_CONFIG_PATH="$MODEL_ROOT/transformer"
```

The scripts derive `ACTOR_CONFIG_PATH` as `$(dirname "$MODEL_PATH")/transformer`
when it is not set explicitly. Do not replace the official `FL2VA/transformer`
with a symlink to the root Diffusers transformer; rollout and Actor loading use
different checkpoint layouts.

## Prepare the data

T2VA uses prompt-only data and reuses the MiniMax H3 DiffusionNFT converter.
Prepare an input directory containing either:

- `train.txt` and `test.txt`, with one prompt per line; or
- `train.jsonl` and `test.jsonl`, with a `prompt`, `text`, or `caption` field.

Convert the splits to verl-omni parquet files:

```bash
export RAW_PROMPT_DIR=/path/to/raw_prompts
export DATA_DIR="$HOME/data/vid_prompt/verl_omni"

python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2av_data.py \
  --input_dir "$RAW_PROMPT_DIR" \
  --output_dir "$DATA_DIR"
```

This writes `$DATA_DIR/train.parquet` and `$DATA_DIR/test.parquet`, the paths
consumed by both launchers. Use `--train_size` or `--val_size` to create a
smaller debugging dataset.

## Install reward dependencies

The provided launchers enable both CLAP and ImageBind rewards. Install their
dependencies before training. CLAP uses `transformers` and `torchaudio`, which
are included in the standard training environment. Its default checkpoint is
downloaded from `laion/larger_clap_general` unless `CLAP_MODEL_PATH` points to
a local copy.

ImageBind is distributed separately under the CC-BY-NC-SA 4.0 non-commercial
license. Install it and its video dependency separately:

```bash
uv pip install 'git+https://github.com/facebookresearch/ImageBind.git'
uv pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'
```

Download `imagebind_huge.pth` and set its location:

```bash
export IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth
```

By default, CLAP runs on `$REWARD_DEVICE:0` and ImageBind on
`$REWARD_DEVICE:1`, where `REWARD_DEVICE` is `cuda` for the GPU launcher and
`npu` for the NPU launcher. Both devices must be visible to the reward worker.


## Launch

### NVIDIA GPU

```bash
MODEL_PATH="$MODEL_ROOT/FL2VA" \
DATA_DIR="$HOME/data/vid_prompt/verl_omni" \
IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

The GPU launcher uses Actor `_flash_3_varlen_hub` and rollout
`FLASH_ATTN_3_HUB`. On hardware without FA3 support, append compatible Hydra
overrides:

```bash
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh \
  actor_rollout_ref.model.attn_backend=native \
  actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA
```

### Ascend NPU

```bash
MODEL_PATH="$MODEL_ROOT/FL2VA" \
DATA_DIR="$HOME/data/vid_prompt/verl_omni" \
IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_npu.sh
```

The NPU launcher uses Actor `_native_npu`, rollout `TORCH_SDPA`, and Actor
parameter and optimizer offload. It sources the Ascend toolkit and ATB
environments from `ASCEND_HOME_PATH`, which defaults to
`/usr/local/Ascend/ascend-toolkit`.

Both launchers default to online W&B logging. Set `WANDB_MODE=offline` to keep
metrics local. Checkpoints and logs are written under
`outputs/<launcher-name>/` unless `OUTPUT_DIR` is set.

## Default configuration

| Setting | Default |
| --- | --- |
| Devices | 8 GPU / 16 NPU |
| Rollout DiT TP | 2 GPU / 4 NPU |
| Text-encoder TP | Same as rollout TP |
| Training batch size | 32 |
| PPO mini-batch / per-device micro-batch | 16 / 1 |
| Rollouts per prompt | 8 |
| LoRA rank / alpha | 64 / 128 |
| Learning rate | `3e-4` |
| Training output | `256x384`, 121 frames at 24 FPS |
| Validation output | `512x768`, 121 frames at 24 FPS, 40 inference steps |
| Rollout inference steps | 10 |
| CPS window | 3 contiguous transitions from `[0, 8)` |
| Total training steps | 100 |

`NUM_GPUS` must be divisible by `ROLLOUT_TP`. `TEXT_ENCODER_TP` cannot exceed
`ROLLOUT_TP`; H3 supports text-encoder TP sizes 1, 2, 4, and 8. The recipe uses
an Actor micro-batch of 1 because samples with different packed
video/audio/text layouts cannot share one H3 forward. A larger micro-batch is
valid only when every sample has the same packed layout.

MiniMax H3 requires a named `ASPECT_RATIO`, one of `21:9`, `16:9`, `4:3`,
`1:1`, `3:4`, or `9:16`. The explicit height and width select the generated
canvas and must be multiples of 32; the provided launchers use `256x384` with
`ASPECT_RATIO=16:9`.

Common environment overrides are:

| Variable | Purpose |
| --- | --- |
| `WORKSPACE` | Base directory for default model and data paths |
| `MODEL_PATH` | Official `MiniMax-H3/FL2VA` rollout pipeline |
| `ACTOR_CONFIG_PATH` | Root Diffusers Actor weights and config directory |
| `DATA_DIR` | Directory containing `train.parquet` and `test.parquet` |
| `OUTPUT_DIR` | Checkpoint and log root |
| `NUM_GPUS` | Devices per node |
| `ROLLOUT_TP` | vLLM-Omni DiT tensor parallel size |
| `TEXT_ENCODER_TP` | H3 text-encoder tensor parallel size |
| `REWARD_NUM_WORKERS` | Number of reward workers |
| `REWARD_DEVICE` | Reward device type, such as `cuda` or `npu` |
| `CLAP_MODEL_PATH` | CLAP model ID or local path |
| `IMAGEBIND_MODEL_PATH` | Local ImageBind checkpoint path |
| `ASPECT_RATIO` | Named H3 canvas ratio |
| `HEIGHT` | Training output height |
| `WIDTH` | Training output width |
| `NUM_FRAMES` | Training and validation frame count |
| `INFER_STEPS` | Training rollout inference steps |
| `VAL_HEIGHT` | Validation output height |
| `VAL_WIDTH` | Validation output width |
| `TOTAL_TRAINING_STEPS` | Number of trainer steps |

Extra Hydra overrides may be appended to either launcher command.

## Current limitations

- Only T2VA rollout and training are supported. FL2VA and Ref2VA are TODOs.
- CLAP and ImageBind are required by the provided launchers; change the reward
  configuration explicitly if either reward is unavailable.
