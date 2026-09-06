# MiniMax H3 T2VA, FL2VA, and Ref2VA FlowGRPO

Last updated: 09/02/2026

These recipes train `MiniMaxAI/MiniMax-H3` LoRA adapters with FlowGRPO for
text-to-audio-video (T2VA), first-frame image-to-audio-video (FL2VA), and
reference-to-audio-video (Ref2VA) generation. The launchers configure a
Diffusers H3 Actor and vLLM-Omni rollout for joint video and audio generation,
with CLAP and ImageBind as the default rewards.

T2VA supports NVIDIA GPUs and Ascend NPUs. The FL2VA and full multimodal
Ref2VA paths target NVIDIA GPUs.

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
uv pip install vllm==0.28.0
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
|-- FL2VA/             # vLLM-Omni T2VA rollout pipeline
|   `-- transformer/
|-- Ref2VA/            # vLLM-Omni Ref2VA rollout pipeline
|   `-- transformer/
|-- transformer/       # Diffusers T2VA Actor weights and config
`-- transformer_ref/   # Diffusers Ref2VA Actor weights and config
```

Set the corresponding paths before launching:

```bash
export MODEL_PATH="$MODEL_ROOT/FL2VA"
export ACTOR_CONFIG_PATH="$MODEL_ROOT/transformer"
```

The scripts derive `ACTOR_CONFIG_PATH` as `$(dirname "$MODEL_PATH")/transformer`
when it is not set explicitly. The Ref2VA launcher takes the repository root
as `MODEL_PATH` and resolves `Ref2VA/` and `transformer_ref/` separately. Do not
replace either official rollout transformer with a symlink to a Diffusers
transformer; rollout and Actor loading use different checkpoint layouts.

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
consumed by the T2VA launchers. Use `--train_size` or `--val_size` to create a
smaller debugging dataset.

FL2VA conditions each clip on a first frame. Reuse the DiffusionNFT FL2VA
converter (symlinked here as `prepare_fl2va_data.py`), which emits one
`<image>` token and `frame_indices=[0]`:

```bash
export RAW_FL2VA_DIR=/path/to/raw_fl2va
export FL2VA_DATA_DIR="$HOME/data/fl2va/verl_omni"

python3 examples/flowgrpo_trainer/minimax_h3/prepare_fl2va_data.py \
  --input_dir "$RAW_FL2VA_DIR" \
  --output_dir "$FL2VA_DATA_DIR" \
  --frame_mode first
```

Each `train.jsonl` / `test.jsonl` row carries a prompt and a first-frame image
path relative to the input directory; see the DiffusionNFT FL2VA recipe for the
exact schema.

For Ref2VA, create `train.jsonl` and `test.jsonl` under one input directory.
Each row accepts `images`, `videos`, and `audios`; paths may be absolute or
relative to the input directory. At least one image or video is required.
Video entries may be paths or objects with `path` and `start_time_seconds`:

```json
{"prompt":"The subject waves while the scene remains unchanged.","images":["refs/person.png"],"videos":[{"path":"refs/motion.mp4","start_time_seconds":0.0}],"audios":["refs/music.wav"]}
```

Convert both splits with:

```bash
python3 examples/flowgrpo_trainer/minimax_h3/prepare_ref2va_data.py \
  --input_dir /path/to/ref2va_jsonl_and_media \
  --output_dir "$HOME/data/minimax_h3_ref2va"
```

The official limits are at most 9 images, 3 videos, 3 standalone audios, and
12 files total. Each video/audio clip must be 2–15 seconds and total reference
media duration must not exceed 15 seconds. A standalone audio reference needs
at least one visual reference. Video soundtracks are detected and conditioned
automatically; do not list them again under `audios`.

Reference-condition rows are padded to `MAX_PROMPT_EMBEDS`, with explicit row
counts retained for Actor replay. The value must cover the largest per-sample
condition layout; multiple references at the default 2048-pixel short edge may
require a larger cap or a smaller `REF_IMAGE_SHORT_EDGE`. Generated target
trajectories remain fixed-size and never repeat reference rows across SDE steps.

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
These rewards validate generated audio/video alignment but do not directly
measure fidelity to the supplied references.


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

### NVIDIA GPU (FL2VA)

```bash
MODEL_PATH="$MODEL_ROOT/FL2VA" \
DATA_DIR="$HOME/data/fl2va/verl_omni" \
IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

FL2VA reuses the same CPS FlowGRPO configuration as T2VA. The first-frame
condition rows are held fixed across the reverse-SDE window and re-injected
after every transition, so only the target video/audio rows are scored.

### NVIDIA GPU (Ref2VA)

```bash
MODEL_PATH="$MODEL_ROOT" \
DATA_DIR="$HOME/data/minimax_h3_ref2va" \
IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth \
REF_IMAGE_SHORT_EDGE=512 \
VAL_REF_IMAGE_SHORT_EDGE=1024 \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_ref2va_lora.sh
```

Ref2VA preserves the original Agent Loop token IDs and adds the official
reference presentation in rollout. Reference image/video rows use timestep
`0.999`, reference-audio rows use `1.0`, and all reference rows remain fixed
through every stochastic reverse-SDE transition. They are transported once and
reinserted by the Actor; only generated video and audio rows are stored in the
trajectory and contribute to the FlowGRPO log probability. The launcher keeps
the official reference-image short edge of 2048 by default. Set
`REF_IMAGE_SHORT_EDGE` for training and `VAL_REF_IMAGE_SHORT_EDGE` for
validation to multiples of 32 from 256 through 2048. The validation setting
defaults to the training value.

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

The Ref2VA launcher defaults to rollout/text-encoder TP 4, `448x288`, 96
requested frames, 10 inference steps, `MAX_PROMPT_EMBEDS=12288`, rollout n=8,
and Actor micro-batch 1. It enables layerwise rollout offload and FSDP2 Actor
parameter/optimizer offload because reference presentations can be much longer
than T2VA prompts.

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
| `MAX_PROMPT_EMBEDS` | Prompt/reference-row padding cap; defaults to 12288 |
| `REF_IMAGE_SHORT_EDGE` | Ref2VA training image short edge; defaults to 2048 |
| `VAL_REF_IMAGE_SHORT_EDGE` | Ref2VA validation image short edge; defaults to the training value |
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

- The TransferQueue-specific Agent Loop path does not yet support variable
  reference-row padding; the default diffusion Agent Loop manager does.
- Distilled checkpoint-specific sigma schedules are rejected because Actor and
  rollout replay currently use the standard H3 video/audio schedules.
- CLAP and ImageBind are required by the provided launchers; change the reward
  configuration explicitly if either reward is unavailable.
