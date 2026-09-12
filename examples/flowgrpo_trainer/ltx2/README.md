# LTX-2.3 audio-video FlowGRPO

Last updated: 09/06/2026

These recipes train `dg845/LTX-2.3-Diffusers` LoRA adapters for text-to-audio-video
(T2AV) or single-first-frame text-and-image-to-audio-video (TI2VA) generation with
a diffusers + FSDP actor, vLLM-Omni rollout, joint audio-video CPS transitions,
and the CLAP plus ImageBind rewards.
The checkpoint advertises `_class_name: LTX2Pipeline`; the registered rollout
adapter uses vLLM-Omni's LTX-2.3-specific `LTX23Pipeline` implementation behind
that checkpoint architecture key.

## Prepare data

The prompt corpus is derived from
[VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM), a CC-BY-NC 4.0
dataset containing 1.67 million unique text-to-video prompts from real users.
This recipe uses the 50,000-prompt subset from
[`video_prompts.txt`](https://github.com/XueZeyue/DanceGRPO/blob/main/assets/video_prompts.txt)
with the following split:

- `train.txt`: 48,976 prompts
- `test.txt`: 1,024 prompts

Reproduce the data selection and split from the downloaded
`video_prompts.txt` file with:

```python
import random

with open("video_prompts.txt") as f:
    prompts = [line.strip() for line in f]

random.seed(42)
random.shuffle(prompts)

test = prompts[:1024]
train = prompts[1024:]
```

Use `dataset/vid_prompt/train.txt` and `test.txt`:

```bash
python3 examples/flowgrpo_trainer/ltx2/prepare_data.py \
  --input_dir ./dataset/vid_prompt \
  --output_dir "$WORKSPACE/data/vid_prompt/verl_omni" \
  --val_size 128
```

The documented recipe uses all training prompts and 128 validation prompts.
The script defaults both `--train_size` and `--val_size` to `-1`, which converts
all prompts when a limit is not provided.

For TI2VA, provide `train.jsonl` and `test.jsonl` with one condition image per row:

```json
{"prompt": "A fox turns toward the camera while snow falls.", "image": "images/fox.png"}
```

Paths are relative to the input directory. Convert the splits with:

```bash
python3 examples/flowgrpo_trainer/ltx2/prepare_ti2va_data.py \
  --input_dir ./dataset/ltx2_ti2va \
  --output_dir "$WORKSPACE/data/ltx2_ti2va/verl_omni"
```

The converter embeds exactly one image in each parquet row. The shared
`RLHFDataset` and LTX agent loop transport the image separately without a HF
processor, keeping it out of the Gemma-3 text-token stream. The rollout keeps its
first latent frame fixed, stores only generated video rows plus audio in each
trajectory state, and transports the fixed frame once for Actor replay.

## Install reward dependencies

CLAP uses the existing `transformers` and `torchaudio` dependencies. ImageBind
is optional software under the CC-BY-NC-SA 4.0 non-commercial license:

```bash
pip install 'git+https://github.com/facebookresearch/ImageBind.git'
pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'
```

Review the ImageBind license before enabling this reward in your environment.

## Launch

### T2AV GPU

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora.sh
```

### TI2VA GPU

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_ti2va_lora.sh
```

The GPU recipes default to 8 GPUs, vLLM-Omni tensor parallel size 2, and one
reward worker. CLAP and ImageBind run on `cuda:0` and `cuda:1`, respectively.
The TI2VA recipe supports exactly one first-frame image and does not train the
fixed condition frame as part of the stochastic policy transition.

The TI2VA launcher ships the validated long-run configuration: rollout tensor
parallel size 1, 512 samples per step (batch 32 x n=16), 121-frame 10-step
256x384 training rollouts with FSDP2 and micro batch 16, SDE window 3 sampled
non-contiguously from `[0, 10)`, 121-frame 50-step validation, FA3 attention
on both the actor (`_flash_3_varlen_hub`) and rollout (`FLASH_ATTN_3_HUB`)
sides, and checkpoints every 20 steps. On 8x H800 with a 27,687-sample FLUX
first-frame dataset this configuration trained stably for 140 steps with the
CLAP+ImageBind training reward climbing from 0.16 to 0.63; FA3 was verified
error-free on Hopper (compute capability 9.0) for the whole run.

### Ascend NPU

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora_npu.sh
```

The NPU recipe defaults to 16 NPUs, vLLM-Omni tensor parallel size 4, one reward
worker, and reward devices `npu:0` and `npu:1`. It sources the Ascend toolkit
and ATB environment from `ASCEND_HOME_PATH`, which defaults to
`/usr/local/Ascend/ascend-toolkit`.

All launch scripts accept `WORKSPACE`, `MODEL_PATH`, `DATA_DIR`, `OUTPUT_DIR`,
`NUM_GPUS`, `ROLLOUT_TP`, `TOTAL_TRAINING_STEPS`, and `WANDB_MODE` through
environment variables. The NPU script additionally accepts `ASCEND_HOME_PATH`,
`CLAP_MODEL_PATH`, `IMAGEBIND_MODEL_PATH`, `REWARD_DEVICE`, and
`REWARD_NUM_WORKERS`. Extra Hydra overrides can be appended to either command.

The current scripts use a training batch size of 32, eight rollouts per prompt,
a PPO mini-batch size of 16, and 100 total training steps by default. Outputs
are written below `OUTPUT_DIR`, including checkpoints and timestamped logs.

For each rollout, the recipe samples three non-contiguous SDE transitions from
the step range `[0, 10)`. This is configured with `sde_window_size=3`,
`sde_window_range=[0,10]`, and `sde_contiguous=False`. The default
`sde_contiguous=True` retains the consecutive-window behavior used by existing
diffusion recipes.

The reference training recipe maintains a separate EMA evaluation copy.
The current verl-omni FlowGRPO trainer evaluates and checkpoints the live LoRA
policy, so the EMA-only evaluation behavior is the one reference option not
mapped by this launch script.
