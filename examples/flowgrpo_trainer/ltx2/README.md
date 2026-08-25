# LTX-2.3 text-to-audio-video FlowGRPO

Last updated: 08/21/2026

This recipe trains `dg845/LTX-2.3-Diffusers` LoRA adapters with a diffusers +
FSDP actor, vLLM-Omni rollout, joint audio-video CPS transitions, and the CLAP
plus ImageBind rewards.
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

## Install reward dependencies

CLAP uses the existing `transformers` and `torchaudio` dependencies. ImageBind
is optional software under the CC-BY-NC-SA 4.0 non-commercial license:

```bash
pip install 'git+https://github.com/facebookresearch/ImageBind.git'
pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'
```

Review the ImageBind license before enabling this reward in your environment.

## Launch

### GPU

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora.sh
```

The GPU recipe defaults to 8 GPUs, vLLM-Omni tensor parallel size 2, and one
reward worker. CLAP and ImageBind run on `cuda:0` and `cuda:1`, respectively.

### Ascend NPU

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora_npu.sh
```

The NPU recipe defaults to 16 NPUs, vLLM-Omni tensor parallel size 4, one reward
worker, and reward devices `npu:0` and `npu:1`. It sources the Ascend toolkit
and ATB environment from `ASCEND_HOME_PATH`, which defaults to
`/usr/local/Ascend/ascend-toolkit`.

Both launch scripts accept `WORKSPACE`, `MODEL_PATH`, `DATA_DIR`, `OUTPUT_DIR`,
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
