# Quickstart: SD3.5 FlowGRPO with a latent reward model

Last updated: 07/20/2026

Post-train Stable Diffusion 3.5 Medium with FlowGRPO and a
diffusion-native reward model (DRM).

## Introduction

This example trains `stabilityai/stable-diffusion-3.5-medium` on
PickScore-SFW prompts and uses
[DiNa-LRM](https://github.com/HKUST-C4G/diffusion-rm) as the reward model.
Unlike image reward models, DiNa-LRM scores the clean diffusion latent and the
same prompt embeddings used during generation. Training therefore skips VAE
decoding and sends those tensors to an external scorer through a versioned
safetensors HTTP protocol.

The policy trainer and DRM server run in separate Python environments and use
separate GPUs. This avoids dependency conflicts between their Diffusers, PEFT,
and PyTorch stacks.

## Prerequisite

- Install VeRL-Omni by following the {doc}`installation guide <install>`.
- Prepare a `diffusion-rm` environment with its HTTP server dependencies.
- Use a machine with 8 GPUs for the provided script: 7 GPUs for actor and
  rollout workers, and 1 GPU for the DRM server.
- Authenticate with Hugging Face so the SD3.5 and DiNa-LRM checkpoints can be
  downloaded.
- Run the commands below from the VeRL-Omni repository root unless noted
  otherwise.

The DRM HTTP server has its own package requirements and should not be
installed into the VeRL-Omni training environment.

## Step 1: Prepare the PickScore-SFW dataset

Set `WORKSPACE` to a writable directory. It defaults to `$HOME` when unset:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}
```

Download the prompt split used by the original Flow-GRPO project:

```bash
mkdir -p "$WORKSPACE/data/pickscore_sfw/raw"
wget -O "$WORKSPACE/data/pickscore_sfw/raw/train.txt" \
  https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore_sfw/train.txt
wget -O "$WORKSPACE/data/pickscore_sfw/raw/test.txt" \
  https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore_sfw/test.txt
```

Convert the text files to the parquet schema consumed by the diffusion
trainer:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_pickscore_sfw.py \
  --input-dir "$WORKSPACE/data/pickscore_sfw/raw" \
  --output-dir "$WORKSPACE/data/pickscore_sfw/sd3"
```

The command writes `train.parquet`, `test.parquet`, and `metadata.json` under
`$WORKSPACE/data/pickscore_sfw/sd3`. PickScore-SFW supplies prompts only;
DiNa-LRM remains the reward model.

## Step 2: Start the DRM server

Start the scorer from its separate `diffusion-rm` environment. This example
reserves physical GPU 7 for the server:

```bash
cd ../
git clone https://github.com/HKUST-C4G/diffusion-rm.git
cd diffusion-rm
conda activate diffusion-rm-server

CUDA_VISIBLE_DEVICES=7 python -m diffusion_rm.server \
  --model-path liuhuohuo/DiNa-LRM-SD35M-12layers \
  --base-model-path stabilityai/stable-diffusion-3.5-medium \
  --dtype float32 \
  --max-batch-size 8 \
  --max-wait-ms 10 \
  --port 8000
```

Wait until the server is ready:

```bash
curl --fail http://127.0.0.1:8000/healthz
```

The example server does not provide authentication. Bind it to a trusted
network or place it behind an authenticated proxy when trainer and scorer run
on different hosts.

## Step 3: Perform FlowGRPO training

Return to the VeRL-Omni environment, expose GPUs 0-6 to the trainer, and launch
the provided script:

```bash
cd ../verl-omni
source .venv/bin/activate

export WORKSPACE=${WORKSPACE:-$HOME}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

bash examples/flowgrpo_trainer/sd35/run_sd35_medium_drm_lora.sh
```

Edit the paths and principal training variables at the top of the script when
adapting the example to another environment. The script enables online W&B
logging, so configure your W&B credentials before launching or change
`WANDB_MODE` in the script.

## How latent reward scoring works

The rollout pipeline supports three output modes:

- `image`: return a decoded image.
- `latent`: return the clean latent and skip VAE decoding.
- `both`: return a decoded image and attach the clean latent as rollout
  metadata.

The example uses `latent` for training and `both` for validation. This avoids
the VAE decoding cost during training while preserving decoded validation
images for logging.

`latent_http_scorer_client` serializes the clean latent, prompt embeddings,
pooled prompt embeddings, and DRM noise parameters as safetensors. The scorer
returns one raw score, which the example maps with
`score = raw_score * 0.1 + 1.0`. DRM scoring is marked as required, so missing
tensors, invalid responses, or exhausted HTTP retries stop training instead
of silently contributing a zero reward.

## Further reading

- See {doc}`flowgrpo_quickstart` for the standard SD3.5 OCR FlowGRPO example.
- See {doc}`../algo/flowgrpo` for the algorithm and configuration reference.
