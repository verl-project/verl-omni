# MiniMax-H3 text-to-audio-video DiffusionNFT training

Last updated: 08/22/2026

This recipe trains a rank-64 MiniMax H3 LoRA with online DiffusionNFT for
text-to-audio-video (T2VA). A Diffusers transformer is trained with FSDP2 while
vLLM-Omni generates joint video and audio rollouts. CLAP and ImageBind provide
the default multi-reward (audio-video alignment).

A ready-made FL2VA first-frame dataset built with the data pipeline below is
published at https://huggingface.co/datasets/zyfenghit/dancegrpo-t2av

## Video Examples



<table>
  <tr>
    <th align="center">ID</th>
    <th>Prompt</th>
    <th align="center">MiniMax H3 base model</th>
    <th align="center">MiniMax H3 + DiffusionNFT</th>
  </tr>
  <tr>
    <td align="center">1</td>
    <td>stickman monigote shooting a energy sphere from his hands</td>
    <td align="center"><video src="https://github.com/user-attachments/assets/fc460b8f-3e49-4a23-9a60-a04f45df2190" width="280" controls></video></td>
 <td align="center"><video src="https://github.com/user-attachments/assets/5fb70550-6c1e-428d-bd0c-dad21154745e" width="280" controls></video></td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td>a husky dog with sunglasses riding on santas sled</td>
    <td align="center"><video src="https://github.com/user-attachments/assets/e424acf9-6d40-4485-bb8b-c0daed244fc6" width="280" controls></video></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/d311b43c-0b49-4679-8e30-26711e91fd86" width="280" controls></video></td>
  </tr>
  <tr>
    <td align="center">3</td>
    <td>minimalist polygonal human skull in green flames with strong movement, uhd</td>
    <td align="center"><video src="https://github.com/user-attachments/assets/95b260fe-cda3-4f4b-8e9a-5a670a5cf5e2" width="280" controls></video></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/15c823c7-4a8c-460e-8af9-9e9f720d07e8" width="280" controls></video></td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td>17th century sailing ship making a path through the waves during a storm</td>
    <td align="center"><video src="https://github.com/user-attachments/assets/7730a885-e6a6-4753-9722-ebaf13bb0c45" width="280" controls></video></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/a2d40943-71d1-4cef-abd7-ae7a31c456ad" width="280" controls></video></td>
  </tr>
  <tr>
    <td align="center">5</td>
    <td>close up of a skin texture, two hands with black gloves tatoo a red butterfly over it</td>
    <td align="center"><video src="https://github.com/user-attachments/assets/b31a480b-5677-4d60-a65d-af2175f7a7cd" width="280" controls></video></td>
    <td align="center"><video src="https://github.com/user-attachments/assets/b7e9fff2-e711-42ec-a21e-7e9f53113ac2" width="280" controls></video></td>
  </tr>
</table>

## Install

Follow the project [installation guide](../../../docs/start/install.md),
then install the repository-pinned vLLM-Omni revision:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@245d78fb48f1c87dfb560a94be6e191c9f9f1c0"
```

The explicit Diffusers revision is the tested API target that provides
`MiniMaxH3Transformer3DModel`.

## Checkpoint

`MODEL_PATH` must be a local MiniMax-H3 repo root containing `FL2VA/`
(vLLM-Omni rollout checkpoint) and `transformer/` (converted Diffusers
`MiniMaxH3Transformer3DModel` for FSDP training). Do not replace the official
rollout transformer with a symlink to the Diffusers conversion.

## Data Preparation

### T2VA (prompt-only)

Convert prompt splits to prompt-only parquet (no condition images, and no
negative prompts, since H3 is CFG-distilled):

```bash
python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2av_data.py \
    --input_dir /path/to/raw_prompts \
    --output_dir /path/to/h3_t2va_data
```

Input is `train.txt`/`test.txt` (one prompt per line) or
`train.jsonl`/`test.jsonl` (`prompt`/`text`/`caption` fields).

### FL2VA first-frame data

Offline pipeline for MiniMax H3 FL2VA (text+image to audio-video) RL training:
turn a prompt list into FLUX reference images, pair them into train/test JSONL,
and feed the FL2VA `prepare_data.py` converter.

#### Pipeline

```
prompts.txt ──► gen_flux_images.py ──► images/{index:06d}.jpg
                     │                        │
                     │                        ▼
                     │              build_fl2va_jsonl.py
                     │                        │
                     ▼                        ▼
              (same index)          train.jsonl / test.jsonl
                                             │
                                             ▼
                              prepare_data.py --frame_mode first
                                             │
                                             ▼
                                train.parquet / test.parquet
```

#### Getting the prompt file

`dancegrpo_consist-id.txt` is the filtered ConsisID prompt list released by
DanceGRPO (27,815 prompts, one per line). Download it directly from the
DanceGRPO repository:

```bash
curl -L -o dancegrpo_consist-id.txt \
  https://raw.githubusercontent.com/XueZeyue/DanceGRPO/main/assets/consist-id.txt
```

The prompts originate from
[ConsisID-preview-Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data)
captions; DanceGRPO ships the filtered result as-is and does not document the
exact filtering criteria, so downloading the released file is the reproducible
way to get the identical prompt set (verified line-for-line against the copy
used for the published dataset).

#### Scripts

##### `gen_flux_images.py`

Multi-GPU FLUX batch image generator. Reads one prompt per line, shards the
prompts across ranks (`torchrun`), and writes one JPEG per prompt plus a
per-rank `metadata_rankN.jsonl` (image <-> prompt <-> index mapping).
Deterministic per-prompt seed (`seed + index`), and re-running skips images
that already exist, so interrupted jobs resume safely. Defaults mirror
DanceGRPO's online reference pipeline (400x640, 30 steps, guidance 3.5,
max_sequence_length 512), so prompt and condition image are semantically
aligned by construction.

```bash
torchrun --nproc_per_node=8 examples/diffusionnft_trainer/minimax_h3/gen_flux_images.py \
    --prompt_file dancegrpo_consist-id.txt  # see "Getting the prompt file" \
    --model_path /path/to/FLUX.1-dev \
    --output_dir data/flux_images \
    --height 400 --width 640
```

##### `build_fl2va_jsonl.py`

Pairs each prompt with its same-index image, verifies all images exist,
shuffles with a fixed seed, and writes `train.jsonl` / `test.jsonl` with
relative image paths — the input format of `prepare_data.py`:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/build_fl2va_jsonl.py \
    --prompt_file dancegrpo_consist-id.txt  # see "Getting the prompt file" \
    --image_dir data/flux_images/images \
    --output_dir data/flux_images \
    --test_size 128 --seed 42
```

#### Reference dataset recipe

- Prompts: 27,815 English video captions from
  [ConsisID-preview-Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data),
  as filtered by [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
  (`assets/consist-id.txt`)
- Images: FLUX.1-dev, 400x640, 30 steps, guidance 3.5, per-index seeds
- Split: seed-42 shuffle -> 27,687 train / 128 test
- Convert: `prepare_data.py --frame_mode first`, then train with
  `rollout.pipeline.task=fl2va` and `frame_indices='[0]'`. The rollout
  pipeline LANCZOS-resizes condition images to the sampling resolution, so
  training at e.g. 288x464 (same ~1:1.61 aspect) works directly.

## Launch

```bash
export MODEL_PATH=/path/to/MiniMax-H3
export DATA_DIR=/path/to/h3_t2va_data

bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

MiniMax H3 t2va requires an explicit named `aspect_ratio` (one of
`21:9/16:9/4:3/1:1/3:4/9:16`); the launch script sets `16:9` and explicit
`height`/`width` control the actual canvas (must be multiples of 32).

The H3-specific agent loop (`minimax_h3_diffusion_single_turn_agent`) is
required: it tokenizes raw text once and sends those token IDs directly to
the H3 text encoder.

Training rollouts sample with `INFER_STEPS=10` diffusion steps for
throughput; validation always uses 40. Raise `INFER_STEPS` (e.g. 50) for
higher-quality rollouts. Common overrides:

```bash
NUM_GPUS=8 ROLLOUT_TP=4 ROLLOUT_N=4 INFER_STEPS=50 \
TOTAL_TRAINING_STEPS=100 OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

## FL2VA (image-conditioned) training

This recipe trains the FL2VA checkpoint with online DiffusionNFT. The rollout
uses vLLM-Omni's official first/last-frame contract and the Actor applies the
NFT forward-process objective only to generated video/audio rows.

### Data

Prepare `train.jsonl` and `test.jsonl`. Each row has a prompt and either an
`images` list or explicit first/last names:

```json
{"prompt":"A sunrise becomes a starry night.","images":["images/first.png","images/last.png"]}
```

Convert it with:

```bash
python examples/diffusionnft_trainer/minimax_h3/prepare_data.py \
  --input_dir /path/to/raw \
  --output_dir /path/to/parquet \
  --frame_mode first_last
```

`frame_mode` can be `first`, `last`, or `first_last`. Set the matching launcher
value:

```bash
export FRAME_INDICES='[0,-1]'  # '[0]' or '[-1]' for one-image datasets
```

### Checkpoint

`MODEL_PATH` is a single MiniMax-H3 repo root that already ships both transformer
layouts as siblings: `FL2VA/` (the fused QKV+GEGLU rollout checkpoint used by
vLLM-Omni, with rollout weights under `FL2VA/transformer/`) and `transformer/`
(the diffusers `MiniMaxH3Transformer3DModel` used for FSDP actor training). The
official `MiniMaxAI/MiniMax-H3` repo provides both, so no manual conversion is
needed. Do not overwrite `FL2VA/transformer/` with the diffusers `transformer/`:
that silently breaks the rollout weight loader.

### Run

```bash
MODEL_PATH=/path/to/MiniMax-H3 \
DATA_DIR=/path/to/parquet \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

The latest vLLM-Omni contract requires 4–15 seconds at 24 FPS. The launcher's
`NUM_FRAMES=96` is aligned by vLLM-Omni to the next valid `17n+5` boundary.
Sampling edges must be multiples of 32 (the H3 pipeline silently floors
anything else); the recipe uses 288x448 for training and 576x928 for
validation. Training rollouts sample with `INFER_STEPS=10` diffusion steps for
throughput and validation uses 40; 2–4 steps are suitable only for contract
smoke tests. Do not use the old short 22/29-frame settings.

To verify that the two checkpoint directories represent exactly the same base
policy after fused-QKV and GEGLU conversion, run:

```bash
python tests/special_e2e/minimax_h3_checkpoint_parity.py \
  --vllm-transformer "$MODEL_PATH/FL2VA/transformer" \
  --diffusers-transformer "$MODEL_PATH/transformer"
```

The H3-specific agent loop is required: it tokenizes raw text once and lets
vLLM-Omni prepend the `<Picture N>` vision presentation. Replacing it with the
generic diffusion agent loop changes the prompt contract.

CPU offload is enabled and the default rollout TP is 4: with TP=2, the
colocated dummy-load phase places a full text encoder beside one DiT shard
before offload activates and leaves no room for Actor-to-rollout weight
synchronization on a 96 GB GPU. With the 8-GPU recipe, `ROLLOUT_N=4` also
makes the two-prompt Actor batch divisible by the FSDP data-parallel world
size.

The launcher defaults to FlashAttention 3 for both the Actor
(`_flash_3_varlen_hub`) and the rollout (`FLASH_ATTN_3_HUB`), which requires
the `kernels` package. Override `ACTOR_ATTN_BACKEND` and
`ROLLOUT_ATTN_BACKEND` together (e.g. `native` / `TORCH_SDPA`) only after
validating the replacement backends.

Rollout quantization is intentionally not enabled. On the pinned vLLM-Omni
commit, a native custom pipeline combined with online FP8 can hit a meta-tensor
placement failure during custom-pipeline initialization; BF16 TP=4 is the
validated path.

## T2VA performance reference

The run below trains the rank-64 LoRA with the default CLAP + ImageBind
multi-reward; the weighted-sum reward rises steadily as audio-video alignment
improves.

### Train Reward

![Train reward](https://github.com/user-attachments/assets/5caf849a-f76c-4edd-a7cb-1820fbae08c1)

![Train reward (detail)](https://github.com/user-attachments/assets/61cef5be-a552-4e97-91bc-b4f1a27ad4e2)

### Eval Reward

![Eval reward](https://github.com/user-attachments/assets/73bd6b28-60af-4fec-a661-e6e9ce9a90d7)

### Time consumption

![Time consumption](https://github.com/user-attachments/assets/68e82be9-175d-49d5-88a0-efe434a92698)

## FL2VA performance reference

The FL2VA (image-conditioned) run uses the same rank-64 LoRA and CLAP +
ImageBind multi-reward as T2VA and follows the same layout as above.

<!-- TODO(fl2va): replace the placeholders below with the FL2VA DiffusionNFT run
     curves. Drag the wandb screenshots into a PR comment and paste the resulting
     GitHub user-attachments CDN URLs here, mirroring the T2VA image blocks. -->

### Train Reward
![Train reward](https://github.com/user-attachments/assets/718f85c4-8cd0-4bb8-987f-c4e3dd597dfb)

![Train reward (detail)](https://github.com/user-attachments/assets/7a355c98-4ce0-4bdc-a3b4-db560a7f67c5)


### Eval Reward
![Eval reward](https://github.com/user-attachments/assets/9508dd62-3def-4bcb-abe4-0fc5394f4b11)

### Time consumption

![Time consumption](https://github.com/user-attachments/assets/02f53c81-b98e-4e73-ab16-5683792ae574)

## License

- Prompts: CC-BY-4.0 (ConsisID-preview-Data)
- Images: generated with FLUX.1-dev (non-commercial license); datasets built
  with this pipeline inherit the non-commercial restriction
