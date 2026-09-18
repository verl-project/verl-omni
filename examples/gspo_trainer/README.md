# Qwen3-Omni Thinker GSPO Trainer

Last updated: 09/14/2026

This example shows how to post-train the **Qwen3-Omni-30B-A3B Thinker** with
**GSPO** on multimodal reasoning tasks, using FSDP for the actor and `vllm-omni` as
the async rollout backend. Four input recipes are supported: **text → text**
(`gsm8k`), **image → text** (`MMK12`),
**text + image + audio → text** (`AVQA-R1-6K`), and
**video frames + video audio + text → text** (`NExT-QA`).

Both **GPU** and **NPU** training platforms are supported:

- `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh`
  — **GPU**, **LoRA (r=32)** on a single node with **4 × H800 80GB**.
- [`run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh`](qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh)
  — **GPU**, **LoRA (r=32) V1** for text + image + audio AVQA training.
- [`run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh`](qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh)
  — **NPU**, **full-parameter V1** for text + image + audio AVQA training.
- [`run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh`](qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh)
  — **NPU**, **full-parameter V1** for video and soundtrack NExT-QA training.

For the base environment setup, see the [installation guide](../../docs/start/install.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base
environment. In short:

```bash
git clone https://github.com/verl-project/verl-omni.git && cd verl-omni
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
# flash-attn is required for GPU training
uv pip install flash-attn>=2.8.3
```

> **Tested with** `transformers==5.13.1`, `accelerate==1.14.0`, `peft==0.19.1`.

Verify:

```bash
python -c "import verl, verl_omni, vllm, vllm_omni; print('OK')"
```

The GPU V1 and AVQA NPU launchers use `verl_omni.trainer.main_omni` and set
`VERL_USE_EXTERNAL_MODULES=verl_omni`. Processor/model setup is handled by the
registered Qwen3-Omni V1 adapter, so these launchers do not load model
monkey-patches through `external_lib`.

The launchers colocate the FSDP actor and the `vllm-omni` rollout on the same
devices. `run_qwen3_omni_thinker_gspo_lora_v1.sh` targets a single node with
**4 × H800 80GB**; `run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh` targets a single
**Atlas 800T A3** node with **16 × Ascend 910C 64GB**. It dynamically generates
a thinker-only deploy config for each rollout replica from that replica's
visible devices, avoiding cross-replica device-rank collisions.

## Prepare the model

The GPU V1 scripts default `MODEL_PATH` to `$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct`
(~60 GB). The AVQA NPU script defaults to the HuggingFace Hub ID `Qwen/Qwen3-Omni-30B-A3B-Instruct`.
The NExT-QA script defaults to `/models/Qwen3-Omni-30B-A3B-Instruct` and requires
the original full checkpoint.
To use a different local copy or Hub ID, set `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/local/Qwen3-Omni-30B-A3B-Instruct
```

## Training with `gsm8k`

### Prepare the dataset

A parquet dataset of GSM8K math problems, defaulting to
`~/data/gsm8k/{train,test}.parquet`. Use verl's
[`gsm8k.py`](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py) converter:

```bash
python gsm8k.py --local_save_dir ~/data/gsm8k
ls ~/data/gsm8k/   # train.parquet  test.parquet
```

### Run training

Launch from the repository root — pick the flavor that matches your hardware:

```bash
# GPU, LoRA (r=32), 4 × H800 — V1 trainer (recommended)
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh

# NPU, AVQA, Atlas 800T A3 (16 × Ascend 910C 64GB) — V1 trainer
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh
```

The V1 launchers use pure CLI overrides on `verl_omni.trainer.main_omni`
(no `--config-path/--config-name`, no recipe YAML). Config precedence,
lowest to highest:

```
verl omni_trainer defaults  →  CLI overrides (run script)  →  "$@" extra args
```

Any field can be overridden from the command line without editing the script:

```bash
MODEL_PATH=/local/Qwen3-Omni-30B-A3B-Instruct \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.lr=2e-6
```

### What is trained

Only the **Thinker** (`Qwen3OmniMoeThinkerForConditionalGeneration`):

- **GPU (LoRA)** — rank 32, alpha 64, on
  `target_modules="['q_proj','k_proj','v_proj','o_proj']"`
  (the V1 `Qwen3OmniThinkerAdapter.configure_model` handles Thinker-forward
  redirection and `_verl_strip_modules` via `get_strip_modules`,
  so `exclude_modules` only needs to cover the heads/encoders).
- **NPU (full-parameter)** — LoRA is disabled (`lora_rank=0`); all Thinker
  parameters are updated under FSDP.
- `exclude_modules` strips talker / code2wav / code_predictor / visual /
  audio_tower; `freeze_vision_tower=True` keeps the vision encoder cold.
- `configure_model` in the registered adapter
  (`verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py`) redirects
  `module.forward` → `module.thinker.forward` and sets
  `_no_split_modules` after the default base-class stripping.

Reward comes from the `naive` reward manager (math accuracy on parsed answers).

Healthy signals (gsm8k, 4×H800, LoRA r=32):

- `training/rollout_actor_probs_pearson_corr` > 0.995 (actor ↔ rollout agree
  after weight sync) — the primary correctness signal.
- `rollout_corr/log_ppl_diff` ≈ 0.001 (near zero, confirms rollout↔actor
  log-prob consistency).
- `actor/loss` ≈ 1e-5, `actor/grad_norm` ∈ [1e-3, 1e-2], no OOM
  (`actor/perf/max_memory_allocated_gb` < 45).
- `val-core/openai/gsm8k/acc/mean@1` rising with steps.

## Training with `MMK12`

For visual math reasoning we ship an end-to-end pipeline on top of the
[MMK12](https://huggingface.co/datasets/FanqingM/MMK12) dataset (image
input + text output, K12 math). It reuses the same GSPO recipe as the
text-only path — only the data preprocessing and the reward scorer differ.
Use the dedicated V1 GPU/LoRA script:

### Prepare the dataset

Download the raw MMK12 parquet shards (from ModelScope or HuggingFace) into a
local directory — the loader expects filenames like `train-*.parquet` and
`test-*.parquet` — and convert them into the verl RL parquet layout with:

```bash
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path /path/to/mmk12/ \
    --local_save_dir ~/data/mmk12
```

The converter emits one verl RL row per problem, with
`data_source="math_dapo"`, a system prompt that constrains the model to emit
`<answer>…\boxed{…}…</answer>`, and the image bytes carried inline in the
`images` column so the parquet stays self-contained. Input / kept / dropped
counts and answer-type tallies are printed at the end. See the module docstring
in [`examples/gspo_trainer/data_process/mmk12.py`](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/data_process/mmk12.py) for the exact output schema.

### Run training

The MMK12 reward scorer grades responses with
[`math_verify`](https://github.com/huggingface/math-verify). Multimodal data
processing also requires [`qwen-vl-utils`](https://github.com/QwenLM/Qwen2.5-VL)
for vision info extraction. Install both explicitly:

```bash
pip install math-verify qwen-vl-utils
```

Then launch the MMK12 V1 training script:

```bash
TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh
```

For Ascend NPU training, use the NPU variant:

~~~bash
TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1_npu.sh
~~~

Override the model, dataset, or MMK12 reward scorer path without editing the script:

~~~bash
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/test.parquet \
REWARD_FUNCTION_PATH=/path/to/custom_reward.py \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1_npu.sh
~~~

Compared with the GPU script, the NPU variant includes two important Ascend
settings:

- `export VLLM_ASCEND_ENABLE_NZ=0` disables the NZ format in vLLM Ascend.
- `actor_rollout_ref.rollout.cudagraph_capture_sizes` limits the graph shapes
  captured by the rollout engine. Capturing too many shapes can cause runtime
  errors, so keep this list sparse. The current script uses capture sizes
  `[1,2,4,16,64,128,512,1024,2048,3072,4096]`.
The script registers the custom reward scorer internally (no yaml edits
required). Override LR or other fields via "$@" extras:

```bash
TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh \
    actor_rollout_ref.actor.optim.lr=3e-6
```

The scorer combines `math_verify` accuracy with a progressive format reward on
the `<answer>…\boxed{}…</answer>` template; see
[`verl_omni/utils/reward_score/mmk12_reward.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/utils/reward_score/mmk12_reward.py)
for the full formula.

### MMK12 On-Policy Distillation (OPD)

OPD distills a teacher's distribution into the student during GSPO training.
The student is the noised Qwen3-Omni-30B-A3B-Instruct (25% weight noise) and
the teacher is the original (un-noised) model, served by `vllm_omni` in AR mode.

Validated on 2 × Ascend 910C machines — student rollout/actor on 16 GPUs of
node 1, teacher model on 16 GPUs of node 2:

```bash
# 1. On the master node (node 1): ray start --head
# 2. On the slave node (node 2): ray start --address='<head_ip>:<port>'
# 3. Run on the master node:
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1_opd_npu.sh
```

The script sets `distillation.enabled=true` with a `vllm_omni` teacher
(`loss_mode=kl`, `use_policy_gradient=true`). Teacher and student must share
the same tokenizer (same model family).

## Training with `AVQA-R1-6K`

The AVQA recipe trains the Qwen3-Omni Thinker to answer a four-way question
from question text, one image, and one WAV clip. The output is text ending in a
single option tag such as `<answer>B</answer>`.

### Prepare the dataset

```bash
python examples/gspo_trainer/data_process/avqa.py \
    --input_dir /path/to/AVQA_R1 \
    --output_dir ~/data/avqa_r1_6k
```

This writes `train.parquet` and `validation.parquet`. The parquet stores
absolute image/audio paths, so the AVQA media directory must be mounted at the
same path on every Ray worker. The converter validates modalities, options,
labels, and media existence and prints kept/dropped counts for each split.

Image and audio paths are decoded by Qwen's `qwen_omni_utils.process_mm_info`
through
[`QwenOmniRLHFDataset`](../../verl_omni/utils/dataset/omni_rl_datasets.py). Install
the official media loader without changing the NPU engine stack with
`pip install -e ".[audio]"`. `ffmpeg` is only required when the dataset carries
compressed audio (mp3/m4a/aac/ogg) or http(s) audio URLs — those go through
`audioread`/ffmpeg. Plain local WAV files decode via `librosa`/`soundfile`
(libsndfile) and need no ffmpeg.

### Run GPU training

Launch the GPU LoRA script (4 × H800 80GB, LoRA r=32, same GSPO recipe as the
other GPU recipes). The audio-specific settings it adds on top of the base
recipe are:

1. `data.custom_cls` = `QwenOmniRLHFDataset` — the audio-aware dataset class
   that parses `<audio>` placeholders and loads the WAV files (the default
   dataset handles images only).
2. `+data.mm_processor_kwargs.sampling_rate=16000` — the Qwen3-Omni feature
   extractor rate, used when filtering overlong multimodal prompts.
3. Rollout memory-margin knobs for colocated audio workloads —
   `gpu_memory_utilization=0.7` (the base GPU recipe uses `0.8`),
   `rollout.prompt_length=4160`, `engine_kwargs.vllm_omni.max_num_seqs=256`,
   and `cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256]` — see *Sizing
   rollout memory in colocated sleep mode* in the [integrating
   guide](../../docs/contributing/integrating_an_omni_model.md).

```bash
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh
```

The reward extracts the first `<answer>...</answer>` payload from the response
and returns a binary exact-match score against the tagged dataset label.

### Run NPU training

Use the dedicated V1 AVQA NPU launcher. It uses FSDP2 with CPU offload, a
16-NPU topology, rollout TP=4, and four rollout workers without changing the
existing generic NPU script.

```bash
TRAIN_FILE=$HOME/data/avqa_r1_6k/train.parquet \
VAL_FILE=$HOME/data/avqa_r1_6k/validation.parquet \
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh
```

The launcher uses a 4096-token multimodal prompt budget, a 12288-token response
budget, and 128 prompts with 16 responses each per rollout. It trains for 10
epochs, caps dynamic actor and log-prob batches at 20480 tokens per NPU, and
computes entropy in 2048-token chunks to reduce peak NPU memory. It registers
the audio-aware dataset class by importable package path so multiprocessing
preserves its `RLHFDataset` base class, sets rollout NPU memory utilization to
`0.6`, uses deterministic validation, and wires
[`choice_reward.py`](../../verl_omni/utils/reward_score/choice_reward.py).

## Training with `NExT-QA`

This recipe uses the official NExT-QA train and validation annotations with
verl-omni's GSPO loss, Qwen3-Omni V1 trainer, `vllm-omni` rollout, and Ascend
NPU FSDP2 actor. Each sample contains one video and a five-way question. Both
sampled frames and the video's audio track are passed to Qwen3-Omni; the
expected completion ends in a single tag such as `<answer>C</answer>`.

### Prepare the dataset

Clone the official annotation repository. The upstream files live under
`dataset/nextqa/`, so copy the three files used by this recipe to the documented
`repo/` annotation root:

```bash
mkdir -p /datasets/NextQA
cd /datasets/NextQA
git clone https://github.com/doc-doc/NExT-QA.git repo
cp repo/dataset/nextqa/{train.csv,val.csv,map_vid_vidorID.json} repo/
```

Download [`NExTVideo.zip`](https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view?usp=share_link)
from the official NExT-QA repository link and place it directly in
`/datasets/NextQA`. Extract it from that directory:

```bash
cd /datasets/NextQA
unzip NExTVideo.zip
```

`NExTVideo.zip` already contains a top-level `NExTVideo/` directory. Do not use
`unzip NExTVideo.zip -d NExTVideo`; that produces the invalid directory
`NExTVideo/NExTVideo/`, which the converter deliberately rejects.

The resulting input must have this layout:

```text
NextQA/
├── repo/
│   ├── train.csv
│   ├── val.csv
│   └── map_vid_vidorID.json
├── NExTVideo.zip
└── NExTVideo/
    ├── 0001/
    ├── ...
    └── 0083/
        └── 5572343997.mp4
```

Run the converter from the verl-omni repository root:

```bash
python examples/gspo_trainer/data_process/nextqa.py \
    --input_dir /datasets/NextQA \
    --output_dir /datasets/NextQA
```

This writes `/datasets/NextQA/train.parquet` from official `train.csv`
and `/datasets/NextQA/validation.parquet` from official `val.csv`; it
does not randomly re-split records. The converter uses `map_vid_vidorID.json`
to resolve each CSV video ID, validates fields and media paths, and uses the
real `ffprobe` and `ffmpeg` binaries to retain only videos whose first audio
stream can decode at least one frame. Each probe has a 30-second timeout.
Install FFmpeg and ensure both binaries are available in `PATH` before
conversion. The printed JSON reports
input, kept, dropped-by-reason, answer, output, and unique-video audio statistics
for each split. `dropped` counts QA records; `audio` counts unique videos, so
multiple questions for one rejected video increase the former but not the
latter. Absolute media paths are stored in parquet and must be mounted
identically on every Ray worker.

The output files can also be referenced directly in a trainer data config:

```yaml
data:
  train_files: /datasets/NextQA/train.parquet
  val_files: /datasets/NextQA/validation.parquet
```

Video sampling uses 1 FPS, 32--128 visual tokens per frame (`25088--100352` pixels), and at most 32 frames. These values are embedded in each parquet video item for `qwen_omni_utils.process_mm_info`; override them at conversion time with `--fps`, `--min_pixels`, `--max_pixels`, or `--max_frames`.

Install the Qwen Omni media loader with:

```bash
pip install -e ".[audio]"
```

The `audio` extra already installs `qwen-omni-utils>=0.0.9`.
Install the system FFmpeg package on the conversion host and every Ray worker;
both `ffmpeg` and `ffprobe` must be available in `PATH`. For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
ffprobe -version
```

Installing a Python FFmpeg wrapper alone does not provide these required
system commands. The converter invokes both commands, and training invokes
`ffmpeg` directly to decode each video's soundtrack.

TorchCodec is an optional video-frame decoder, independent of soundtrack
decoding. The launcher does not set `FORCE_QWENVL_VIDEO_READER`; the selected
backend depends on the installed Qwen utilities, available decoder packages,
and any inherited environment setting. Current Qwen video utilities prefer
TorchCodec when installed, then Decord, then torchvision. Check the startup
`qwen-vl-utils using ... to read video` message and any fallback warnings to
confirm the backend used by the training environment.

To use TorchCodec on Ascend, install a CPU build compatible with the existing
PyTorch version and platform, following the
[TorchCodec installation and compatibility guide](https://github.com/meta-pytorch/torchcodec#installing-torchcodec).
Keep the NPU PyTorch stack intact. Verify the decoder imports on every worker,
then select it before launching training:

```bash
python -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec import OK')"
export FORCE_QWENVL_VIDEO_READER=torchcodec
export TORCHCODEC_NUM_THREADS=8
```

An import check does not validate decoding; confirm the selected backend can
read the actual dataset videos. TorchCodec does not replace the `ffmpeg` and
`ffprobe` command-line requirements above.

### Video and audio inputs

The launcher selects
[`NextQARLHFDataset`](../../verl_omni/utils/dataset/nextqa_rl_dataset.py).
For each parquet video item, it inserts a separate audio item and decodes the
soundtrack with FFmpeg to mono audio at 16 kHz. The actor and rollout receive
sampled video frames and the decoded waveform through the shared multimodal
payload. Video metadata preserves the sampled frame timing.

`data.mm_processor_kwargs.use_audio_in_video=false` selects separate video
and audio inputs; it does not disable soundtrack loading. Keep this setting
and `sampling_rate=16000` for this dataset class. Clips must contain a
decodable soundtrack. The launcher sets `OMP_NUM_THREADS=8` by default and
uses 128 prompt-filter workers; it does not select a video reader explicitly.

### Reward

The recipe uses the shared
[`choice_reward.py`](../../verl_omni/utils/reward_score/choice_reward.py)
multiple-choice scorer through the `naive` reward manager. It extracts the
first `<answer>...</answer>` payload, strips surrounding whitespace, and
returns `score` and `accuracy` of `1.0` for an exact match with the tagged
label, otherwise `0.0`. Every converted label is one of A--E. The prompt asks
for reasoning inside `<think>...</think>` and a final answer such as
`<answer>C</answer>`.

### Run NPU training

Follow the [NPU installation guide](../../docs/start/install_npu.md), source
CANN/ATB, and ensure FFmpeg and the media dependencies are available on every
Ray worker.

This recipe requires verl with
[`use_no_sync_for_gradient_accumulation` support (#7458)](https://github.com/verl-project/verl/pull/7458).
Commit `a0feb78fe8229fde644aec3bbec20b5dc4583509` (`0.10.0.dev`) includes
that change. The repository's current verl pin `fefb080` and the v0.9.0
release do not include the option. This recipe therefore requires a separate
verl update after the standard environment installation; the repository-wide
dependency pin is unchanged. The development version string alone is not
sufficient to identify a compatible installation.

To update verl in an existing NPU training environment without replacing its
installed PyTorch or other runtime dependencies, run:

```bash
python -m pip install --no-deps --force-reinstall \
    "verl @ git+https://github.com/verl-project/verl.git@a0feb78fe8229fde644aec3bbec20b5dc4583509"
```

Restart the training processes and Ray workers after updating. Installing
`.[train]` again may restore the repository's older pin; apply the recipe-specific
verl update after that installation.

Launch with the original full model checkpoint:

```bash
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/datasets/NextQA/train.parquet \
VAL_FILE=/datasets/NextQA/validation.parquet \
OUTPUT_DIR=/checkpoints/qwen3_omni_nextqa \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh
```

All Thinker parameters are trainable, including the vision tower; LoRA is
disabled. The actor uses BF16 FSDP2, gradient checkpointing, parameter and
optimizer offload, and a micro-batch size of 1 per NPU. The launcher sets
`actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false`
to reduce and shard gradients after each backward, lowering peak NPU HBM usage
at the cost of additional gradient communication. Gradient accumulation and
the optimizer-step boundary are preserved. This uses the upstream verl option;
no local gradient-synchronization override is needed.

The revision installed above defaults this option to `false` in its FSDP YAML, while its
`FSDPEngineConfig` dataclass defaults to `true`. The launcher sets it explicitly
so this recipe does not depend on how the configuration was constructed.
For a deferred-synchronization comparison, explicitly override it to `true`;
omitting the argument does not necessarily enable deferred synchronization.

| Setting | Default |
| --- | --- |
| Nodes and NPUs per node | 1 × 16 |
| Rollout tensor parallelism | 4 |
| Questions per batch × responses per question | 32 × 8 |
| Prompt / response token limits | 8192 / 1024 |
| Rollout memory fraction / maximum concurrent sequences | 0.65 / 4 |
| Graph capture sizes | `[1,2,4]` |
| Learning rate / warmup | `1e-6` / 5%, then constant |
| Training limit | 500 steps, with 10 epochs configured |
| Validation | Before training and every 10 steps |
| Validation sampling | 1 response, temperature 1.0, top-p 0.7 |
| Checkpoints | Every 20 steps; keep the latest 1; automatic resume |

Training uses the full converted train and validation splits, shuffles the
training data, and uses GSPO loss with GRPO advantages. Actor and reward KL
penalties are disabled. Training samples use temperature 1.0 and top-p 1.0.

Set `N_GPUS_PER_NODE`, `NNODES`, `ROLLOUT_TP`, `TRAIN_BATCH_SIZE`, `ROLLOUT_N`,
`TOTAL_TRAINING_STEPS`, `LEARNING_RATE`, `ROLLOUT_MEMORY_FRACTION`,
`ROLLOUT_MAX_NUM_SEQS`, `EXPERIMENT_NAME`, or `OUTPUT_DIR` to customize the run.
The default batch size is twice the total NPU count. It must be divisible by
the total NPU count, and `ROLLOUT_N` must be at least 2. Per-node NPU count must
be divisible by rollout TP. Graph capture sizes follow rollout concurrency
(1 through 8). Trailing Hydra overrides take precedence:

```bash
TOTAL_TRAINING_STEPS=200 \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh \
    trainer.save_freq=20
```

For two nodes with 8 NPUs each, create the Ray cluster on both machines, then
run on the head node with `N_GPUS_PER_NODE=8 NNODES=2`. Model and media paths
must be accessible at the same locations on every node.

The default experiment name is `qwen3_omni_nextqa_fullparam`, with output under
`outputs/${EXPERIMENT_NAME}` in the repository. Checkpoints and validation
outputs go to `${OUTPUT_DIR}/checkpoints` and `${OUTPUT_DIR}/validation`.
Logging uses console and TensorBoard; the launcher also writes
`run_qwen3omni_npu_nextqa_full_ms_16.log` in the repository root.

## Performance

All GPU results measured on a single node of **4 × H800 80GB**, actor and
rollout colocated, LoRA r=32, GSPO. Curves are hosted in the shared
[`verl-omni/gspo_demo`](https://wandb.ai/verl-omni/gspo_demo) W&B project; the
runs were trained with the
[`release/v0.2.0`](https://github.com/verl-project/verl-omni/tree/release/v0.2.0)
branch.

| Script | Dataset | # Cards | Batch × `rollout.n` | lr | Steps | val acc@1 / reward@1 | rollout↔actor pearson | GPU memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`gsm8k (wandb)`](https://wandb.ai/verl-omni/gspo_demo/runs/0tma6mas) | gsm8k | 4 | 128 × 16 = 2048 | 3e-6 | 578 | acc 0.971 | 0.998 | ~43 GB |
| [`MMK12 (wandb)`](https://wandb.ai/verl-omni/gspo_demo/runs/mls202j1) | MMK12 | 4 | 128 × 16 = 2048 | 3e-6 | 392 | reward 0.811 | 0.998 | ~59 GB |
| [`AVQA-R1-6K (wandb)`](https://wandb.ai/verl-omni/gspo_demo/runs/kzzrq9pr) | AVQA-R1-6K | 4 | 128 × 16 = 2048 | 3e-6 | 348 | reward 0.877 | 0.996 | ~46 GB |

**gsm8k** ([wandb](https://wandb.ai/verl-omni/gspo_demo/runs/0tma6mas), `naive`
reward, math accuracy): `critic/rewards/mean` rose from ~0.88 to ~0.97,
`val-core/openai/gsm8k/acc/mean@1` reached **0.971**.
`rollout_corr/log_ppl_diff` stayed near zero (~0.002).

**MMK12** ([wandb](https://wandb.ai/verl-omni/gspo_demo/runs/mls202j1), composite
reward, `math_verify` + format): `critic/rewards/mean` rose from ~0.71 to ~0.81,
`val-core/mmk12/reward/mean@1` reached **0.811** (last logged at
step 392). `rollout_corr/log_ppl_diff` stayed near zero (~0.002).

**AVQA-R1-6K** ([wandb](https://wandb.ai/verl-omni/gspo_demo/runs/kzzrq9pr),
binary `<answer>` exact-match reward): `critic/rewards/mean` rose from ~0.73 to
~0.94, `val-core/avqa_r1_6k/reward/mean@1` reached **0.877**.
`rollout_corr/log_ppl_diff` stayed near zero (~0.007).

## Logging

The NExT-QA launcher uses console and TensorBoard logging. For launchers
configured to use W&B:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
# trainer.project_name / experiment_name are already set in the script
```

## File map

```
examples/gspo_trainer/
├── qwen3_omni/
│   ├── run_qwen3_omni_thinker_gspo_lora_v1.sh       ← V1 launch script (GPU, LoRA r=32, text)
│   ├── run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh  ← V1 launch script (GPU, LoRA r=32, image)
│   ├── run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh   ← V1 launch script (GPU, LoRA r=32, audio + image)
│   ├── run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh    ← V1 launch script (NPU, AVQA)
│   ├── run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh  ← V1 launch script (NPU, NExT-QA)
├── data_process/
│   ├── mmk12.py                                      ← MMK12 → verl RL parquet converter
│   ├── avqa.py                                       ← AVQA → verl RL parquet converter
│   └── nextqa.py                                     ← NExT-QA → verl RL parquet converter
└── README.md                                         ← (this file)
```
