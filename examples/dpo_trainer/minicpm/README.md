# MiniCPM Offline DPO

This example trains MiniCPM multimodal understanding with offline DPO through
`verl_omni.trainer.main_omni`.  It uses turn-based preference rows and keeps the
training path simplex: vision/audio understanding modules and the AR language
model can be trained, while audio generation modules such as talker, codec,
TTS, audio decoder, and code2wav are excluded by default.

## Supported batch kinds

MiniCPM's remote-code processor supports two training batch types only:

1. **Image-only** batches from `image/*.parquet` rows.
2. **Audio-only** batches from `audio/*.parquet` rows with standalone `audios`
   paths and `<audio>` prompt placeholders.


## Data

Prepare Omni-Preference parquet files by following
[`omni_preference_dpo_dataset.md`](../data_process/omni_preference_dpo_dataset.md).

Convert Omni-Preference into the offline MLLM DPO parquet schema:

```bash
python examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py \
  --dataset_root "$HOME/Omni-Preference" \
  --output_dir "$HOME/Omni-Preference/parquet_dpo" \
  --modalities image audio
```

The generated parquet schema is model-agnostic. MiniCPM-specific behavior is
handled later by `data.base_transform=minicpm` in the dataset transform, not by a
separate Omni-Preference converter.

Parquet prompts should keep compact semantic markers (`<image>`, `<audio>`).
The MiniCPM transform rewrites them to processor slots (`<image>./</image>`,
`<audio>./</audio>`) before calling `MiniCPMOProcessor`.

## Training

```bash
DATASET_ROOT="$HOME/Omni-Preference" \
DATA_DIR="$DATASET_ROOT/parquet_dpo" \
MODEL_PATH=openbmb/MiniCPM-o-4_5 \
bash examples/dpo_trainer/minicpm/run_minicpm_omni_preference_lora.sh
```

`OmniFSDPEngine._build_module` loads MiniCPM-o through
`MiniCPMThinkerAdapter.auto_model_class` (`architectures[0]` is `MiniCPMO`). The launch script sets `init_tts=false` through the
Hugging Face config override so the inference-only TTS module is not initialized
for training.

Key defaults in the launch script:

- `data.train_files`: image + audio parquet only
- `ModalityGroupedBatchSampler` weights: `{image, audio}` only
- `actor_rollout_ref.model.exclude_modules`: skip LoRA on `vpm` / `apm` and
  generation-only modules.
- `actor_rollout_ref.actor.strategy=fsdp2`: required. `get_fsdp_ignored_module_names`
  returns `["apm", "vpm", "resampler"]`, and the engine rejects a non-empty ignore
  list under `strategy=fsdp`. The Whisper encoder adds its `embed_positions` to
  `inputs_embeds`, which errors once that tensor is a DTensor, so `apm` is passed
  as root `ignored_params` and stays unsharded; the frozen vision towers ride
  along so an unsharded forward cannot desync the ranks. Ignored parameters must
  stay frozen, which is why `apm` and `vpm` are also LoRA-excluded.

## Validation accounting

Two defaults decide how many pairs `val/reward_accuracy` is actually averaged over,
and both must hold for metrics to be comparable between runs:

- `VAL_MAX_SAMPLES` defaults to `VAL_BATCH_SIZE * 4`, i.e. 2 modalities x 2 full
  batches. `ModalityGroupedBatchSampler` runs with `drop_last=true` so that no
  DataLoader batch mixes image and audio rows, which discards the trailing partial
  chunk of each modality. Requesting a count that is not a multiple of
  `2 * VAL_BATCH_SIZE`, or whose per-modality share exceeds that modality's test
  rows, therefore scores fewer pairs than `val_max_samples` implies.
- `DATA_SEED` pins `data.seed`. `data.shuffle` defaults to `true`, so
  `balance_max_samples_by_modality` draws the `val_max_samples` subset with
  `np.random.default_rng(seed)`; with the default `data.seed=null` every run
  evaluates a different subset. Pin it before comparing two runs' metrics.
