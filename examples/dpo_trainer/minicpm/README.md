# MiniCPM Offline DPO

This example trains MiniCPM multimodal understanding with offline DPO through
`verl_omni.trainer.main_omni`.  It uses turn-based preference rows and keeps the
training path simplex: vision/audio understanding modules and the AR language
model can be trained, while audio generation modules such as talker, codec,
TTS, audio decoder, and code2wav are excluded by default.

## Supported batch kinds

MiniCPM's remote-code processor is reliable for two training batch types only:

1. **Image-only** batches from `image/*.parquet` rows.
2. **Video+audio** batches from `video/*.parquet` rows with
   `mm_configs.use_audio_in_video=true`, so audio is decoded from the video file
   instead of using standalone `audio/*.parquet` rows.

Do **not** include `audio/train.parquet` in `data.train_files` for MiniCPM DPO.
The dataset loader rejects standalone audio-only rows when
`data.base_transform=minicpm`.

## Data

Convert Omni-Preference into the offline MLLM DPO parquet schema:

```bash
python examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py \
  --dataset_root "$HOME/Omni-Preference" \
  --output_dir "$HOME/Omni-Preference/parquet_dpo" \
  --modalities image video
```

The generated parquet schema is model-agnostic. MiniCPM-specific behavior is
handled later by `data.base_transform=minicpm` in the dataset transform, not by a
separate Omni-Preference converter.

## Training

```bash
DATASET_ROOT="$HOME/Omni-Preference" \
DATA_DIR="$DATASET_ROOT/parquet_dpo" \
MODEL_PATH=openbmb/MiniCPM-o-4_5 \
bash examples/dpo_trainer/minicpm/run_minicpm_omni_preference_lora.sh
```

The script uses `AutoModel.from_pretrained(..., trust_remote_code=True)` through
the MiniCPM omni adapter. It auto-detects the architecture from the checkpoint
config and sets `init_tts=false` through the Hugging Face config override so the
inference-only TTS module is not initialized for training.

Key defaults in the launch script:

- `data.train_files`: image + video parquet only
- `mm_configs.use_audio_in_video=true`
- `ModalityGroupedBatchSampler` weights: `{image, video}` only
