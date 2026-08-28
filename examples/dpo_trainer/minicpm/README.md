# MiniCPM Offline DPO

This example trains MiniCPM multimodal understanding with offline DPO through
`verl_omni.trainer.main_omni`.  It uses turn-based preference rows and keeps the
training path simplex: vision/audio understanding modules and the AR language
model can be trained, while audio generation modules such as talker, codec,
TTS, audio decoder, and code2wav are excluded by default.

## Data

Convert Omni-Preference into the offline MLLM DPO parquet schema:

```bash
python examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py \
  --dataset_root "$HOME/Omni-Preference" \
  --output_dir "$HOME/Omni-Preference/parquet_dpo" \
  --modalities image video audio
```

The generated parquet schema is model-agnostic. MiniCPM-specific behavior is
handled later by `data.base_transform=minicpm` in the dataset transform, not by a
separate Omni-Preference converter.

## Training

```bash
DATASET_ROOT="$HOME/Omni-Preference" \
DATA_DIR="$DATASET_ROOT/parquet_dpo" \
MODEL_PATH=openbmb/MiniCPM-o-4_5 \
MODEL_ARCHITECTURE=MiniCPMOForConditionalGeneration \
bash examples/dpo_trainer/minicpm/run_minicpm_omni_preference_lora.sh
```

The script uses `AutoModel.from_pretrained(..., trust_remote_code=True)` through
the MiniCPM omni adapter.  Override `MINICPM_FROM_PRETRAINED_KWARGS` for
checkpoint-specific load flags, for example `{init_tts:false}` for MiniCPM-o.

Set `MINICPM_NO_SPLIT_MODULES` to the decoder layer names from the target
checkpoint if FSDP wrapping needs explicit layer boundaries.
