# BAGEL Example SFT


## Data Source

Data preparation flow for the example data:

```bash
wget -O bagel_example.zip \
  https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
unzip bagel_example.zip -d /data
```

Expected hierarchy:

```text
bagel_example
├── t2i/
├── editing/
│   ├── seedxedit_multi/
│   └── parquet_info/
└── vlm/
    ├── images/
    └── llava_ov_si.jsonl
```

The SFT example consumes this BAGEL example directory directly through
`BagelExampleSFTDataset`; no intermediate JSONL conversion is required.

```bash
python examples/sft_trainer/bagel/verify_bagel_example_data.py \
  --bagel_example_dir /data/bagel_example
```

The verifier smoke-checks the native BAGEL example layout:

```text
bagel_example
├── t2i/
├── editing/
│   ├── seedxedit_multi/
│   └── parquet_info/
└── vlm/
    ├── images/
    └── llava_ov_si.jsonl
```

The adapter reuses the BAGEL data readers for local `.json`, `.jsonl`, and
`.parquet` shards in the example dataset.

## Task Mapping

`bagel_example_data_config.yaml` mirrors BAGEL's native data grouping:

- `t2i_pretrain`: text-to-image.  No context image is consumed; each
  `<image_start>` opens the next target image from `image_list`.
- `unified_edit`: image/text editing.  `image_list[0]` is context, generated
  images are teacher-forced from later `image_list` entries.
- `vlm_sft`: visual-language supervised fine-tuning.  `image_list[0]` is
  context and text spans are trained with CE; no generated image is required.

The current SFT path supports BAGEL example-format data with `t2i_pretrain`,
`unified_edit`, and `vlm_sft` groups.

## Optional Preprocessing Columns

The dataset keeps heavy image encoders out of DataLoader workers.  A separate
preprocessing job can materialize these optional columns:

- `image_hidden_states`: noisy image latents or model-specific image tokens.
- `image_velocity_target`: flow target, usually `noise - clean_latent`.
- `image_loss_mask`: generated-image span mask.
- `timesteps`: sampled flow timesteps.
- `latent_pos_ids`: BAGEL latent patch position ids.

When these columns are present, `bagel_example_sft_collate_fn` stacks them into
the training batch and `BagelSFTDiffusersFSDPEngine` forwards them to the SFT
loss.

## Launch

```bash
bash examples/sft_trainer/bagel/run_bagel_example_lora.sh
```

Important defaults are based on the referenced BAGEL example-data config:

- `lr=2e-5`
- `lora_rank=256`
- `lora_alpha=512`
- `save_freq=500`
- `total_training_steps=3000`

Override `BAGEL_EXAMPLE_DIR`, `BAGEL_DATA_CONFIG`, `BAGEL_MODEL_PATH`, or
`NUM_GPUS` as needed.
