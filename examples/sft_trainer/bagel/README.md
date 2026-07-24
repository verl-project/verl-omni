# BAGEL Uni-COT SFT


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

Convert the example data into the Uni-COT schema consumed by
`UniCOTSFTDataset`:

```bash
python examples/sft_trainer/bagel/prepare_unicot_sft_data.py \
  --bagel_example_dir /data/bagel_example \
  --output_dir /data/bagel_example/unicot_sft
```

The converter writes:

```text
bagel_example/unicot_sft
├── train.jsonl
├── val.jsonl
└── images/
```

Each JSONL row contains `image_list`, `instruction_list`, `output_text_list`,
and `task_type`. Local `.json`, `.jsonl`, and `.parquet` files are supported for
converted shards.

## Task Mapping

`unicot_data_config.yaml` mirrors BAGEL's native data grouping:

- `t2i_pretrain`: text-to-image.  No context image is consumed; each
  `<image_start>` opens the next target image from `image_list`.
- `unified_edit`: image/text editing.  `image_list[0]` is context, generated
  images are teacher-forced from later `image_list` entries.
- `vlm_sft`: visual-language supervised fine-tuning.  `image_list[0]` is
  context and text spans are trained with CE; no generated image is required.

Rows can specify `task_type` as `t2i`, `editing`, or `vlm_sft`.  If absent,
the dataset uses the default Uni-COT reasoning behavior, where the first image
is treated as context.

## Optional Preprocessing Columns

The dataset keeps heavy image encoders out of DataLoader workers.  A separate
preprocessing job can materialize these optional columns:

- `image_hidden_states`: noisy image latents or model-specific image tokens.
- `image_velocity_target`: flow target, usually `noise - clean_latent`.
- `image_loss_mask`: generated-image span mask.
- `timesteps`: sampled flow timesteps.
- `latent_pos_ids`: BAGEL latent patch position ids.

When these columns are present, `unicot_sft_collate_fn` stacks them into the
training batch and `BagelSFTDiffusersFSDPEngine` forwards them to the SFT loss.

## Launch

```bash
bash examples/sft_trainer/bagel/run_bagel_unicot_lora.sh
```

Important defaults are based on the referenced TorchUMM BAGEL Uni-COT config:

- `lr=2e-5`
- `lora_rank=256`
- `lora_alpha=512`
- `save_freq=500`
- `total_training_steps=3000`

Override `BAGEL_EXAMPLE_DIR`, `UNICOT_TRAIN_FILE`, `UNICOT_VAL_FILE`,
`UNICOT_DATA_CONFIG`, `BAGEL_MODEL_PATH`, or `NUM_GPUS` as needed.
