# Qwen-Image SFT Example

This example provides a lightweight supervised fine-tuning entrypoint for Qwen-Image on CoRT-style atomic image data. It follows the data-loader, distributed sampler, checkpoint, and FSDP shape of the existing verl SFT trainer, but uses a diffusion flow-matching MSE objective instead of language-model cross entropy.

The trainer is a standalone PyTorch/diffusers script. It is not a Ray or Hydra worker integration.

## Supported Data

The dataset accepts atomic JSONL/JSON/parquet rows and CoRT k-turn metadata rows.

### CoRT k-turn Rows

This loader is aligned with the CoRT explicit/latent dataset schema:

| Field | Required | Meaning |
| --- | --- | --- |
| `sample_id` or `source_id` | Yes | Sample directory name or unique row id. |
| `prompt` | Yes | Original text-to-image prompt. |
| `num_turns` | Yes | Number of reflection/edit turns. `0`, `1`, and `2` are the common CoRT cases. |
| `img0`, `img1`, `img2` | Optional in meta JSONL | Inline image bytes/path fields. If absent, images are loaded from `--cort_intermediate_dirs`. |
| `reflection1`, `reflection2` | Required when the turn exists | Reflection text, or `{ "problem": "...", "fix": "..." }`. |
| `gt_img` | Optional | Final target image key such as `img0`, `img1`, or `img2`; defaults to `img{num_turns}`. |
| `source` | Optional | CoRT source namespace used to resolve intermediate directories. |
| `generator_model` | Optional | CoRT generator namespace used to resolve intermediate directories. |

CoRT row expansion follows the same turn semantics:

- `0-turn`: `prompt -> img0`, emitted as one `t2i` entry.
- `1-turn`: `prompt -> img0 -> reflection1 -> img1`, emitted as one `t2i` entry plus one `edit` entry from `img0` to `img1`.
- `2-turn`: `prompt -> img0 -> reflection1 -> img1 -> reflection2 -> img2`, emitted as one `t2i` entry plus two `edit` entries.

The intermediate directory layout matches CoRT:

```text
<intermediate_root>/<source>/<generator_model>/<sample_id>/img0.png
<intermediate_root>/<source>/<generator_model>/<sample_id>/img1.png
<intermediate_root>/<source>/<generator_model>/<sample_id>/meta.json
```

The loader also accepts the CoRT fallback layout `<intermediate_root>/<source>/<sample_id>/...` and flat `<intermediate_root>/<sample_id>/...`.

### Atomic Rows

Atomic rows are a convenience format for training or testing already-expanded entries. They use the same `t2i` / `edit` / `und` buckets as the CoRT Stage 1-1 losses (`gen_t2i`, `gen_edit`, `und`).

Atomic t2i rows:

```json
{"entry_type": "t2i", "sample_id": "sample_0", "prompt": "a green square", "target_image": "images/target.png"}
```

Atomic edit rows:

```json
{"entry_type": "edit", "sample_id": "sample_1", "prompt": "make it cyan", "source_image": "images/source.png", "target_image": "images/edited.png", "reflection": "<problem>it is yellow</problem>\n<fix>make it cyan</fix>"}
```

Atomic understanding rows are parsed for CoRT compatibility, but skipped by this trainer because Qwen-Image does not expose a text CE head in the diffusion objective.

Accepted entry aliases are `gen_t2i -> t2i`, `gen_edit -> edit`, `image_edit -> edit`, and `understanding -> und`.

## Weights

Use the model family that matches the entries you train:

- `Qwen/Qwen-Image-Edit` for `edit` rows with `--pipeline_class qwen_image_edit`;
- `Qwen/Qwen-Image` for plain `t2i` rows with `--pipeline_class qwen_image`.

The diffusers `from_pretrained` call downloads weights automatically. To pre-download them:

```bash
huggingface-cli download Qwen/Qwen-Image-Edit --local-dir /data1/weights/Qwen-Image-Edit
huggingface-cli download Qwen/Qwen-Image --local-dir /data1/weights/Qwen-Image
```

## Tests

The CPU unit test builds temporary dummy CoRT/atomic data and verifies dataset expansion plus the loss path:

```bash
python -m pytest -q tests/trainer/diffusion/test_qwen_image_sft_example_on_cpu.py
```

## LoRA Training

Default edit-model LoRA run:

```bash
MODEL_NAME=/data1/weights/Qwen-Image-Edit \
TRAIN_FILES=/path/to/train.jsonl \
VAL_FILES=/path/to/val.jsonl \
CORT_INTERMEDIATE_DIR=/path/to/intermediate \
OUTPUT_DIR=/path/to/output \
bash examples/qwen_image_sft_trainer/run_qwen_image_sft_lora.sh
```

Set `VAL_FILES=` to omit validation.

For t2i-only LoRA training:

```bash
MODEL_NAME=/data1/weights/Qwen-Image \
TRAIN_FILES=/path/to/t2i_train.jsonl \
VAL_FILES=/path/to/t2i_val.jsonl \
TRAIN_ENTRY_TYPES=t2i \
bash examples/qwen_image_sft_trainer/run_qwen_image_sft_lora.sh --pipeline_class qwen_image
```

For edit rows with the base t2i pipeline, `--edit_as_t2i` is available, but it intentionally ignores the source image and trains only on the target image.

## NPU LoRA Training

The standalone trainer can run on Ascend NPUs when `torch_npu` and the Ascend CANN runtime are installed. The NPU launcher mirrors the GPU LoRA launcher and uses `torchrun` with HCCL:

```bash
MODEL_NAME=/data1/weights/Qwen-Image-Edit \
TRAIN_FILES=/path/to/train.jsonl \
VAL_FILES=/path/to/val.jsonl \
CORT_INTERMEDIATE_DIR=/path/to/intermediate \
OUTPUT_DIR=/path/to/output \
bash examples/qwen_image_sft_trainer/run_qwen_image_sft_lora_npu.sh
```

Useful environment variables are `ASCEND_HOME_PATH`, `ASCEND_RT_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `NNODES`, `MASTER_PORT`, `IMAGE_RESOLUTION`, and `TRAIN_ENTRY_TYPES`.

## Checkpoints

The trainer saves:

```text
<output_dir>/training_args.json
<output_dir>/global_step_<step>/transformer/
<output_dir>/global_step_<step>/trainer_state.pt
<output_dir>/global_step_<step>/training_args.json
```

Use `--resume_from <output_dir>/global_step_<step>` to restore optimizer and scheduler state. If LoRA is enabled, the saved transformer directory contains the PEFT-wrapped transformer state.
