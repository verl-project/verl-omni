# Train Qwen-Image-Edit-2511 with FlowGRPO

Last updated: 07/14/2026.

This guide shows how to prepare an image-edit dataset and train
[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
with LoRA and FlowGRPO.

The current adapter is registered for `QwenImageEditPlusPipeline`. A replacement
checkpoint must use that value in `model_index.json::_class_name`. The older
`QwenImageEditPipeline` architecture is not supported by this recipe.

For implementation details or support for another image-edit architecture, see
[How to Integrate an Image-to-Image Diffusion Model](../contributing/integrating_an_i2i_diffusion_model.md).

## Prerequisites

The example reward is PickScore. Its model weights are downloaded from Hugging
Face the first time the reward workers start.

## Prepare the Dataset

The example converter expects the following input layout:

```text
my_image_edit_data/
|-- images/
|   |-- 000001.png
|   `-- 000002.png
|-- train.jsonl
`-- test.jsonl
```

Each line of `train.jsonl` and `test.jsonl` contains an editing instruction and
an image path relative to `images/`:

```json
{"prompt": "Change the background to blue", "image": "000001.png"}
```

Convert the data to training parquet files:

```bash
python examples/flowgrpo_trainer/qwen_image_edit/prepare_data.py \
    --input_dir my_image_edit_data \
    --output_dir data/qwen_image_edit \
    --image_size 512
```

The command writes:

```text
data/qwen_image_edit/
|-- train.parquet
`-- test.parquet
```

The converter letterboxes each condition image onto a square canvas and stores
it as PNG bytes. The fields relevant to training follow this structure:

```python
{
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Picture 1: <image>Change the background to blue"},
    ],
    "negative_prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Picture 1: <image> "},
    ],
    "images": [{"bytes": condition_image_png_bytes}],
    "data_source": "image_edit",
    "reward_model": {
        "style": "model",
        "ground_truth": "Change the background to blue",
    },
    "extra_info": {
        "instruction": "Change the background to blue",
        "image": "000001.png",
    },
}
```

The number of `<image>` placeholders must match the number of entries in
`images`. Qwen-Image-Edit-2511 training currently requires exactly one
condition image per sample.

## Launch Training

The example recipe trains a LoRA adapter with PickScore:

```bash
WORKSPACE=$PWD \
NUM_GPUS_ACTOR_ROLLOUT_REWARD=8 \
bash examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora.sh \
    trainer.logger=console
```

By default, the launcher reads:

```text
$WORKSPACE/data/qwen_image_edit/train.parquet
$WORKSPACE/data/qwen_image_edit/test.parquet
```

Set `TRAIN_FILES` and `VAL_FILES` to use different parquet files.

### Environment Overrides

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_PATH` | `Qwen/Qwen-Image-Edit-2511` | Hugging Face model ID or local compatible pipeline path. |
| `TRAIN_FILES` | `$WORKSPACE/data/qwen_image_edit/train.parquet` | Training parquet path. |
| `VAL_FILES` | `$WORKSPACE/data/qwen_image_edit/test.parquet` | Validation parquet path. |
| `NUM_GPUS_ACTOR_ROLLOUT_REWARD` | `8` | Total GPUs visible to actor, rollout, and reward workers. |
| `ACTOR_SP` | `1` | Actor Ulysses sequence-parallel size. Values greater than one require the attention overrides below. |
| `ROLLOUT_TP` | `1` | Rollout tensor-parallel size. |
| `REWARD_WORKERS` | `4` | Asynchronous reward worker count. |
| `IMAGE_RESOLUTION` | `512` | Square target output resolution. |
| `MAX_PROMPT_LENGTH` | `8192` | Token and prompt-embedding length limit. |
| `REWARD_FUNCTION_PATH` | `pkg://verl_omni.utils.reward_score.pickscore_reward` | Reward module import path. |

The launcher selects `compute_score_pickscore` from the reward module.

Additional Hydra overrides can be appended to the command:

```bash
bash examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora.sh \
    actor_rollout_ref.rollout.n=4 \
    data.train_batch_size=8 \
    trainer.total_training_steps=20 \
    trainer.logger=console
```

Keep `actor_rollout_ref.rollout.n` greater than one for group-relative
advantages. When reducing GPU count, also reduce the training batch size,
rollout count, micro-batch sizes, and reward worker count to fit memory.

The launcher enables console, TensorBoard, and W&B logging by default. The
first command overrides this with `trainer.logger=console` so it can run
without W&B credentials. To use the launcher's default loggers, export
`WANDB_API_KEY` before starting training.

For `ACTOR_SP > 1`, use the SP-capable attention backends on both actor and
rollout:

```bash
ACTOR_SP=2 \
bash examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora.sh \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    trainer.logger=console
```

## Important Configuration

The example launcher sets the model-specific fields required by the adapter:

```text
actor_rollout_ref.model.algorithm=flow_grpo
actor_rollout_ref.rollout.name=vllm_omni
actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0
actor_rollout_ref.rollout.pipeline.height=512
actor_rollout_ref.rollout.pipeline.width=512
actor_rollout_ref.rollout.pipeline.num_inference_steps=12
actor_rollout_ref.rollout.algo.sde_type=sde
```

Do not remove the negative prompt while `true_cfg_scale > 1`. Positive and
negative prompt encoding both consume the condition image.

PickScore measures instruction/image alignment but does not directly enforce
source-image preservation. Use an edit-aware reward, or combine rewards, when
the task requires strict preservation of identity, layout, or background.

## Image and Sequence Constraints

The rollout and actor adapters validate the following constraints at their
respective execution stages:

- Each sample has exactly one condition image.
- The condition-image aspect ratio matches the target output aspect ratio.
- Samples in a rollout batch produce compatible packed latent lengths.
- With actor sequence parallelism, the combined target and condition token
  count is divisible by `ACTOR_SP`.
- `true_cfg_scale > 1` has a negative prompt.

The example converter and launcher both use square images. For non-square
training, use a custom converter that preserves the intended aspect ratio, then
set `actor_rollout_ref.rollout.pipeline.height` and
`actor_rollout_ref.rollout.pipeline.width` to that same ratio. Do not mix
condition-image aspect ratios in one batch.

Qwen-Image-Edit checkpoints may contain `processor/` without
`processor/config.json`. The adapter creates the missing config before the
driver loads the processor. The model directory must be writable when that
file is absent.

## Run the End-to-End Smoke Test

The special E2E test covers parquet loading, multimodal prompt processing,
vLLM-Omni rollout, trajectory transport, reward computation, FSDP LoRA
training, and weight synchronization:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
bash tests/special_e2e/run_flowgrpo_qwen_image_edit.sh
```

By default it expects a tiny random checkpoint at
`~/models/tiny-random/qwen-image-edit-plus`. Build it before the first run:

```bash
python tests/special_e2e/build_qwen_image_edit_plus_tiny_random.py \
    --output-dir ~/models/tiny-random/qwen-image-edit-plus
```

The builder copies tokenizer, processor, and scheduler assets from the locally
cached `Qwen/Qwen-Image-Edit-2511` snapshot without loading its weight shards.
Use `--source-model <local-path>` if those assets are stored elsewhere. The
builder does not download missing source assets.

Set `MODEL_PATH` on the smoke-test command to use another compatible tiny
checkpoint. A successful run ends with:

```text
FlowGRPO Qwen-Image-Edit e2e test passed (training completed successfully).
```

Training logs and generated validation images are written below
`$WORKSPACE/outputs/qwen_image_edit_lora/` by the full example launcher.
