# Train Boogu-Image with FlowGRPO

Last updated: 08/19/2026

RL post-training for [Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)
(text-to-image) and [Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit)
(TI2I editing) with the FlowGRPO trainer. Both checkpoints share the
`BooguImagePipeline` architecture, so a single adapter pair serves both; a
run is T2I or Edit depending on whether the dataset carries condition
images. The Turbo variants (4-step Decoupled-DMD, `BooguImageTurboPipeline`)
are out of scope until vllm-omni supports that pipeline class.

## Prerequisites

On top of the [standard install](../../../docs/start/install.md):

```bash
pip install "boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git"
```

The training engine loads the checkpoint's canonical
`BooguImageTransformer2DModel` through `diffusers.AutoModel` with
`trust_remote_code=True`. The checkpoint's `transformer/transformer_boogu.py`
is a shim that re-exports the class from the `boogu` package, so that package
must be importable on trainer workers. Rollout workers do not need it —
vllm-omni ships its own Boogu pipeline port.

## Data

```bash
python examples/flowgrpo_trainer/data_process/boogu_image_ocr.py \
    --input_dir ~/data/ocr --output_dir ~/data/ocr/boogu_image
```

The converter's chat template ports the upstream Boogu system prompts
verbatim. Note the deliberate quirk: the upstream pipeline encodes empty
instructions — including the default negative prompt `""` — with the *TI2I
unified* system prompt, not the T2I one. Do not "fix" this; a template
mismatch between the data and the released model silently shifts the policy's
prompt distribution and collapses rewards.

## Launch

```bash
bash examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora.sh
```

Model-specific constraints baked into the recipe:

- `tensor_model_parallel_size=1` — the vllm-omni `BooguImagePipeline`
  supports neither TP, SP, CFG-parallel, nor HSDP. The 20 GB bf16 DiT plus
  the Qwen3VL encoder and VAE must fit on one rollout GPU.
- `pipeline.guidance_scale=4.0` — Boogu uses standard sequential text CFG
  (upstream default 4.0), driven by `guidance_scale`, not Qwen-style
  `true_cfg_scale`. Set `1.0` to disable CFG (halves rollout NFE).
- `pipeline.height/width` must be multiples of 16 and at most 2048².
- LoRA `target_modules` name the double-stream attention processors
  (`img_to_*`, `instruct_to_*`), single-stream/refiner attention (`to_*`),
  and both feed-forward variants — these match the released checkpoint's
  parameter tree.

## Edit (TI2I) specifics

- Boogu conditioning is **not** concat-and-crop: the reference latents enter
  the transformer through the `ref_image_hidden_states` refiner path and the
  transformer output is already target-length.
- The parquet follows the I2I schema (an `images` column of
  `{"bytes": ...}` dicts plus an `<image>` placeholder in the user turn);
  one reference image per sample.
- `align_res`: the output resolution follows the (VAE-preprocessed)
  reference image dimensions; `pipeline.height/width` act as an upper bound
  via the 2048² clamp, so keep condition images at the resolution you want
  to generate at (multiples of 16).
- Known deviation from upstream inference: the VLM copy of the reference is
  sized by the checkpoint processor's own rules — identical to how the agent
  loop tokenized the prompt — instead of upstream's 384² downscale cap. This
  keeps the image-placeholder token count consistent between tokenization
  and encoding; validate reward quality against upstream inference when
  changing image resolutions.
- Text CFG keeps the reference latents in the *unconditional* forward
  (upstream behaviour); the negative instruction itself is encoded
  text-only. Image guidance (`guidance_scale_2` / double guidance) is not
  wired up.

## Tests

The pipeline is guarded by model-level CPU tests, which cover the conventions
that break RL silently rather than loudly — the velocity negation and text CFG
that reach `scheduler.step`, the `t = 1 - sigma` timestep mapping, the float32
trajectory rule, and the SDE-window collection contract:

```bash
pytest tests/pipelines/test_boogu_image_adapters_on_cpu.py \
       tests/pipelines/test_boogu_image_common_on_cpu.py \
       tests/pipelines/test_boogu_image_rollout_on_cpu.py
```

A special E2E test covers parquet loading, vLLM-Omni rollout, reward
computation, FSDP LoRA training, and weight synchronization. Boogu-Image
shares the Qwen-Image training path, so it is **not** wired into the GPU smoke
suite; run it manually when touching the rollout or training adapters:

```bash
# T2I
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash tests/special_e2e/run_flowgrpo_boogu_image.sh
# Edit (TI2I) — also exercises the reference-latent refiner path
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MODE=edit bash tests/special_e2e/run_flowgrpo_boogu_image.sh
```

The script builds a tiny random checkpoint at
`~/models/tiny-random/Boogu-Image` on first use (a no-op afterwards). The
builder copies the processor and scheduler directories verbatim from the
locally cached `Boogu/Boogu-Image-0.1-Base` snapshot without loading its weight
shards — the scheduler config carries Boogu-specific time-shift keys that a
diffusers config round-trip would drop. Use `SOURCE_MODEL` if those assets are
stored elsewhere:

```bash
SOURCE_MODEL=/path/to/Boogu-Image-0.1-Base MODE=edit \
bash tests/special_e2e/run_flowgrpo_boogu_image.sh
```
