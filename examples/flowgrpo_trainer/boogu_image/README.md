# Train Boogu-Image with FlowGRPO

RL post-training for [Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)
(text-to-image) with the FlowGRPO trainer. The Edit (TI2I) checkpoint shares
the same pipeline architecture but is not integrated yet; Edit rollouts fail
with a clear error.

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
