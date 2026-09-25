# Train Boogu-Image with DiffusionNFT

Last updated: 09/24/2026

RL post-training for [Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)
(text-to-image) with the DiffusionNFT trainer, using a `vllm-omni` rollout and
`Qwen3-VL-8B-Instruct` as a visual generative reward model on an OCR-style task.

The launcher is [`run_boogu_image_ocr_lora.sh`](run_boogu_image_ocr_lora.sh). It is the Boogu
sibling of [`../qwen_image/run_qwen_image_ocr_lora.sh`](../qwen_image/run_qwen_image_ocr_lora.sh)
and differs in the model path, the LoRA targets and FSDP layer prefixes, the guidance knob
(`pipeline.guidance_scale` rather than Qwen's `true_cfg_scale`), rollout TP=1, and the dataset.

## Prerequisites

On top of the [standard install](../../../docs/start/install.md):

```bash
pip install "boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git"
```

The training engine loads the checkpoint's canonical `BooguImageTransformer2DModel` through
`diffusers.AutoModel` with `trust_remote_code=True`. The checkpoint's
`transformer/transformer_boogu.py` is a shim that re-exports the class from the `boogu`
package, so that package must be importable on trainer workers. Rollout workers do not need
it — vllm-omni ships its own Boogu pipeline port.

## Data

```bash
python examples/flowgrpo_trainer/data_process/boogu_image_ocr.py \
    --input_dir ~/data/ocr --output_dir ~/data/ocr/boogu_image
```

The converter is shared with the FlowGRPO Boogu recipe and produces
`~/data/ocr/boogu_image/{train,test}.parquet`, which is where the launcher looks by default.
Note the deliberate quirk documented there: the upstream pipeline encodes empty instructions
— including the default negative prompt `""` — with the *TI2I unified* system prompt, not the
T2I one. Do not "fix" this; a template mismatch between the data and the released model
silently shifts the policy's prompt distribution and collapses rewards.

The parquet must carry a `negative_prompt` column. Boogu is a *guided* model
(`pipeline.guidance_scale=4.0`), and both the training adapter and the rollout fail closed
when guidance is active without negative embeddings, rather than silently sampling unguided.

### Edit (TI2I) dataset

[`run_boogu_image_edit_lora.sh`](run_boogu_image_edit_lora.sh) is the TI2I sibling of
the launcher above. It reads a separate dataset, produced by the edit-specific converter:

```bash
python examples/flowgrpo_trainer/data_process/boogu_image_edit_ocr.py \
    --input_dir ~/data/ocr_edit --output_dir ~/data/ocr/boogu_image_edit_pickscore --image_size 512
```

Each row pairs the source image in `images` with an "edit this text" instruction and a
`target_text`. The reward is **PickScore** — the same reward as the verified
[Qwen-Image-Edit TI2I recipe](../../flowgrpo_trainer/qwen_image_edit/README.md) — which
CLIP-encodes `reward_model.ground_truth` as the prompt and scores its similarity to the
generated image. `ground_truth` is therefore the **instruction**, not the target word; the
converter writes it that way and keeps `target_text` in `extra_info`. With the OCR GenRM
(rather than PickScore) a bare target word would be the right `ground_truth`, so the two are
not interchangeable — the directory name carries the distinction so they cannot be confused.
PickScore runs CLIP locally in the reward workers, so this recipe serves no reward model.

That converter deliberately emits a **text-only** negative prompt: guided TI2I encodes the
negative instruction without the reference image (`use_input_images_4_neg_instruct=False`), and
the reference latents reach the unconditional forward separately. A negative prompt that
references no media is therefore allowed to consume fewer media than the row carries. Do not
silence a load failure by adding `<image>` to the negative prompt -- that satisfies the media
count check while feeding the negative branch a placeholder token that is never expanded into
image features, which quietly shifts the guidance.

The converter also sets `data_source` to `diffusion_nft/pickscore_edit`, which is the label the
trainer files this dataset's validation reward under. `data_source` is the middle segment of
every validation key, so this run logs
`val-core/diffusion_nft/pickscore_edit/reward/mean@1` rather than inheriting the
`flow_grpo/ocr_edit` name its FlowGRPO ancestor used. Both segments have to be right: the recipe
is DiffusionNFT, not FlowGRPO, and the reward is PickScore, not OCR. Unlike the T2I dataset above
— which is shared with `examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora.sh` and
cannot be renamed without mislabelling genuine FlowGRPO runs — this one is read by no FlowGRPO
recipe. The name is stored in the parquet at conversion time, so an existing dataset keeps its
old key until it is regenerated. See [metrics](../../../docs/start/metrics.md) for what the
per-step DiffusionNFT keys mean.

## Launch

```bash
bash examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh
```

The script is configured for a single node with `4` GPUs. Model-specific constraints baked
into the recipe:

- `tensor_model_parallel_size=1` — the vllm-omni `BooguImagePipeline` supports neither TP nor
  SP nor CFG-parallel. The 20 GB bf16 DiT plus the Qwen3VL encoder and VAE must fit on one
  rollout GPU.
- `pipeline.guidance_scale=4.0` — Boogu uses standard sequential text CFG (upstream default
  `4.0`), driven by `guidance_scale` and not Qwen-style `true_cfg_scale`. Set `1.0` to disable
  CFG (halves rollout NFE).
- `pipeline.height/width` must be multiples of 16 and at most 2048².
- LoRA `target_modules` names the double-stream attention processors (`img_to_*`,
  `instruct_to_*`), single-stream and refiner attention (`to_*`, plus the joint attention's
  output projection), and both feed-forward variants. These are the paths the released
  checkpoint actually exposes and that FSDP layered-summon can transport to the rollout; a
  broader `all-linear` list names top-level embedders that never bind, so the sync would drop
  them. The list is validated at startup rather than allowed to fail silently.
- The objective is rebalanced away from the config defaults: `mix_beta=0.1`,
  `ref_kl_coef=10.0`, `adv_clip_max=1.0`, `clip_ratio=1e-5` (defaults are `0.5`, `0.0`, `5.0`,
  `1e-4`). With the defaults the reward term is compressed by the advantage clip while the
  reward-free contraction dominates the reported loss, and the graded reward collapses even
  though `positive_loss` looks like it is improving.

## Tests

Model-level CPU tests cover the conventions that break RL silently rather than loudly — the
velocity negation and text CFG that reach `scheduler.step`, the `sigma -> timestep` mapping,
the LoRA name translation, and the DiffusionNFT loss decomposition:

```bash
pytest tests/pipelines/test_boogu_image_diffusion_nft_on_cpu.py \
       tests/pipelines/test_boogu_image_diffusion_nft_engine_on_cpu.py \
       tests/pipelines/test_boogu_image_lora_mapping_on_cpu.py
```

The special E2E test covers parquet loading, vllm-omni rollout, reward computation, FSDP LoRA
training, and weight synchronization, and asserts the rollout engine actually bound every
actor delta (a pass with a dropped sync would be vacuous). It is wired into the GPU smoke
suite as `tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh`; run it directly with:

```bash
# T2I
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash tests/special_e2e/run_diffusionnft_boogu_image.sh
# Edit (TI2I) — also exercises the reference-latent refiner path
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MODE=edit bash tests/special_e2e/run_diffusionnft_boogu_image.sh
```
