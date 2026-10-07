# Train Boogu-Image with DiffusionNFT

Last updated: 09/24/2026

RL post-training for [Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)
(text-to-image) with the DiffusionNFT trainer, using a `vllm-omni` rollout and
`Qwen3-VL-8B-Instruct` as a visual generative reward model on an OCR-style task.

The launcher is [`run_boogu_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh). It is the Boogu
sibling of [`../qwen_image/run_qwen_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora.sh)
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
It writes `data_source: "ocr"` — the reward that scores these rows, not the task or the
trainer. Both trainers read `data_source` as `data.reward_fn_key` and the validation trainer
splices it into every metric key as `<val-core|val-aux>/<data_source>/<var>/<metric>`, so
T2I OCR runs log `val-core/ocr/reward/mean@1`. See the Edit section below for the naming rule.
Note the deliberate quirk documented there: the upstream pipeline encodes empty instructions
— including the default negative prompt `""` — with the *TI2I unified* system prompt, not the
T2I one. Do not "fix" this; a template mismatch between the data and the released model
silently shifts the policy's prompt distribution and collapses rewards.

The parquet must carry a `negative_prompt` column. Boogu is a *guided* model
(`pipeline.guidance_scale=4.0`), and both the training adapter and the rollout fail closed
when guidance is active without negative embeddings, rather than silently sampling unguided.

### Edit (TI2I) dataset

[`run_boogu_image_edit_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/boogu_image/run_boogu_image_edit_lora.sh) is the TI2I sibling of
the launcher above. It defaults to the edit-specialised
[`Boogu/Boogu-Image-0.1-Edit`](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit)
checkpoint rather than `Boogu-Image-0.1-Base`, which the T2I launcher above uses. It reads
a separate dataset, produced by the edit-specific converter:

```bash
python examples/flowgrpo_trainer/data_process/boogu_image_edit_ocr.py \
    --input_dir ~/data/ocr_edit --output_dir ~/data/ocr/boogu_image_edit_pickscore \
    --image_size 512 --reward pickscore
```

Each row pairs the source image in `images` with an "edit this text" instruction and a
`target_text`. The reward is **PickScore** — the same reward as the verified
[Qwen-Image-Edit TI2I recipe](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image_edit/README.md) — which
CLIP-encodes `reward_model.ground_truth` as the prompt and scores its similarity to the
generated image. `ground_truth` is therefore the **instruction**, not the target word; the
converter writes it that way and keeps `target_text` in `extra_info`. With the OCR GenRM
(rather than PickScore) a bare target word would be the right `ground_truth`, so the two are
not interchangeable. `--reward` selects the arm and writes both fields together, so they
cannot drift apart:

| `--reward` | `ground_truth` | `data_source` | scored by |
| --- | --- | --- | --- |
| `pickscore` (default) | the instruction | `pickscore` | `pickscore_reward.py` (CLIP) |
| `ocr` | the target text | `ocr` | `genrm_ocr.py` (Qwen3-VL GenRM) |

`data_source` is the reward, and it is not cosmetic: both trainers read it as
`data.reward_fn_key` and the validation trainer splices it into every metric key as
`<val-core|val-aux>/<data_source>/<var>/<metric>`. Naming it after the task
(`ocr_edit`) or the algorithm (`flow_grpo/...`) makes `val-core/pickscore/reward/mean@1`
impossible to read and mislabels diffusionnft runs as flow_grpo; the T2I/EDIT
distinction belongs in `trainer.experiment_name`.

PickScore runs CLIP locally in the reward workers, so this recipe serves no reward model.

The converter writes a **text-only** negative prompt (`negative_prompt` carries no `<image>`
placeholder) for the edit arm: guided TI2I encodes the negative instruction without the reference
image (upstream default `use_input_images_4_neg_instruct=False`), so a placeholder there would be
tokenized but never expanded into image features — it would satisfy the media-count check while
quietly shifting the guidance. `RLHFDataset._build_messages` therefore permits the negative key to
consume **fewer** media than the row carries (`image_offset <= len(images)`), and only the negative
key; the positive prompt still has to match exactly. The T2I converter (`boogu_image_ocr.py`) and
the edit e2e fixture (`tests/special_e2e/create_dummy_image_edit_data.py --negative-prompt-mode
text-only`) emit the same text-only form. Qwen-Image-Edit's converter instead keeps the placeholder, but that is a different model family
whose negative branch does feed the image; do not copy it here.

The name is baked into the parquet at conversion time, so an existing dataset keeps its old key
until it is regenerated or rewritten in place with
[`set_data_source.py`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/data_process/set_data_source.py). See
[metrics](../../../docs/start/metrics.md) for what the per-step DiffusionNFT keys mean.

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
