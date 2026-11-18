# MiniMax H3 Image FlowGRPO

Last updated: 11/18/2026.

Text-to-image LoRA training using the released six-step PDD grid. The actor uses
Diffusers' native `MiniMaxH3ImageTransformer2DModel`; rollout uses vLLM-Omni's
`MiniMaxH3ImagePipeline`. The integration reuses the existing H3 agent loop,
FlowGRPO scheduler, loss, and LoRA weight mapping. Actor sequence parallelism is 1.

Training samples two adjacent nonterminal transitions with CPS noise 0.8 and
records both endpoints for replay. Validation keeps the release's Euler updates.

## Dependencies and weights

**This recipe requires Diffusers 0.43.0 or newer.** The repository-wide minimum
remains unchanged. Use an isolated environment and the vLLM-Omni
revision pinned by this repository. No Diffusers fork, source patch, injected
PDD head, empty-audio workaround, or replacement actor kernel is required.

The native release directory must provide `model_index.json`, `pdd_config.json`,
the native rollout weights and tokenizer/processor, plus the published Diffusers
actor weights under `transformer_diffusers/` (or set `ACTOR_PATH`). Keep the native
rollout's `pdd_fused_heads.safetensors`; the actor uses its own native PDD heads.
This recipe does not convert older video/audio checkpoints into image models.

Install through the repository's documented `uv` workflow, then explicitly select
`diffusers>=0.43.0` for this recipe. The native image model currently lacks
`PeftAdapterMixin`; the image adapter adds that interface with a temporary local
subclass, preserving the native forward and checkpoint layout. It uses the
native class directly once upstream includes the mixin.
CPU API tests do not establish a working accelerator environment.

## Minimal run

Run from the repository root after installing a compatible runtime. Provide
standard prompt parquet files with `prompt` chat messages, `data_source`,
`reward_model.ground_truth`, and `extra_info`. Existing OCR parquet files can be
reused; dataset preparation and reward services are not duplicated here.

```bash
export MODEL_PATH=/path/to/MiniMax-H3-Image
export TRAIN_FILES=/path/to/train.parquet
export VAL_FILES=/path/to/test.parquet
export OUTPUT_DIR=/path/to/flowgrpo-output
bash examples/flowgrpo_trainer/minimax_h3_image/run_minimax_h3_image_lora.sh
```

The default reward is the existing CPU JPEG-compressibility scorer, a minimal
training signal rather than an image-quality evaluation. Select another existing
scorer through `reward.custom_reward_function.path` and `.name` overrides.
Set `DEVICE=npu` after configuring the vendor runtime to select native NPU attention;
`NUM_DEVICES` and `ROLLOUT_TP` control the colocated actor/rollout pool.

## Validation and scope

CPU tests cover native PDD-head replay and gradients, LoRA lifecycle/export,
trajectory transport, and the release sigma grid. Real accelerator actor/rollout
log-probability alignment and end-to-end training remain to be validated for this
unpatched implementation. No training-quality or convergence claim is made.

This recipe includes FlowGRPO only. It derives image integration details from
the earlier [H3 Image work](https://github.com/verl-project/verl-omni/pull/890),
while reusing the H3 support already on main; no checkout of that PR is needed.
