# How to add a new Diffusion Model

Last updated: 04/27/2026.

This guide walks through everything you need to do to plug a new diffusion
model into VeRL-Omni so that it can be used end-to-end for FlowGRPO-style RL
training. We use the **Z-Image** integration as a worked example – see the
files added under [`verl_omni/pipelines/z_image_flow_grpo/`](../../verl_omni/pipelines/z_image_flow_grpo/__init__.py)
and the matching [`tests/special_e2e/run_flowgrpo_z_image.sh`](../../tests/special_e2e/run_flowgrpo_z_image.sh)
for the full source.

The Qwen-Image pipeline ([`verl_omni/pipelines/qwen_image_flow_grpo/`](../../verl_omni/pipelines/qwen_image_flow_grpo/__init__.py))
is the other reference. Whenever a step diverges between the two it is called
out as a "model-specific decision".

## 0. Prerequisites

A new diffusion model must already be supported by:

- **diffusers** — provides the transformer (`<Name>Transformer2DModel`),
  scheduler config and reference inference pipeline. The training adapter
  imports the transformer class from `diffusers.models.transformers`.
- **vllm-omni** — provides the rollout-side `<Name>Pipeline` that wraps the
  text encoder, transformer and VAE for serving. Our rollout adapter inherits
  from this class.

If either of these is missing you will need to upstream the model first.

## 1. Map the new model onto the verl-omni interfaces

Two registry-based interfaces drive everything:

| Interface | Purpose | Source |
|---|---|---|
| [`DiffusionModelBase`](../../verl_omni/pipelines/model_base.py) | Training-side adapter: scheduler construction, model-input building, single denoising step with log-probabilities. | `verl_omni/pipelines/model_base.py` |
| [`VllmOmniPipelineBase`](../../verl_omni/pipelines/model_base.py) | Rollout-side adapter: SDE sampling loop with log-probability collection inside the vllm-omni runtime. | `verl_omni/pipelines/model_base.py` |

Both registries dispatch on the `_class_name` field of the model's
`model_index.json`, which is auto-detected into
[`DiffusionModelConfig.architecture`](../../verl_omni/workers/config/diffusion/model.py).
Pick the same string for the `@register("...")` decorator on both adapters.

For Z-Image the architecture string is `"ZImagePipeline"`.

## 2. Identify per-model differences before writing code

Skim the upstream diffusers pipeline (`__call__`) and the vllm-omni rollout
pipeline (`forward`) and answer:

1. **Latents shape** — packed sequence (e.g. Qwen-Image
   `(B, seq, 4·C)`) or 4-D `(B, C, H, W)` (e.g. Z-Image)?
2. **Text encoder output shape** — fixed `(B, L, D)` plus a mask
   (Qwen-Image), or per-sample variable-length list (Z-Image)?
3. **Transformer signature** — what positional/keyword names does it use?
   Does it need extra metadata (`img_shapes`, `txt_seq_lens`, etc.)?
4. **Timestep convention** — does the model expect `t/1000`, `(1000-t)/1000`,
   or another rescaling?
5. **Output sign** — is the predicted velocity / noise negated before being
   handed to the scheduler?
6. **CFG flavour** — "true CFG" with renormalization (Qwen-Image) or
   standard CFG with optional clipping (Z-Image)? At what threshold is CFG
   active (`> 1` vs `> 0`)?
7. **VAE post-processing** — `latents / scaling_factor + shift_factor` vs
   `latents / std + mean`, etc.
8. **Prompt template** — does the upstream `_encode_prompt` prepend a
   hard-coded system prompt (Qwen-Image), or does it pass only the user
   message into `apply_chat_template` (Z-Image)? Your data preprocessor must
   match this exactly so training-time and inference-time tokenizations
   agree.

These answers determine which helpers you need and what the rollout/training
adapters must do differently. The Z-Image migration table below summarises
ours.

### Z-Image vs Qwen-Image at a glance

| Concern | Qwen-Image | Z-Image |
|---|---|---|
| Latents shape | `(B, seq, C·4)` packed | `(B, C, H, W)` |
| Latent unpack helper | `_unpack_latents` required | not needed |
| Text encoder hidden state | `hidden_states[-1]` | `hidden_states[-2]` |
| Prompt features | dense `(B, L, D)` + `mask` | per-sample list of `(L_i, D)` |
| Transformer extras | `img_shapes`, `encoder_hidden_states_mask`, `guidance` | none |
| Timestep transform | `t / 1000` | `(1000 - t) / 1000` |
| Output sign | as-is | negate (`noise_pred = -noise_pred`) |
| CFG formula | `apply_true_cfg` (norm-clipped true CFG) | `pos + s·(pos-neg)` + optional norm clip |
| CFG active when | `true_cfg_scale > 1` | `guidance_scale > 0` |
| VAE decode | `latents / std + mean` | `latents / scaling_factor + shift_factor` |
| Prompt template | hard-coded `prompt_template_encode` with system role + `drop_idx=34` | user-only message via `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`; no system prompt. **Pass `data.apply_chat_template_kwargs="{enable_thinking: true}"` so training-side tokenization matches rollout.** |
| Default negative prompt | `" "` | `""` |

Anything column-specific stays inside the model package
(`pipelines/<model>_flow_grpo/common.py`); anything shared is moved into
`pipelines/utils.py` or `pipelines/model_base.py`.

## 3. Lay out the new package

```
verl_omni/pipelines/<model>_flow_grpo/
├── __init__.py                       # re-exports + lazy vllm-omni import
├── common.py                         # constants and pure-Python helpers
├── diffusers_training_adapter.py     # DiffusionModelBase subclass
└── vllm_omni_rollout_adapter.py      # VllmOmniPipelineBase subclass
```

Use [`verl_omni/pipelines/qwen_image_flow_grpo/`](../../verl_omni/pipelines/qwen_image_flow_grpo/__init__.py)
or [`verl_omni/pipelines/z_image_flow_grpo/`](../../verl_omni/pipelines/z_image_flow_grpo/__init__.py)
as a starting template. Then register the new package by editing
[`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py) so
it is picked up by both registries when `verl_omni.pipelines` is imported.

The `__init__.py` should keep the vllm-omni import lazy / optional so the
training-only entrypoint still works without vllm-omni installed:

```python
from .diffusers_training_adapter import ZImage

__all__ = ["ZImage"]

try:
    from .vllm_omni_rollout_adapter import ZImagePipelineWithLogProb
except ImportError:
    ZImagePipelineWithLogProb = None

if ZImagePipelineWithLogProb is not None:
    __all__.append("ZImagePipelineWithLogProb")
```

## 4. Write `common.py`

Put every helper that depends only on `torch` here. Typical contents:

- A `<MODEL>_VAE_SCALE_FACTOR` constant.
- Functions that adapt verl-omni's batched representation to the model's
  native one: see
  [`split_padded_embeds_to_list`](../../verl_omni/pipelines/z_image_flow_grpo/common.py)
  and [`latents_to_transformer_input`](../../verl_omni/pipelines/z_image_flow_grpo/common.py).
- Functions that adapt the model's native output back to a batched tensor:
  see [`stack_transformer_output`](../../verl_omni/pipelines/z_image_flow_grpo/common.py).
- The CFG combiner: see
  [`apply_z_image_cfg`](../../verl_omni/pipelines/z_image_flow_grpo/common.py)
  vs [`apply_true_cfg`](../../verl_omni/pipelines/qwen_image_flow_grpo/common.py).

Keep these helpers stateless — they are exercised by CPU unit tests.

## 5. Write `diffusers_training_adapter.py`

Subclass `DiffusionModelBase`, decorate with the architecture string and
implement the four abstract classmethods.

### 5.1 `build_scheduler` & `set_timesteps`

Both Qwen-Image and Z-Image reuse
[`FlowMatchSDEDiscreteScheduler`](../../verl_omni/pipelines/schedulers/flow_match_sde.py)
because the FlowGRPO algorithm only requires a flow-matching scheduler that
exposes `sample_previous_step(...)`. If your model needs a different
flow-matching variant, add it under `verl_omni/pipelines/schedulers/` first.

Compute `image_seq_len` and `mu` exactly as the upstream diffusers pipeline
does so the noise schedule matches inference at deployment time.

### 5.2 `prepare_model_inputs`

Convert the *batched* tensors that the FSDP engine hands you into the
*native* model inputs. The base class slices `latents[:, step]` and
`timesteps[:, step]` for you; you only need to:

1. Apply any per-model timestep rescaling
   (e.g. `(1000 - timesteps[:, step]) / 1000` for Z-Image).
2. Convert padded embeddings + mask to the model's preferred format. For
   Z-Image we call `split_padded_embeds_to_list`; for Qwen-Image we pass the
   padded tensor and mask straight through.
3. Build the **positive** input dict and, when CFG is enabled, the
   **negative** input dict with the same latent / timestep but the negative
   text features.

The dict keys must exactly match the kwargs of the diffusers transformer
class, because the FSDP engine calls `module(**model_inputs)`.

### 5.3 `forward_and_sample_previous_step`

Call the transformer once for the positive prompt; if CFG is active, call it
again for the negative prompt and combine via your `common.py` helper.
Always end with `scheduler.sample_previous_step(...)` and return
`(log_prob, prev_sample_mean, std_dev_t)` — that triple is consumed by
[`DiffusersFSDPEngine.prepare_model_outputs`](../../verl_omni/workers/engine/fsdp/diffusers_impl.py).

> **Tip:** if your transformer returns a list (e.g. Z-Image), wrap the call
> in a helper such as `stack_transformer_output(...)` so the rest of the
> code keeps the `(B, C, H, W)` tensor convention.

## 6. Write `vllm_omni_rollout_adapter.py`

Subclass the upstream `<Name>Pipeline` from `vllm_omni.diffusion.models`.
The class must:

1. Replace the upstream Euler scheduler with `FlowMatchSDEDiscreteScheduler`.
2. Override `encode_prompt` so it accepts pre-tokenized `prompt_ids` and the
   tokenizer attention mask (the agent loop sends these — not raw strings).
   Always return a padded `(B, L, D)` tensor + `(B, L)` mask so the
   downstream agent loop's
   [`_agent_loop_postprocess`](../../verl_omni/agent_loop/diffusion_agent_loop.py)
   can transport them as plain tensors.
3. Provide a `diffuse(...)` method that runs the SDE loop, optionally
   applies CFG, and collects `all_latents`, `all_log_probs`, `all_timesteps`.
4. Override `forward(req, ...)` so that:
   - Sampling parameters are pulled from `req.sampling_params` (with
     `extra_args` for SDE-specific knobs).
   - `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds` and
     `negative_prompt_embeds_mask` end up in the returned
     `DiffusionOutput.custom_output`. The diffusion agent loop
     ([`diffusion_agent_loop.py`](../../verl_omni/agent_loop/diffusion_agent_loop.py))
     uses these field names verbatim, so do **not** rename them.

## 7. Wire into the training launcher

No code changes are needed beyond the package registration. At runtime:

- `DiffusionModelConfig.architecture` is auto-detected from `model_index.json`.
- `DiffusionModelBase.get_class(model_config)` resolves to your adapter.
- `VllmOmniPipelineBase.get_class("ZImagePipeline")` resolves to the rollout
  adapter and is consumed by the vllm-omni rollout worker via
  `get_pipeline_path(...)`.

Provide a runnable example shell script so users can launch training without
trial and error. For Z-Image we ship
[`examples/flowgrpo_trainer/run_z_image_ocr_lora.sh`](../../examples/flowgrpo_trainer/run_z_image_ocr_lora.sh)
and the data preprocessor
[`examples/flowgrpo_trainer/data_process/zimage_ocr.py`](../../examples/flowgrpo_trainer/data_process/zimage_ocr.py).

Important config knobs to set per model. **Always copy the defaults from the
upstream HuggingFace model card** so RL exploration starts from a known-good
operating point.

- `actor_rollout_ref.model.height` / `.width` — must be a multiple of
  `vae_scale_factor * 2` (e.g. `1024` for Z-Image; pixel area within
  `512×512` to `2048×2048` per the Z-Image model card).
- `actor_rollout_ref.rollout.num_inference_steps` — steps used **during training
  rollout**. The default is `10` regardless of the model's full-quality step
  count; do **not** override this in the run script unless you have a specific
  reason. Full-quality steps for validation are set separately via
  `actor_rollout_ref.rollout.val_kwargs.num_inference_steps` (e.g. `50` for
  both Qwen-Image and Z-Image).
- `actor_rollout_ref.rollout.true_cfg_scale` — for models that use verl-omni's
  Qwen-Image style true-CFG parameter name (e.g. **Qwen-Image**: `4.0`). Default
  `1.0` (disabled).
- `actor_rollout_ref.rollout.guidance_scale` — for models whose upstream
  diffusers / vllm-omni pipeline uses `guidance_scale` (e.g. **Z-Image**: `4.0`
  per HF card; Z-Image-Turbo requires `0.0` as it is distilled). Default `null`
  delegates to the pipeline's built-in default.
- `actor_rollout_ref.rollout.cfg_normalization` — bool flag matching the diffusers
  / vllm-omni `cfg_normalization` parameter. `false` (default) disables
  norm-clipping after CFG combination, matching the Z-Image HF card. Qwen-Image
  does not use this field.
- `actor_rollout_ref.rollout.max_sequence_length` — must accommodate the
  templated prompt length used by your tokenizer (Qwen-Image: `256`;
  Z-Image: `512`).

> **Config hygiene**: every new model-specific field added to
> `DiffusionRolloutConfig` must also appear in
> [`verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml`](../../verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml)
> with a matching default and a one-line comment. This keeps the generated
> config docs in sync and makes Hydra overrides discoverable.

## 8. Add tests

Add an end-to-end smoke test script under `tests/special_e2e/`. The script
should be modelled on
[`tests/special_e2e/run_flowgrpo_z_image.sh`](../../tests/special_e2e/run_flowgrpo_z_image.sh)
and must exercise the full pipeline with a `tiny-random/<ModelName>` checkpoint:

1. Generate dummy parquet data via `create_dummy_diffusion_data.py`.
2. Launch `verl_omni.trainer.diffusion.main_flowgrpo` with model-specific config
   knobs (architecture, prompt template, CFG parameters, sequence lengths).
3. Assert the script exits `0` (training completes without error).

## 9. When to refactor instead of duplicating

If you find yourself copy-pasting more than a few lines from another model's
adapter, prefer one of:

- Extending [`pipelines/utils.py`](../../verl_omni/pipelines/utils.py) with a
  new generic helper.
- Adding a new method to `DiffusionModelBase` or
  `VllmOmniPipelineBase` so that future models do not need to re-discover
  the contract.
- Promoting a helper from one model's `common.py` to a shared
  `pipelines/common.py` once two or more models need it.

Refactor opportunistically: keep model-specific quirks in the model's own
`common.py` until a third model demands the same code, then unify.

## 10. Checklist

Before opening the PR:

- [ ] `verl_omni/pipelines/<model>_flow_grpo/{__init__,common,diffusers_training_adapter,vllm_omni_rollout_adapter}.py` exist.
- [ ] `verl_omni/pipelines/__init__.py` re-exports the new package.
- [ ] E2E smoke test script exists: `tests/special_e2e/run_flowgrpo_<model>.sh`.
- [ ] Example script in `examples/flowgrpo_trainer/` and a matching data
      preprocessor under `examples/flowgrpo_trainer/data_process/`.
- [ ] Docs updated (this guide, plus the relevant `docs/algo/...` page if
      you introduce new algorithm-level concepts).
- [ ] PR title follows `[diffusion] feat: add <Name>Pipeline support`.
