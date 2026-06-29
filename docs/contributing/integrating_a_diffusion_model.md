# How to Integrate a New Diffusion Model for FlowGRPO Training

Last updated: 06/02/2026.

This guide walks you through everything required to integrate a new diffusion
model into VeRL-Omni so it can be trained end-to-end with the **FlowGRPO**
algorithm. The contracts described below (registry hooks, adapter
classmethods, scheduler choice, custom-output field names) are specific to
the FlowGRPO trainer; other RL algorithms may impose different requirements.
Use
[`integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md`](integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md)
for PPO-like policy-gradient algorithms, and
[`integrating_a_new_direct_preference_algorithm_for_diffusion_model.md`](integrating_a_new_direct_preference_algorithm_for_diffusion_model.md)
for direct-preference algorithms.

We use the **Qwen-Image** integration
([`verl_omni/pipelines/qwen_image_flow_grpo/`](../../verl_omni/pipelines/qwen_image_flow_grpo/__init__.py))
as the worked example throughout. Read the source alongside this guide — the
code is the canonical reference.

---

## TL;DR

A new model needs **three files in one new package** plus **two registry
hooks**. The same adapters work with both the default diffusers + FSDP2
backend and the optional [VeOmni backend](#53-use-the-veomni-backend-optional) — backend selection is purely a configuration concern.

```
verl_omni/pipelines/<model>_flow_grpo/
├── __init__.py                       # re-exports both adapters
├── diffusers_training_adapter.py     # subclass of DiffusionModelBase
└── vllm_omni_rollout_adapter.py      # subclass of VllmOmniPipelineBase
```

Both adapters are picked up by string-based registries that dispatch on
the pair `(model_index.json::_class_name, algorithm)`. By default the
algorithm is read from `actor_rollout_ref.model.algorithm`, which is the
source-of-truth in the current trainer wiring. Register the package by importing it
from
[`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py),
add an example launch script, and add a smoke test.

---

## Mental Model

There are two execution contexts you must serve, and they share the **same
algorithm** (FlowGRPO) but use **different runtimes**:

| Context | Runtime | What you implement |
|---|---|---|
| **Rollout** (sampling trajectories) | vllm-omni | `VllmOmniPipelineBase` subclass — runs the SDE loop and returns latents, log-probs, and prompt embeddings. |
| **Training** (per-step forward + loss) | FSDP + diffusers | `DiffusionModelBase` subclass — re-runs one denoising step per micro-batch slot to compute fresh log-probs for the policy gradient. |

The trainer runtime can also be VeOmni's FSDP2-based DiT trainer; see [§ 5.3](#53-use-the-veomni-backend-optional). The training-adapter contract (`prepare_model_inputs` / `forward_and_sample_previous_step`) is identical on both backends.

```text
  ┌─────────────────────────┐                ┌──────────────────────────┐
  │ Rollout worker          │   trajectory   │ Trainer worker           │
  │ (vllm-omni)             │ ─────────────▶ │ (FSDP + diffusers)       │
  │                         │  latents,      │                          │
  │ VllmOmniPipelineBase    │  log_probs,    │ DiffusionModelBase       │
  │  └─ diffuse() + SDE     │  prompt embeds │  └─ prepare_model_inputs │
  │                         │                │  └─ forward_and_sample…  │
  └─────────────────────────┘                └──────────────────────────┘
```

The two adapters must agree on:

- **Architecture string** (the first `@register(...)` argument). It must
  match `model_index.json::_class_name` exactly. For Qwen-Image this is
  `"QwenImagePipeline"`.
- **Algorithm string** (the `algorithm=` keyword on `@register(...)`).
  For this guide the value is always `"flow_grpo"`. When integrating a
  different RL algorithm use the appropriate algorithm name and the matching
  algorithm-family guide.
- **Prompt-encoding format** of the embeddings shipped through the agent
  loop. The rollout always returns padded `(B, L, D)` + `(B, L)` mask;
  the training adapter is free to convert to whatever the transformer
  needs.
- **Scheduler choice** so log-probs computed on each side are comparable.

---

## Prerequisites

Before you start, the new model must already be supported upstream by:

- **diffusers** — provides the transformer (`<Name>Transformer2DModel`),
  scheduler config, and a reference inference pipeline.
- **vllm-omni** — provides the rollout-side `<Name>Pipeline`. Your
  rollout adapter inherits from this class.

If either is missing, upstream the model first. Nothing below will work
without them.

---

## Step 1 — Read the Upstream Pipelines and Note the Differences

Open the upstream diffusers pipeline (`__call__`) and the vllm-omni
rollout pipeline (`forward`). Answer these questions before writing any
code — the answers determine every helper you need:

1. **Latent shape.** Packed sequence `(B, seq, 4·C)` (Qwen-Image) or 4-D
   `(B, C, H, W)`?
2. **Text encoder output.** Fixed `(B, L, D)` plus a mask, or a list of
   variable-length per-sample tensors?
3. **Transformer signature.** What kwargs does it accept? Any extras
   (`img_shapes`, `txt_seq_lens`, `guidance`, …)?
4. **Timestep convention.** `t/1000`? `(1000 - t)/1000`? Something else?
5. **Output sign.** Is the predicted velocity / noise negated before
   being passed to the scheduler?
6. **CFG flavour.** "True CFG" with renormalisation? Standard CFG with
   optional norm clipping? At what threshold is CFG active?
7. **VAE post-processing.** `latents / scaling_factor + shift_factor`,
   `latents / std + mean`, or other?
8. **Prompt template.** Does the upstream `_encode_prompt` prepend a
   hard-coded system prompt? Whatever it does, your **data preprocessor
   must match exactly** so training-time and inference-time tokenisation
   agree.

Anything model-specific belongs inside the model's own package;
anything reusable belongs in
[`pipelines/utils.py`](../../verl_omni/pipelines/utils.py) or
[`pipelines/model_base.py`](../../verl_omni/pipelines/model_base.py).

---

## Step 2 — Scaffold the Package

Create the new package and start by copying
[`verl_omni/pipelines/qwen_image_flow_grpo/`](../../verl_omni/pipelines/qwen_image_flow_grpo/__init__.py)
as a template:

```
verl_omni/pipelines/<model>_flow_grpo/
├── __init__.py
├── diffusers_training_adapter.py
└── vllm_omni_rollout_adapter.py
```

The `__init__.py` re-exports both adapters so the `@register(...)`
decorators run on import — follow the existing Qwen-Image pattern:

```python
from .diffusers_training_adapter import MyModel
from .vllm_omni_rollout_adapter import MyModelPipelineWithLogProb

__all__ = ["MyModel", "MyModelPipelineWithLogProb"]
```

Finally, **register the package** by adding a star-import to
[`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py)
so both registries learn about your model when `verl_omni.pipelines` is
imported:

```python
from .qwen_image_flow_grpo import *  # noqa: F401, F403
from .my_model_flow_grpo import *    # noqa: F401, F403

__all__ = list(qwen_image_flow_grpo.__all__)
__all__ += my_model_flow_grpo.__all__
```

> **Note.** `vllm_omni` is a hard dependency of `verl-omni`, so the
> rollout adapter import does not need to be guarded.

---

## Step 3 — Write `diffusers_training_adapter.py`

Subclass [`DiffusionModelBase`](../../verl_omni/pipelines/model_base.py),
decorate it with the architecture string, and implement the four
classmethods:

```python
@DiffusionModelBase.register("MyModelPipeline", algorithm="flow_grpo")
class MyModel(DiffusionModelBase):
    @classmethod
    def build_scheduler(cls, model_config): ...

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device): ...

    @classmethod
    def prepare_model_inputs(cls, module, model_config, latents, timesteps,
                             prompt_embeds, prompt_embeds_mask,
                             negative_prompt_embeds, negative_prompt_embeds_mask,
                             micro_batch, step): ...

    @classmethod
    def forward_and_sample_previous_step(cls, module, scheduler, model_config,
                                         model_inputs, negative_model_inputs,
                                         scheduler_inputs, step): ...
```

### 3.1 `build_scheduler` and `set_timesteps`

Reuse
[`FlowMatchSDEDiscreteScheduler`](../../verl_omni/pipelines/schedulers/flow_match_sde.py)
unless you have a strong reason not to — FlowGRPO only requires a
flow-matching scheduler that exposes `sample_previous_step(...)`.

Compute `image_seq_len` and `mu` exactly as the upstream diffusers
pipeline does. If they drift, the training-time noise schedule will not
match deployment.

### 3.2 `prepare_model_inputs`

This method receives the **full** batched tensors for the entire
denoising trajectory (`latents` of shape `(B, T, ...)`, `timesteps` of
shape `(B, T)`) together with the `step` index. Your implementation is
responsible for slicing to the current step, e.g.
`latents[:, step]` and `timesteps[:, step]`, before building model
inputs. The typical steps are:

1. Slice `latents[:, step]` and `timesteps[:, step]` for the current
   denoising step.
2. Apply per-model timestep rescaling.
3. Convert padded prompt embeddings + mask to whatever format your
   transformer expects.
4. Build the **positive** input dict and, if CFG is enabled, the
   **negative** input dict (same latent + timestep, negative text
   features).

The dict keys must match the kwargs of the diffusers transformer
class verbatim — the FSDP engine calls `module(**model_inputs)`.

### 3.3 `forward_and_sample_previous_step`

Call the transformer once for the positive prompt; if CFG is active,
call it again for the negative prompt and combine them. Always finish with
`scheduler.sample_previous_step(...)` and return the triple
`(log_prob, prev_sample_mean, std_dev_t)` — that is what
[`PPODiffusersFSDPEngine.prepare_model_outputs`](../../verl_omni/workers/engine/fsdp/diffusers_impl.py)
consumes.

> **Tip.** If your transformer returns a list (one element per sample),
> wrap the call in a small helper that re-stacks to `(B, C, H, W)` so
> the rest of the pipeline keeps a single tensor convention.

---

## Step 4 — Write `vllm_omni_rollout_adapter.py`

Subclass the upstream `<Name>Pipeline` from `vllm_omni.diffusion.models`
and decorate with the same architecture/algorithm pair:

```python
@VllmOmniPipelineBase.register("MyModelPipeline", algorithm="flow_grpo")
class MyModelPipelineWithLogProb(MyModelPipeline):
    ...
```

Your subclass must do four things:

1. **Replace the upstream scheduler** (typically Euler-based) with
   `FlowMatchSDEDiscreteScheduler`.
2. **Override `encode_prompt`** to accept pre-tokenised `prompt_ids` and
   the tokenizer attention mask (the agent loop ships these — never raw
   strings). Always return a padded `(B, L, D)` tensor and a `(B, L)`
   mask so the agent loop can ferry them as plain tensors.
3. **Implement `diffuse(...)`** — the SDE loop that optionally applies
   CFG and collects `all_latents`, `all_log_probs`, and
   `all_timesteps`.
4. **Override `forward(req, ...)`** so that:
   - Sampling parameters come from `req.sampling_params` (use
     `extra_args` for SDE-specific knobs).
   - `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds`,
     and `negative_prompt_embeds_mask` are placed in the returned
     `DiffusionOutput.custom_output`. The diffusion agent loop
     ([`diffusion_agent_loop.py`](../../verl_omni/agent_loop/diffusion_agent_loop.py))
     reads these field names verbatim — **do not rename them**.

---

## Step 5 — Configure the Pipeline

No code changes are required in the trainer launcher itself. At runtime:

- `DiffusionModelConfig.architecture` is auto-detected from
  `model_index.json`.
- `DiffusionModelConfig.algorithm` is set by
  `actor_rollout_ref.model.algorithm` (default `flow_grpo` in
  `diffusion_model.yaml`). `algorithm.adv_estimator` is templated to read
  from this same value.
- `DiffusionModelBase.get_class(model_config)` resolves to the training
  adapter registered under `(architecture, algorithm)`.
- `VllmOmniPipelineBase.get_class(architecture, algorithm)` resolves to
  the rollout adapter and is consumed by the vllm-omni rollout worker.

### 5.1 Pipeline Config Knobs

Pipeline sampling parameters live under `actor_rollout_ref.rollout.pipeline.*`
(mapped to
[`DiffusionPipelineConfig`](../../verl_omni/workers/config/diffusion/rollout.py))
and are mirrored in `actor_rollout_ref.model.pipeline.*`.

**Always copy the defaults from the upstream HuggingFace model card** so
RL exploration starts from a known-good operating point.

| Knob | Notes |
|---|---|
| `pipeline.height`, `pipeline.width` | Must be a multiple of `vae_scale_factor * 2`. |
| `pipeline.num_inference_steps` | Steps used **during training rollout**. Default `10` — do not override unless you know why. |
| `actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps` | Full-quality steps for **validation only** (e.g. `50`). |
| `pipeline.true_cfg_scale` | For Qwen-Image-style true CFG (e.g. `4.0`). Default `1.0` (disabled). |
| `pipeline.guidance_scale` | For pipelines whose upstream uses `guidance_scale`. Default `null` defers to the pipeline. |
| `pipeline.max_sequence_length` | Must accommodate the templated prompt length your tokenizer produces. |

> **Config hygiene.** Any new field on
> [`DiffusionPipelineConfig`](../../verl_omni/workers/config/diffusion/rollout.py)
> must also be added to:
>
> - [`diffusion_rollout.yaml`](../../verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml)
>   — both the top-level `pipeline:` section and `val_kwargs.pipeline:`.
> - [`diffusion_model.yaml`](../../verl_omni/trainer/config/diffusion/model/diffusion_model.yaml)
>   — its `pipeline:` section, using
>   `${oc.select:actor_rollout_ref.rollout.pipeline.<field>,<default>}`.

### 5.2 Example Launch Script and Data Preprocessor

Ship a runnable example so users can launch training without trial and
error. Use
[`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`](../../examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh)
and
[`examples/flowgrpo_trainer/data_process/qwenimage_ocr.py`](../../examples/flowgrpo_trainer/data_process/qwenimage_ocr.py)
as templates.

The data preprocessor's tokenisation **must match the upstream
`_encode_prompt` exactly** — same chat template, same special tokens,
same `enable_thinking` flag, etc. Mismatches here cause silent reward
collapse.

(53-use-the-veomni-backend-optional)=
### 5.3 Use the VeOmni Backend

Backend selection is **orthogonal** to model integration: the adapters you wrote in Steps 3–4 work unchanged regardless of whether the actor runs on the default diffusers + FSDP2 engine or on [VeOmni](https://github.com/ByteDance-Seed/VeOmni). Switching is a configuration concern handled by a few Hydra overrides at launch time.

#### What VeOmni reuses from your model adapter

- `DiffusionModelBase` subclass (Step 3) — used verbatim. The VeOmni engine calls the same `prepare_model_inputs` / `forward_and_sample_previous_step` contract.
- `VllmOmniPipelineBase` subclass (Step 4) — used verbatim. Rollout always runs in vllm-omni, independent of the actor backend.
- `FlowMatchSDEDiscreteScheduler` (Step 3.1) — used verbatim.

#### What VeOmni requires that diffusers does not

1. **Upstream support in VeOmni.** Just as diffusers must provide your `<Name>Transformer2DModel`, VeOmni must be able to load your model via its `DiTTrainer` path. If VeOmni does not yet support the architecture, upstream it there first (the diffusers prerequisite from Step 1 still applies for rollout — both upstreams are required).
2. **`config_path` / `transformer_subfolder`.** The VeOmni engine loads the transformer from `<local_path>/<transformer_subfolder>` and the config from `config_path` (falling back to the weights path). These fields are already on `DiffusionModelConfig` and are shared with the diffusers backend, so no new model-specific fields are needed.

#### Launching with the VeOmni backend

`diffusion/model_engine=veomni_diffusion` switches the entire actor / reference Hydra schema; the other actor-engine fields then live under `actor_rollout_ref.actor.veomni_config.*` and `actor_rollout_ref.ref.veomni_config.*`:

```bash
python3 -m verl_omni.trainer.main_diffusion \
    diffusion/model_engine=veomni_diffusion \
    actor_rollout_ref.actor.strategy=veomni \
    actor_rollout_ref.actor.veomni_config.strategy=veomni \
    actor_rollout_ref.ref.veomni_config.strategy=veomni \
    ...  # everything else identical to your diffusers/FSDP2 recipe
```

See [`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh`](../../examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh) for a complete VeOmni recipe that mirrors [`run_qwen_image_ocr.sh`](../../examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr.sh) line-for-line — the diff is only the engine-selection fields. Install instructions for VeOmni alongside vLLM 0.20.2 are in [`docs/start/install.md`](../start/install.md#optional-engine-backends).


#### Mixing override schemas — don't

`diffusion/model_engine=veomni_diffusion` selects the Hydra schema as a whole. Do **not** mix `actor.fsdp_config.*` and `actor.veomni_config.*` overrides in the same run — the fields for the other engine will be rejected as unknown keys at config-resolution time.

---

## Step 6 — Add a Smoke Test

Add an end-to-end smoke test under `tests/special_e2e/` modelled on
[`tests/special_e2e/run_flowgrpo_qwen_image.sh`](../../tests/special_e2e/run_flowgrpo_qwen_image.sh).
The script must exercise the full pipeline against a `tiny-random/<ModelName>`
checkpoint:

1. Generate dummy parquet data via
   [`tests/special_e2e/create_dummy_diffusion_data.py`](../../tests/special_e2e/create_dummy_diffusion_data.py).
2. Launch `verl_omni.trainer.main_diffusion` with model-specific
   knobs (architecture, prompt template, CFG parameters, sequence
   lengths).
3. Assert exit code `0`.

Then register the script in
[`tests/gpu_smoke/run_gpu_smoke_tests.sh`](../../tests/gpu_smoke/run_gpu_smoke_tests.sh)
as a new numbered test entry. The runner already exports
`PYTHONUNBUFFERED=1` and `RAY_DEDUP_LOGS=0` for readable logs — no need
to set them in your script.

---

## When to Refactor Instead of Duplicating

If you are copy-pasting more than a few lines from another model's
adapter, prefer one of:

- Extending [`pipelines/utils.py`](../../verl_omni/pipelines/utils.py)
  with a generic helper.
- Adding a method to `DiffusionModelBase` or `VllmOmniPipelineBase` so
  future models do not re-discover the contract.
- Promoting a helper to a shared module once a second model needs it.

Refactor opportunistically: keep model-specific quirks local until a
third model demands the same code, then unify.

---

## Final Checklist

Before opening the PR, confirm every box:

- [ ] `verl_omni/pipelines/<model>_flow_grpo/` contains `__init__.py`,
      `diffusers_training_adapter.py`, and `vllm_omni_rollout_adapter.py`.
- [ ] [`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py)
      imports the new package.
- [ ] Architecture string on both `@register(...)` decorators matches
      `model_index.json::_class_name`; the `algorithm=` keyword matches
      the algorithm you are integrating against (e.g. `"flow_grpo"` for
      FlowGRPO).
- [ ] Scheduler returns latents in fp32 (no `model_output.dtype` cast in `step()`),
      `diffuse()` casts to model dtype before transformer forward and casts
      noise_pred to float32 before `scheduler.step()`
      — see [Common Pitfalls](common_pitfalls.md#float32-precision-loss-in-stored-rollout-latents).
- [ ] Any new `DiffusionPipelineConfig` field is mirrored in **both**
      [`diffusion_rollout.yaml`](../../verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml)
      and
      [`diffusion_model.yaml`](../../verl_omni/trainer/config/diffusion/model/diffusion_model.yaml).
- [ ] Example launch script in `examples/flowgrpo_trainer/` plus a
      matching data preprocessor under
      `examples/flowgrpo_trainer/data_process/`.
- [ ] Smoke test `tests/special_e2e/run_<algo>_<model>.sh` exists and
      is wired into
      [`tests/gpu_smoke/run_gpu_smoke_tests.sh`](../../tests/gpu_smoke/run_gpu_smoke_tests.sh).
- [ ] Docs updated (this guide if the contract changed; the relevant
      `docs/algo/...` page if you introduce algorithm-level concepts).
