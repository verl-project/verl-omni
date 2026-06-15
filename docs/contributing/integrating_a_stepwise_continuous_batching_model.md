# How to Add Continuous Batching (Stepwise) Support for a Diffusion Model

Last updated: 06/15/2026.

This guide explains how to extend an existing FlowGRPO (or MixGRPO) model
integration so that it supports **continuous-batching (stepwise) rollout**
in vllm-omni. You must already have a working full-forward integration
following
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md).

We use the Qwen-Image FlowGRPO stepwise adapter
([`verl_omni/experimental/qwen_image_flow_grpo_stepwise/`](../../verl_omni/experimental/qwen_image_flow_grpo_stepwise/__init__.py))
as the worked example.

---

## TL;DR

A stepwise adapter needs **one file in one experimental package** plus **one
registry entry**:

```
verl_omni/experimental/<model>_<algo>_stepwise/
├── __init__.py                       # re-exports the stepwise adapter
└── vllm_omni_rollout_adapter.py      # subclass of the standard rollout adapter
```

The adapter is picked up by a `_stepwise` registry alias that the async server
selects when `step_execution=True` in the rollout config. Only the rollout side
changes — the training adapter is **unchanged**.

---

## Mental Model

In standard (full-forward) mode, vllm-omni calls `forward()` once per request,
which runs the entire SDE diffusion loop internally. In stepwise (continuous-
batching) mode, the engine interleaves one denoising step from each in-flight
request rather than running one request's full trajectory before starting the
next.

```text
  Full-forward mode                        Stepwise (CB) mode
  ────────────────                         ─────────────────
  for each request:                        for each step:
      forward()  ← runs full SDE loop          for each in-flight request:
                                                    denoise_step()
                                                    step_scheduler()
                                                post_decode()  ← packages trajectory
```

The stepwise engine calls four methods in sequence per request lifetime:

| Phase | Method | Purpose |
|---|---|---|
| Setup | `prepare_encode()` | Encode prompt, initialise latents, scheduler, SDE state |
| Per-step | `denoise_step()` | Run transformer forward, return noise prediction |
| Per-step | `step_scheduler()` | One scheduler step with SDE noise + log-prob bookkeeping |
| Finalise | `post_decode()` | Decode final latents, package trajectory into `custom_output` |

Your stepwise adapter must override at least `prepare_encode`, `step_scheduler`,
and `post_decode`. `denoise_step` only needs an override if your model has
non-standard CFG or transformer kwargs.

---

## Prerequisites

1. A working full-forward integration with both `diffusers_training_adapter.py`
   and `vllm_omni_rollout_adapter.py` registered under the base algorithm
   (e.g. `flow_grpo`).
2. Your standard rollout adapter's `forward()` / `diffuse()` must already
   collect `all_latents`, `all_log_probs`, and `all_timesteps` — the stepwise
   overrides must replicate this bookkeeping.

---

## Step 1 — Understand the Engine Contract

The stepwise engine (vllm-omni side) manages a pool of `DiffusionRequestState`
objects. Each denoising step it:

1. Calls `denoise_step(input_batch)` on the batched latents/timesteps — your
   override receives a batched `input_batch` with `.latents`, `.timesteps`,
   `.guidance`, `.prompt_embeds`, etc.
2. Calls `step_scheduler(state, noise_pred)` **per request** with the
   noise prediction sliced back to per-request shape.

The engine feeds `noise_pred` from step 1 into each request's step 2, then
re-gathers `state.latents` for the next round. After the final step it calls
`post_decode(state)` and ships the `DiffusionOutput` to the HTTP server.

Your `prepare_encode` must populate **all** fields on `state` that the
per-step methods consume — the engine never calls `forward()` so there is no
fallback initialisation.

---

## Step 2 — Scaffold the Experimental Package

Create the package under `verl_omni/experimental/`:

```
verl_omni/experimental/<model>_<algo>_stepwise/
├── __init__.py
└── vllm_omni_rollout_adapter.py
```

The `__init__.py` re-exports the stepwise class:

```python
from .vllm_omni_rollout_adapter import MyModelPipelineWithLogProbStepwise

__all__ = ["MyModelPipelineWithLogProbStepwise"]
```

Register the package by importing it from
[`verl_omni/experimental/__init__.py`](../../verl_omni/experimental/__init__.py):

```python
from . import my_model_flow_grpo_stepwise
from .my_model_flow_grpo_stepwise import *  # noqa: F401, F403

__all__ += list(my_model_flow_grpo_stepwise.__all__)
```

> **Note.** The `verl_omni/__init__.py` already imports `verl_omni.experimental`,
> so no additional wiring to `__init__.py` is needed.

---

## Step 3 — Write the Stepwise Adapter

Subclass your standard rollout adapter and register with the `_stepwise`
algorithm suffix:

```python
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.my_model_flow_grpo.vllm_omni_rollout_adapter import (
    MyModelPipelineWithLogProb,
)

@VllmOmniPipelineBase.register("MyModelPipeline", algorithm="flow_grpo_stepwise")
class MyModelPipelineWithLogProbStepwise(MyModelPipelineWithLogProb):
    ...
```

The architecture string must match your `model_index.json::_class_name` exactly
(same as the standard adapter). The algorithm suffix is always
`<base_algorithm>_stepwise`.

### 3.1 `prepare_encode`

This is the most involved override. It must:

1. **Accept pre-tokenized `prompt_ids`** from `state.prompts[0]` (a dict with
   keys `prompt_token_ids`, `prompt_mask`, `negative_prompt_ids`,
   `negative_prompt_mask`). Provide a fallback that tokenizes raw text prompts
   for the engine's dummy warm-up run.
2. **Encode prompts** via `encode_prompt()` — return padded `(B, L, D)` +
   `(B, L)` mask tensors.
3. **Initialise latents** (random noise in fp32).
4. **Prepare timesteps** from `num_inference_steps`.
5. **Deep-copy the scheduler** per request so concurrent requests don't share
   mutable state.
6. **Set RoPE sequence lengths from padded embed width**, not from
   `mask.sum()`. See [RoPE text length mismatch](#rope-mismatch-cb) below.
7. **Resolve SDE/log-prob knobs** from `sampling.extra_args` (noise_level,
   sde_window_size, sde_window_range, sde_type, logprobs).
8. **Populate `state`** with every field that `step_scheduler` / `denoise_step`
   / `post_decode` will read: `prompt_embeds`, `prompt_embeds_mask`,
   `negative_prompt_embeds`, `negative_prompt_embeds_mask`, `latents`,
   `timesteps`, `step_index`, `scheduler`, `do_true_cfg`, `guidance`,
   `img_shapes`, `txt_seq_lens`, `negative_txt_seq_lens`, `sde_window`,
   `noise_level`, `sde_type`, `logprobs`, and empty lists for
   `all_latents`, `all_log_probs`, `all_timesteps`.
9. **Persist the generator** on `state.sampling.generator` so
   `step_scheduler` draws from the same RNG stream.
10. **For MixGRPO**: call `_maybe_make_progressive_window()` in `prepare_encode`
    before delegating via `super()`. See
    [§ 3.5](#35-mixgrpo-window-positioning).

The canonical reference is
[`QwenImagePipelineWithLogProbStepwise.prepare_encode`](../../verl_omni/experimental/qwen_image_flow_grpo_stepwise/vllm_omni_rollout_adapter.py).

<a id="rope-mismatch-cb"></a>
> **RoPE text length.** In continuous batching, vllm-omni pads prompt
> embeddings to a shared `target_seq_len`. If you compute RoPE `txt_seq_lens`
> from `mask.sum()` (valid token count), each request gets a *different* RoPE
> length even though embeddings share the same width — tokens beyond position
> 50 get wrong positional encoding, causing a rollout/training mismatch in
> `ppo_kl`. Always use `prompt_embeds.shape[1]` instead.

### 3.2 `step_scheduler`

Overrides the default (vanilla scheduler step) to mirror the per-iteration
body of `diffuse()`:

1. **Respect `sde_window`**: noise_level is 0.0 outside the window, and the
   configured `noise_level` inside.
2. **Log the initial latent** when entering the SDE window.
3. **Call `scheduler.step()`** with `noise_pred` cast to fp32.
4. **Store trajectory in fp32**: `new_latents.float()` goes into
   `state.all_latents`; log-prob and timestep go into their respective lists.
5. **Keep `state.latents` in fp32** — NOT model dtype. Under continuous
   batching the engine gathers latents across all in-flight requests; a
   freshly-added request has fp32 latents while a stepped request would have
   bf16 latents, producing a "Mixed dtypes" error. Keeping fp32 throughout
   makes the batch dtype consistent.
6. **Advance `state.step_index`**.

The canonical reference is
[`QwenImagePipelineWithLogProbStepwise.step_scheduler`](../../verl_omni/experimental/qwen_image_flow_grpo_stepwise/vllm_omni_rollout_adapter.py).

### 3.3 `denoise_step` (usually optional)

Override only if your model needs non-standard transformer kwargs or CFG
logic that differs from the parent class. The default `QwenImagePipeline`
implementation works for most models.

If you do override:

1. Cast `input_batch.latents` to the transformer's weight dtype for the forward
   pass (bf16).
2. Build positive/negative kwargs via your model's `_build_denoise_kwargs`.
3. Call `predict_noise_maybe_with_cfg` for CFG combination.
4. **Return `noise_pred.float()`** — `step_scheduler` expects fp32.

### 3.4 `post_decode`

Packages the trajectory collected during `step_scheduler`:

1. Call `super().post_decode(state)` for VAE decoding.
2. Stack `all_latents`, `all_log_probs`, `all_timesteps` from the state lists.
3. Populate `output.custom_output` with the same keys that `forward()` produces:
   `all_latents`, `all_log_probs`, `all_timesteps`, `prompt_embeds`,
   `prompt_embeds_mask`, `negative_prompt_embeds`, `negative_prompt_embeds_mask`.
4. **Move tensors to CPU** so the receiving HTTP server process does not
   initialise CUDA context on GPU 0.

> **Why `custom_output` matters.** In stepwise mode the engine ships the
> `DiffusionOutput` across an inter-process MessageQueue. Downstream consumers
> (`vllm_omni_async_server.generate` → `embeds_padding_2_no_padding`) read
> these field names verbatim. If any field is `None`, it becomes a non-tensor
> `LinkedList` in the training `TensorDict`, breaking `mask.shape[0]`.

### 3.5 MixGRPO: Window Positioning

MixGRPO requires all rollouts in a batch to share one SDE window for correct
advantage estimation. In full-forward mode `_maybe_make_progressive_window()`
runs inside `forward()`. In stepwise mode `forward()` is never called, so the
window is never set.

The MixGRPO stepwise adapter must call `_maybe_make_progressive_window()` in
`prepare_encode` **before** delegating via `super()`. The canonical pattern
uses multiple inheritance:

```python
@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="mix_grpo_stepwise")
class QwenImageMixGRPOPipelineWithLogProbStepwise(
    QwenImageMixGRPOPipelineWithLogProb,        # window logic
    QwenImagePipelineWithLogProbStepwise,        # stepwise overrides
):
    def prepare_encode(self, state, **kwargs):
        # Fix the SDE window before stepwise prepare_encode draws it
        if state.sampling is not None:
            if state.sampling.extra_args is None:
                state.sampling.extra_args = {}
            self._maybe_make_progressive_window(state.sampling.extra_args, kwargs)
        return super().prepare_encode(state, **kwargs)
```

See
[`QwenImageMixGRPOPipelineWithLogProbStepwise`](../../verl_omni/experimental/qwen_image_mix_grpo_stepwise/vllm_omni_rollout_adapter.py).

---

## Step 4 — Enable Stepwise Mode

No code changes in the launcher. At runtime:

```bash
python3 -m verl_omni.trainer.main_diffusion \
    actor_rollout_ref.rollout.step_execution=true \
    ...
```

The wiring:

1. `DiffusionRolloutConfig.step_execution` (default `False`) is read by
   `vLLMOmniHttpServer.run_server`, which sets `engine_args["step_execution"] = True`.
2. `DiffusionRolloutConfig.resolve_algorithm()` checks whether a
   `<algorithm>_stepwise` class is registered for the architecture. If so, it
   updates `model_config.algorithm` in-place to the stepwise variant.
3. `VllmOmniPipelineBase.get_pipeline_path(architecture, algorithm)` resolves
   to your stepwise adapter's dotted path, which is passed to the vllm-omni
   engine as `custom_pipeline_args.pipeline_class`.

If no `_stepwise` class is registered, `resolve_algorithm` is a no-op and the
engine falls back to the standard full-forward pipeline (stepwise mode is not
available for that model/algorithm pair).

---

## Step 5 — Verify Parity

Before claiming stepwise support is complete, verify that `step_execution=True`
produces trajectories **identical** to `step_execution=False`:

1. **fp32 latent storage.** Confirm `all_latents` are fp32, not bf16.
   `ratio_mean ≈ 1.0` at step 1.
2. **Log-prob parity.** `all_log_probs` should match between the two modes
   (within numerical tolerance).
3. **Prompt embeddings.** `prompt_embeds` / `prompt_embeds_mask` shapes and
   values must be identical.
4. **MixGRPO window.** All rollouts in a batch must share the same SDE window.
5. **Tokenizer fallback.** The warm-up (dummy) path must produce the same
   tokenization as the diffusers pipeline.

Add a smoke test under `tests/special_e2e/` that runs with
`step_execution=true` and asserts exit code 0.

---

## When to Refactor Instead of Duplicating

The stepwise adapter pattern currently requires ~400+ lines of code that
mostly duplicate the standard adapter's `diffuse()` / `forward()` logic.
This is a known maintenance burden. The duplication will shrink once
vllm-omni provides native continuous-batching hooks (e.g. `--skip-tokenizer-init`,
`prompt_token_ids` support). Until then:

- Keep the stepwise adapter as thin as possible — delegate to shared helpers
  in the parent class whenever feasible.
- If you find yourself copying more than a few methods, factor the shared
  logic into a `common.py` in the standard pipeline package and import it
  from both adapters.
- File a feature request with vllm-omni for any missing CB hooks you need,
  and link it in a TODO comment.

---

## Relationship to Other Guides

- [`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md) —
  Prerequisite: the standard full-forward integration.
- [`integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md`](integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md) —
  If your algorithm needs a custom stepwise adapter for a policy-gradient
  method other than FlowGRPO/MixGRPO.
- [`common_pitfalls.md`](common_pitfalls.md) — Known issues specific to
  stepwise mode (fp32 latency storage, RoPE mismatch, MixGRPO window bypass,
  tokenizer fallback, device placement).
