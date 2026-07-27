# How to Add Continuous Batching (Step-Execution) Support for a Diffusion Model

Last updated: 07/22/2026.

This guide explains how to extend an existing diffusion rollout adapter so it
supports **continuous batching through vLLM-Omni step execution**.

For a user-facing comparison of step-wise vs request-level batching (when to
enable which), see [`rollout_batching.md`](../start/rollout_batching.md).

You must already have a working full-forward integration following
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md).

The Qwen-Image FlowGRPO adapter is the canonical example:

[`verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py)

Qwen-Image MixGRPO reuses the same implementation and adds its own SDE-window
initialisation:

[`verl_omni/pipelines/qwen_image_mix_grpo/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_mix_grpo/vllm_omni_rollout_adapter.py)

---

## TL;DR

Step execution is implemented **inside the existing rollout adapter**.

For example, the same class remains registered as:

```python
@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_grpo")
class QwenImagePipelineWithLogProb(...):
    ...
```

Add the vLLM-Omni lifecycle methods to that class:

```text
prepare_encode()
denoise_step()
step_scheduler()
post_decode()
```

The normal full-forward path remains in `forward()`.

At runtime, choose the execution mode through rollout configuration:

```bash
actor_rollout_ref.rollout.step_execution=True \
actor_rollout_ref.rollout.max_num_seqs=16
```

Do **not** create a separate package under `verl_omni/experimental`, register an
algorithm such as `flow_grpo_stepwise`, or rewrite the configured algorithm name.
Both execution modes use the original registration, such as `flow_grpo` or
`mix_grpo`.

---

## Execution Model

In full-forward mode, vLLM-Omni calls `forward()` once and the adapter executes
the complete denoising trajectory internally.

In step-execution mode, the engine keeps multiple requests in flight and
interleaves one denoising step from each request.

```text
Full-forward mode                       Step-execution mode
─────────────────                       ───────────────────
forward()                               prepare_encode()
  └─ complete diffusion loop              │
                                           ├─ denoise_step()
                                           ├─ step_scheduler()
                                           ├─ denoise_step()
                                           ├─ step_scheduler()
                                           └─ post_decode()
```

The lifecycle methods have the following responsibilities:

| Phase | Method | Responsibility |
|---|---|---|
| Setup | `prepare_encode()` | Encode prompts and initialise request-local state |
| Per-step | `denoise_step()` | Run the transformer and return a noise prediction |
| Per-step | `step_scheduler()` | Advance one scheduler step and collect trajectory data |
| Finalise | `post_decode()` | Decode and package the final `DiffusionOutput` |

The engine owns request scheduling. The adapter owns model-specific state,
scheduler semantics, precision, and the rollout output contract.

---

## Prerequisites

Before adding step execution, confirm that:

1. The model already has a working rollout adapter registered under the normal
   algorithm name, such as `flow_grpo`.
2. Its full-forward `forward()` or `diffuse()` path already returns the complete
   trajectory required by training.
3. The expected `custom_output` contract is defined. For Qwen-Image FlowGRPO and
   MixGRPO it is:

   ```text
   all_latents
   all_log_probs
   all_timesteps
   prompt_embeds
   prompt_embeds_mask
   negative_prompt_embeds
   negative_prompt_embeds_mask
   ```

4. The corresponding vLLM-Omni model pipeline supports step execution.
5. Step execution is appropriate for the algorithm. Qwen-Image DPO and
   DiffusionNFT use different `custom_output` contracts and must be integrated
   separately rather than forced to emit the FlowGRPO trajectory format.

---

## Step 1 — Keep One Pipeline Registration

Extend the existing rollout adapter directly:

```python
from verl_omni.pipelines.model_base import VllmOmniPipelineBase


@VllmOmniPipelineBase.register(
    "MyDiffusionPipeline",
    algorithm="flow_grpo",
)
class MyPipelineWithLogProb(...):
    def forward(self, req, **kwargs):
        # Existing full-forward implementation.
        ...
```

Do not add a second registration such as:

```python
# Do not use this pattern.
@VllmOmniPipelineBase.register(
    "MyDiffusionPipeline",
    algorithm="flow_grpo_stepwise",
)
```

The configured algorithm must remain stable across execution modes. The engine
selects full-forward or step execution from
`actor_rollout_ref.rollout.step_execution`.

This avoids parallel adapters that duplicate rollout logic and drift apart over
time.

---

## Step 2 — Implement `prepare_encode`

`prepare_encode()` creates all request-local state required by later lifecycle
methods. In step execution, `forward()` is never called, so no initialisation
performed only in `forward()` is available.

A typical signature is:

```python
from vllm_omni.diffusion.worker.utils import DiffusionRequestState


def prepare_encode(
    self,
    state: DiffusionRequestState,
    **kwargs,
) -> DiffusionRequestState:
    ...
```

### Prompt handling

The Qwen-Image RL adapter receives pre-tokenized prompts. It extracts:

```text
prompt_token_ids
prompt_mask
negative_prompt_ids
negative_prompt_mask
```

Current vLLM-Omni stores the request prompt on `state.prompt`. A robust adapter
should also support raw text because the engine's dummy warm-up request may use
a text prompt.

Normalise list inputs to tensors on the model device before encoding.

### Prompt embeddings

Encode positive and optional negative prompts into:

```text
prompt_embeds
prompt_embeds_mask
negative_prompt_embeds
negative_prompt_embeds_mask
```

Preserve the same prompt template, truncation, padding, and CFG semantics used by
the full-forward path.

### Latents and timesteps

Initial latents should be created in float32:

```python
latents = self.prepare_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    torch.float32,
    self.device,
    generator,
    None,
)
```

Prepare the same timestep schedule used by `forward()`.

### RoPE text lengths

Derive text RoPE lengths from the padded embedding width, not from the number of
valid tokens in the mask.

Use the vLLM-Omni helper when available:

```python
from vllm_omni.diffusion.models.qwen_image.rope_utils import (
    txt_seq_lens_from_embeds,
)

txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)
negative_txt_seq_lens = txt_seq_lens_from_embeds(
    negative_prompt_embeds
)
```

This matches the Qwen-Image diffusers pipeline, where text RoPE length follows
`encoder_hidden_states.shape[1]`.

### Request-local scheduler

The scheduler is mutable and must not be shared across concurrent requests:

```python
import copy

request_scheduler = copy.deepcopy(self.scheduler)
request_scheduler.set_begin_index(0)
```

### SDE and log-probability state

Resolve the same sampling values used by `forward()`:

```text
noise_level
sde_window_size
sde_window_range
sde_type
logprobs
```

Persist the request generator on `state.sampling.generator` so scheduler calls
across multiple engine iterations continue the same random stream.

### Required state

Populate every value consumed later, including:

```text
prompt_embeds
prompt_embeds_mask
negative_prompt_embeds
negative_prompt_embeds_mask
latents
timesteps
step_index
scheduler
do_true_cfg
guidance
img_shapes
txt_seq_lens
negative_txt_seq_lens
sde_window
noise_level
sde_type
logprobs
all_latents
all_log_probs
all_timesteps
```

Initialise the trajectory containers as empty lists and return `state`.

---

## Step 3 — Implement `denoise_step`

`denoise_step()` receives a batch assembled from multiple in-flight request
states.

Cast the live float32 latents to the transformer's compute dtype only for the
forward pass:

```python
x = input_batch.latents.to(
    self.transformer.img_in.weight.dtype
)
```

Build the model-specific positive and negative CFG inputs, run the transformer,
and return the noise prediction in float32:

```python
return noise_pred.float()
```

Keeping the returned prediction in float32 matches the scheduler path and avoids
precision loss in log-probability computation.

An override is required when the upstream model implementation does not match
the RL adapter's prompt representation, CFG logic, attention arguments, or
output slicing.

---

## Step 4 — Implement `step_scheduler`

`step_scheduler()` must mirror one iteration of the full-forward `diffuse()`
loop.

It should:

1. Read the current timestep from `state.timesteps[state.step_index]`.
2. Resolve whether the current step is inside the SDE window.
3. Save the initial latent when entering the active window.
4. Call the request-local scheduler using float32 inputs.
5. Append the resulting latent, log probability, and timestep to the trajectory.
6. Keep the live `state.latents` in float32.
7. Increment `state.step_index`.

Example precision pattern:

```python
new_latents, log_prob, _, _ = state.scheduler.step(
    noise_pred.to(torch.float32),
    timestep,
    state.latents.to(torch.float32),
    generator=state.sampling.generator,
    noise_level=current_noise_level,
    sde_type=state.sde_type,
    return_logprobs=state.logprobs,
    return_dict=False,
)

state.latents = new_latents.to(torch.float32)
```

Do not store stepped requests in bf16. Continuous batching may combine a newly
admitted request whose latents are fp32 with older requests that have already
advanced. Mixed live dtypes cause batch assembly failures and also break
rollout/training log-probability parity.

---

## Step 5 — Implement `post_decode`

`post_decode()` finalises the request after the last denoising step.

It should:

1. Call `super().post_decode(state, **kwargs)` to decode the final latent.
2. Stack the trajectory lists using the same dimensions as the full-forward
   path.
3. Preserve all required keys, including optional negative-prompt fields.
4. Move the complete output to CPU before inter-process transfer.

For immutable `DiffusionOutput` objects, use `dataclasses.replace`:

```python
from dataclasses import replace

return replace(
    output,
    custom_output={
        "all_latents": stacked_latents,
        "all_log_probs": stacked_log_probs,
        "all_timesteps": stacked_timesteps,
        "prompt_embeds": state.prompt_embeds,
        "prompt_embeds_mask": state.prompt_embeds_mask,
        "negative_prompt_embeds": state.negative_prompt_embeds,
        "negative_prompt_embeds_mask":
            state.negative_prompt_embeds_mask,
    },
    to_cpu=True,
)
```

`to_cpu=True` prevents the receiving HTTP-server process from retaining device
tensors or initialising an unintended accelerator context.

The step-execution output must match the full-forward output contract exactly.
Downstream training code should not need to know which execution mode produced
the trajectory.

---

## Step 6 — Preserve Algorithm-Specific Initialisation

Some algorithms initialise rollout state in `forward()`. That logic must also
run before the step-execution state is created.

### MixGRPO

MixGRPO adjusts the SDE window before generation.

The existing MixGRPO adapter should apply the same helper in both paths:

```python
def prepare_encode(self, state, **kwargs):
    if state.sampling is not None:
        if state.sampling.extra_args is None:
            state.sampling.extra_args = {}
        self._maybe_make_progressive_window(
            state.sampling.extra_args,
            kwargs,
        )
    return super().prepare_encode(state, **kwargs)


def forward(self, req, **kwargs):
    self._maybe_make_progressive_window(
        req.sampling_params.extra_args,
        kwargs,
    )
    return super().forward(req, **kwargs)
```

The supported strategies remain:

- `random`: when `sde_window_seed` is set, select the window using
  `sde_window_seed + global_steps`, so rollout ranks agree for the same training
  step.
- `progressive`: advance the window from `global_steps`,
  `sde_window_size`, and `iters_per_group`.

Enabling step execution must not change the selected SDE window.

---

## Step 7 — Enable Step Execution

Command-line override:

```bash
python3 -m verl_omni.trainer.main_diffusion \
    actor_rollout_ref.rollout.step_execution=True \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    ...
```

Equivalent YAML:

```yaml
actor_rollout_ref:
  rollout:
    step_execution: true
    max_num_seqs: 16
```

`max_num_seqs` limits the number of diffusion requests that can be scheduled
concurrently. Choose it according to accelerator memory, image resolution,
prompt length, model size, and CFG settings.

The algorithm remains unchanged:

```yaml
actor_rollout_ref:
  model:
    algorithm: flow_grpo
```

or:

```yaml
actor_rollout_ref:
  model:
    algorithm: mix_grpo
```

No `_stepwise` suffix is required.

---

## Dispatch and Compatibility

The rollout configuration passes `step_execution` directly to the vLLM-Omni
engine.

Pipeline lookup still uses the original pair:

```text
(model architecture, configured algorithm)
```

For example:

```text
(QwenImagePipeline, flow_grpo)
(QwenImagePipeline, mix_grpo)
```

The same adapter class therefore provides both paths:

| Configuration | Executed path |
|---|---|
| `step_execution=False` | Adapter `forward()` |
| `step_execution=True` | `prepare_encode` → repeated step methods → `post_decode` |

Only enable step execution for a model/algorithm adapter that implements and
tests the required lifecycle and output contract.

Current Qwen-Image scope:

| Algorithm | Step execution |
|---|---|
| FlowGRPO | Supported |
| MixGRPO | Supported |
| DPO | Not covered by this integration |
| DiffusionNFT | Not covered by this integration |

DPO and DiffusionNFT should receive separate integrations because their rollout
outputs differ from the FlowGRPO trajectory.

---

## Testing

Add a focused regression test that launches the real vLLM-Omni engine with:

```python
rollout_cfg.step_execution = True
rollout_cfg.max_num_seqs = 16
```

For Qwen-Image FlowGRPO, the test should assert that:

```text
all_latents
all_log_probs
all_timesteps
prompt_embeds
prompt_embeds_mask
```

are present, are tensors, and are non-empty.

When no negative prompt is provided, these keys should still be present and may
contain `None`:

```text
negative_prompt_embeds
negative_prompt_embeds_mask
```

Also validate the trajectory shape relationships:

```text
len(all_latents) = len(all_timesteps) + 1
len(all_log_probs) = len(all_timesteps)
prompt_embeds.shape[:-1] = prompt_embeds_mask.shape
```

The canonical regression test is:

[`tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py`](../../tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py)

Run parity checks between `step_execution=False` and `step_execution=True` for:

- output field names and shapes;
- fp32 stored latents;
- prompt embedding and mask values;
- SDE-window selection;
- first-step `ratio_mean`, KL, and clipping metrics;
- seeded reproducibility;
- concurrent requests with different prompt lengths.

---

## Integration Checklist

- [ ] Extend the existing registered rollout adapter.
- [ ] Keep the original algorithm name.
- [ ] Preserve the full-forward `forward()` path.
- [ ] Implement `prepare_encode()`.
- [ ] Implement or verify `denoise_step()`.
- [ ] Implement `step_scheduler()`.
- [ ] Implement `post_decode()`.
- [ ] Deep-copy mutable scheduler state per request.
- [ ] Use padded embedding width for text RoPE lengths.
- [ ] Keep live and stored trajectory latents in fp32.
- [ ] Preserve the full `custom_output` contract.
- [ ] Move returned tensors to CPU.
- [ ] Mirror algorithm-specific setup in both execution paths.
- [ ] Add a real-engine regression test.
- [ ] Test both `step_execution=False` and `step_execution=True`.
- [ ] Do not create an experimental `_stepwise` registration.

---

## Relationship to Other Guides

- [`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md) —
  prerequisite full-forward integration.
- [`integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md`](integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md) —
  algorithm-specific training and rollout contracts.
- [`common_pitfalls.md`](common_pitfalls.md) — precision, RoPE, SDE-window,
  and device-placement failures.
