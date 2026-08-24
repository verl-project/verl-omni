# How to Add Continuous Batching (Step-Execution) Support for a Diffusion Model

Last updated: 07/23/2026.

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

Qwen-Image DiffusionNFT and online DPO are examples of algorithms that reuse
the same engine lifecycle but preserve different training-output contracts:

- [`verl_omni/pipelines/qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py)
- [`verl_omni/pipelines/qwen_image_dpo/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_dpo/vllm_omni_rollout_adapter.py)

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
`mix_grpo`. The same rule applies to `diffusion_nft` and `dpo`.

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
   rollout outputs required by training.
3. The expected rollout output contract is defined. The current Qwen-Image
   integrations use these contracts:

   | Algorithm | Required algorithm-specific fields |
   |---|---|
   | FlowGRPO / MixGRPO | `all_latents`, `all_log_probs`, `all_timesteps` |
   | DiffusionNFT | `latents_clean`, `train_timesteps` |
   | Online DPO | `latents_clean` |

   Every integration also preserves `prompt_embeds`, `prompt_embeds_mask`,
   `negative_prompt_embeds`, and `negative_prompt_embeds_mask`.

4. The corresponding vLLM-Omni model pipeline supports step execution.
5. Step execution is appropriate for the algorithm. Algorithms with different
   rollout output contracts must implement separate adapter hooks rather than
   being forced to emit the FlowGRPO trajectory format.

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

Choose the live-latent dtype from the algorithm's scheduler and full-forward
semantics, and keep it homogeneous across all in-flight requests.

FlowGRPO and MixGRPO create their step-execution latents directly in float32.

Online DPO instead generates the initial noise in `prompt_embeds.dtype` and
then casts it to float32, matching the initial random values produced by its
non-step path:

```python
latents = self.prepare_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    prompt_embeds.dtype,
    self.device,
    generator,
    None,
).float()
```

Prepare the same timestep schedule used by `forward()`.

DiffusionNFT reuses the standard Qwen-Image denoising and scheduler methods, so
it prepares latents in the prompt-embedding/model dtype. The standard scheduler
returns the model-output dtype, keeping newly admitted and already-stepped
requests compatible. If an algorithm's scheduler changes dtype after the first
step, override `denoise_step()` and `step_scheduler()` and keep the live state in
one explicit dtype, as the DPO adapter does.

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

This subsection applies to trajectory-based algorithms such as FlowGRPO and
MixGRPO. DiffusionNFT and DPO do not initialise reverse-SDE trajectory
containers.

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

Populate every shared value consumed later, including:

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
```

For FlowGRPO and MixGRPO, also populate `sde_window`, `noise_level`, `sde_type`,
`logprobs`, and initialise `all_latents`, `all_log_probs`, and `all_timesteps`
as empty lists. DiffusionNFT and DPO only store the state required by their
standard scheduler and final-latent output contracts. Return `state`.

---

## Step 3 — Implement `denoise_step`

`denoise_step()` receives a batch assembled from multiple in-flight request
states. Reuse the upstream model implementation when its prompt representation,
CFG logic, attention arguments, output slicing, and precision already match the
algorithm. DiffusionNFT follows this path.

When live state is kept in float32, cast latents to the transformer's compute
dtype only for the forward pass:

```python
x = input_batch.latents.to(
    self.transformer.img_in.weight.dtype
)
```

Build the model-specific positive and negative CFG inputs, run the transformer,
and return the noise prediction in the dtype required by the scheduler. The
FlowGRPO and DPO adapters return float32:

```python
return noise_pred.float()
```

For FlowGRPO, float32 also avoids precision loss in log-probability computation.
For DPO, it preserves the existing full-forward scheduler behavior and keeps all
live requests in the same dtype.

An override is required when the upstream model implementation does not match
the RL adapter's prompt representation, CFG logic, attention arguments, or
output slicing.

---

## Step 4 — Implement `step_scheduler`

`step_scheduler()` must mirror one iteration of the full-forward `diffuse()`
loop. Reuse the upstream implementation when the algorithm only needs the final
latent and its scheduler semantics already match, as DiffusionNFT does.

For a trajectory-based algorithm such as FlowGRPO or MixGRPO, it should:

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

Do not mix live dtypes. Continuous batching may combine a newly admitted request
with older requests that have already advanced. If new requests start in fp32,
stepped requests must remain fp32; if the standard model path starts in model
dtype, the scheduler must keep returning that dtype. Mixed live dtypes cause
batch assembly failures and can break rollout/training parity.

---

## Step 5 — Implement `post_decode`

`post_decode()` finalises the request after the last denoising step.

It should:

1. Decode the final latent, either through `super().post_decode()` or the
   model's `_decode_latents()` helper.
2. Build the exact rollout output payload and metadata required by the algorithm's full-forward
   path.
3. Preserve optional negative-prompt keys even when their values are `None`.
4. Move the complete output to CPU before inter-process transfer.

Use `with_rollout_data` from [`verl_omni.pipelines.diffusion_rollout_output`](../../verl_omni/pipelines/diffusion_rollout_output.py)
to preserve the decoded output's existing fields while attaching trajectory and prompt embeddings:

```python
from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data

return with_rollout_data(
    output,
    trajectory_latents=stacked_latents,
    trajectory_log_probs=stacked_log_probs,
    trajectory_timesteps=stacked_timesteps,
    prompt_embeddings={
        "prompt_embeds": state.prompt_embeds,
        "prompt_embeds_mask": state.prompt_embeds_mask,
        "negative_prompt_embeds": state.negative_prompt_embeds,
        "negative_prompt_embeds_mask": state.negative_prompt_embeds_mask,
    },
    to_cpu=True,
)
```

`to_cpu=True` prevents the receiving HTTP-server process from retaining device
tensors or initialising an unintended accelerator context.

The step-execution output must match the full-forward output contract exactly.
Downstream training code should not need to know which execution mode produced
the trajectory.

### DiffusionNFT output

DiffusionNFT trains from the final clean latent and the inference timestep
schedule. It reuses the upstream Qwen-Image `denoise_step()` and
`step_scheduler()` implementations, then returns:

```python
latents_clean = state.latents.float()
return with_rollout_data(
    output,
    prompt_embeddings={
        "prompt_embeds": state.prompt_embeds,
        "prompt_embeds_mask": state.prompt_embeds_mask,
        "negative_prompt_embeds": state.negative_prompt_embeds,
        "negative_prompt_embeds_mask": state.negative_prompt_embeds_mask,
    },
    rl={
        "latents_clean": latents_clean,
        "train_timesteps": state.timesteps.unsqueeze(0).expand(
            latents_clean.shape[0], -1
        ),
    },
    to_cpu=True,
)
```

It must not emit synthetic `all_latents`, `all_log_probs`, or `all_timesteps`
fields because DiffusionNFT training does not consume the reverse-SDE
trajectory.

### Online DPO output

Online DPO trains from paired final clean latents. Its step path keeps live
latents and scheduler inputs in float32, but returns no reverse-SDE trajectory:

```python
return with_rollout_data(
    output,
    prompt_embeddings={
        "prompt_embeds": state.prompt_embeds,
        "prompt_embeds_mask": state.prompt_embeds_mask,
        "negative_prompt_embeds": state.negative_prompt_embeds,
        "negative_prompt_embeds_mask": state.negative_prompt_embeds_mask,
    },
    rl={"latents_clean": state.latents.float()},
    to_cpu=True,
)
```

Keep prompt extraction and warm-up fallback logic local to an
algorithm-specific adapter unless multiple existing adapters require exactly
the same behavior. Avoid adding private step-execution helpers to a broadly
shared mixin merely to reduce duplication: doing so changes the method
resolution order of unrelated adapters such as Qwen-Image Edit.

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

For DiffusionNFT or online DPO, use the existing algorithm registration in the
same way:

```yaml
actor_rollout_ref:
  model:
    algorithm: diffusion_nft  # or: dpo
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
(QwenImagePipeline, diffusion_nft)
(QwenImagePipeline, dpo)
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
| DiffusionNFT | Supported with final-latent/timestep output |
| Online DPO | Supported with final-latent output |

The four algorithms share the vLLM-Omni engine lifecycle, but their rollout
adapters preserve their own scheduler precision and rollout output contracts.

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

For DiffusionNFT, assert that these fields are present, are tensors, and are
non-empty:

```text
latents_clean
train_timesteps
prompt_embeds
prompt_embeds_mask
```

For online DPO, assert the same contract without `train_timesteps`:

```text
latents_clean
prompt_embeds
prompt_embeds_mask
```

For both algorithms, keep the negative-prompt keys in `extra_fields`; they may
be `None` when no negative prompt is supplied. Also verify that DPO live latents
remain float32 across concurrent step-execution requests.

The canonical regression test is:

[`tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py`](../../tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py)

Run parity checks between `step_execution=False` and `step_execution=True` for:

- resolved height, width, inference steps, sigmas, guidance scales, output count,
  and maximum sequence length;
- seed/generator handling and initial latent values;
- output field names and shapes;
- fp32 stored latents;
- prompt embedding and mask values;
- text RoPE lengths derived from padded embedding width;
- model-input, timestep, noise-prediction, scheduler-input, and live-state
  dtypes at the first and subsequent steps;
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
- [ ] Keep live request dtypes homogeneous across admission and scheduler steps.
- [ ] Keep FlowGRPO/MixGRPO and DPO live latents in fp32.
- [ ] Preserve the full rollout output contract.
- [ ] Move returned tensors to CPU.
- [ ] Mirror algorithm-specific setup in both execution paths.
- [ ] Keep algorithm-private helpers out of shared mixins unless all consumers
      require identical behavior.
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
