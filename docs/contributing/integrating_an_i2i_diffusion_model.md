# How to Integrate an Image-to-Image Diffusion Model

Last updated: 07/14/2026.

This guide explains the image-to-image (I2I) contracts required to add a new
image-edit diffusion model to VeRL-Omni. It builds on
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md), which
covers the shared FlowGRPO scheduler, registry, rollout, training, and testing
contracts. Read that guide first.

The worked example is the Qwen-Image-Edit-Plus implementation in
[`verl_omni/pipelines/qwen_image_edit_flow_grpo/`](../../verl_omni/pipelines/qwen_image_edit_flow_grpo/__init__.py).
For instructions on running that model, see
[Train Qwen-Image-Edit-2511 with FlowGRPO](../start/qwen_image_edit.md).

## TL;DR

An I2I integration uses the same three-file package and dual registries as a
T2I integration. The additional work is to:

1. Transport condition images from the dataset to the rollout request.
2. Encode condition images for the text/vision encoder when required.
3. VAE-encode condition images and inject their latents into the transformer.
4. Return condition tensors and metadata with the rollout trajectory.
5. Reconstruct the same condition input during training-time log-probability
   recomputation.

For concat-conditioned models, inherit `DiffusionI2IModelBase`, implement
`prepare_condition`, and extend `inject_condition` only for model-specific
metadata such as spatial shapes or position IDs.

## Mental Model

An I2I adapter serves two execution contexts:

| Context | Runtime | I2I responsibility |
|---|---|---|
| Rollout | vLLM-Omni | Parse condition images, encode text/image features, VAE-encode condition images, run stochastic denoising, and return condition fields with the trajectory. |
| Training | diffusers + FSDP | Reconstruct one denoising step, inject condition tensors into transformer inputs, and recompute the policy log-probability. |

The rollout and training sides must use the same:

- Architecture and algorithm registry keys.
- Condition-image preprocessing and latent layout.
- Transformer input construction and output slicing.
- Scheduler, timestep scaling, CFG rule, and output sign.
- Condition metadata such as spatial shapes and position IDs.

For concat-conditioned models such as Qwen-Image-Edit, the data flow is:

```text
parquet images
    -> agent loop decodes condition image
    -> vLLM-Omni request preprocessing
    -> rollout adapter VAE-encodes condition image
    -> transformer([target noise | condition latent])
    -> rollout custom_output["condition_image_latents"]
    -> training micro-batch
    -> prepare_condition()
    -> inject_condition()
    -> transformer([target noise | condition latent])
    -> crop prediction to target token length
```

## Prerequisites

Before adding VeRL-Omni adapters, the model should already have:

- A diffusers transformer and reference inference pipeline.
- A vLLM-Omni pipeline that can preprocess condition images, VAE-encode them,
  and decode generated latents.
- A known-good inference recipe, including prompt template, scheduler,
  timestep convention, CFG behavior, and VAE normalization.

If either upstream runtime cannot load and run the model, add support there
first.

## Step 1: Compare the Upstream T2I and I2I Pipelines

If the model family already has a T2I pipeline, identify only what the I2I
pipeline adds. Record these details before writing code:

1. How condition images enter the request.
2. Whether the text encoder consumes image features or text only.
3. How condition images are resized and VAE-encoded.
4. Whether condition latents are concatenated, used by cross-attention, or
   injected through another path.
5. Which position metadata changes after adding condition tokens.
6. Which part of the transformer output corresponds to target tokens.
7. Whether positive and negative CFG passes use the same condition image.
8. The exact timestep scaling and scheduler inputs.

Reuse the T2I adapter when these behaviors are unchanged. Keep model-specific
I2I logic in the new model package.

## Step 2: Scaffold and Register the Package

Create the same three-file package used by T2I integrations:

```text
verl_omni/pipelines/<model>_flow_grpo/
|-- __init__.py
|-- diffusers_training_adapter.py
`-- vllm_omni_rollout_adapter.py
```

Export both adapters from `__init__.py`:

```python
from .diffusers_training_adapter import MyEditModel
from .vllm_omni_rollout_adapter import MyEditPipelineWithLogProb

__all__ = ["MyEditModel", "MyEditPipelineWithLogProb"]
```

Import the package from
[`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py) so
the decorators run during startup.

Both adapters register under the pair
`(model_index.json::_class_name, algorithm)`. The architecture strings must
match exactly:

```python
@DiffusionModelBase.register("MyEditPipeline", algorithm="flow_grpo")
class MyEditModel(...):
    ...


@VllmOmniPipelineBase.register("MyEditPipeline", algorithm="flow_grpo")
class MyEditPipelineWithLogProb(...):
    ...
```

## Step 3: Implement the Training Adapter

### Reuse an existing T2I adapter

When the model family already has a T2I FlowGRPO adapter, inherit
[`DiffusionI2IModelBase`](../../verl_omni/pipelines/model_base.py) and that T2I
adapter:

```python
from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase
from verl_omni.pipelines.my_model_flow_grpo.diffusers_training_adapter import MyT2IModel


@DiffusionModelBase.register("MyEditPipeline", algorithm="flow_grpo")
class MyEditModel(DiffusionI2IModelBase, MyT2IModel):
    @classmethod
    def prepare_condition(cls, micro_batch, latents, step):
        ...

    @classmethod
    def inject_condition(cls, model_inputs, negative_model_inputs, condition):
        ...
```

This multiple-inheritance pattern reuses the T2I scheduler,
`prepare_model_inputs`, and `forward_and_sample_previous_step` methods while
adding I2I condition hooks.

### Integrate an I2I-only model family

If there is no reusable T2I adapter, inherit `DiffusionI2IModelBase` directly.
Implement the scheduler, input preparation, and sampling methods described in
the [T2I integration guide](integrating_a_diffusion_model.md), then implement
the two I2I hooks:

```python
@DiffusionModelBase.register("MyEditPipeline", algorithm="flow_grpo")
class MyEditModel(DiffusionI2IModelBase):
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

    @classmethod
    def prepare_condition(cls, micro_batch, latents, step):
        ...

    @classmethod
    def inject_condition(cls, model_inputs, negative_model_inputs, condition):
        ...
```

### Implement `prepare_condition`

This hook extracts condition fields from the training micro-batch. Most fields
originate in rollout `custom_output`, while actor-only metadata such as
`sp_size` is injected by the training engine. Return a non-empty dictionary.
The dispatcher treats `None` as a missing rollout/data condition and fails
closed.

```python
from verl.utils import tensordict_utils as tu


@classmethod
def prepare_condition(cls, micro_batch, latents, step):
    del latents, step
    image_latents = micro_batch.get("condition_image_latents", None)
    if image_latents is None:
        return None
    return {
        "image_latents": image_latents,
        "img_shapes": tu.get(micro_batch, "img_shapes"),
        "sp_size": tu.get(micro_batch, "sp_size"),
    }
```

Use `condition_image_latents` as the micro-batch and rollout key. Do not use
`image_latents`: that name is reserved by the MFU FLOPs counter for the
denoised target latent. The hook maps the transport key to the
`image_latents` field expected by `DiffusionI2IModelBase.inject_condition`.

Tensor outputs are stored in the training `TensorDict`. Python metadata such
as nested image-shape lists is transported as non-tensor data. Use TensorDict
utilities to unwrap it before passing it to the transformer.

Do not return `sp_size` from the rollout adapter. The FSDP engine assigns it to
the actor micro-batch before `prepare_condition` runs, because rollout workers
do not own the actor sequence-parallel configuration.

### Implement `inject_condition`

The base implementation supports the common concat-and-crop pattern:

1. Concatenate `condition["image_latents"]` to `hidden_states` on the token
   dimension.
2. Store the original target sequence length in `_target_seq_len`.
3. Apply the same concatenation to negative CFG inputs.
4. Crop the transformer prediction to the target prefix in
   `DiffusionI2IModelBase.forward`.

Call the base implementation, then add model-specific metadata:

```python
@classmethod
def inject_condition(cls, model_inputs, negative_model_inputs, condition):
    model_inputs, negative_model_inputs = super().inject_condition(
        model_inputs,
        negative_model_inputs,
        condition,
    )

    img_shapes = condition.get("img_shapes")
    if img_shapes is not None:
        model_inputs["img_shapes"] = img_shapes
        if negative_model_inputs is not None:
            negative_model_inputs["img_shapes"] = img_shapes

    return model_inputs, negative_model_inputs
```

Qwen-Image-Edit replaces T2I `img_shapes` with metadata containing both target
and condition shapes. A model with additional RoPE IDs should update those IDs
here as well. A model that uses cross-attention or another conditioning
mechanism should override the full method instead of calling the concat
implementation.

Validate sequence-parallel divisibility after determining both target and
condition token counts. Padding condition tokens is usually incorrect because
the padding participates in attention and positional encoding.

### Prepare processor files when required

Override `prepare_processor_files` only when the checkpoint's processor files
need a model-specific fix before the driver calls `hf_processor()`:

```python
@classmethod
def prepare_processor_files(cls, model_path: str) -> str | None:
    ...
```

Return an alternate processor directory when appropriate, or `None` to use the
default `<model_path>/processor` lookup. Keep the hook idempotent because the
driver may restart.

## Step 4: Implement the Rollout Adapter

Subclass the upstream vLLM-Omni I2I pipeline and register the same
architecture/algorithm pair:

```python
@VllmOmniPipelineBase.register("MyEditPipeline", algorithm="flow_grpo")
class MyEditPipelineWithLogProb(MyUpstreamEditPipeline):
    ...
```

The rollout adapter has five I2I-specific responsibilities.

### 4.1 Parse condition images

Use the shared
[`ImageGenerationRequest`](../../verl_omni/pipelines/utils.py) parser instead
of depending on one internal request layout:

```python
custom_prompt = req.prompts[0] if req.prompts else {}
generation_request = ImageGenerationRequest.from_request_payload(custom_prompt)
condition_images = generation_request.images
if not condition_images:
    raise ValueError("MyEditPipeline requires a condition image")
```

The parser normalizes images from top-level `images`/`image` fields,
multimodal request data, and vLLM-Omni's `additional_information` fallback.
Validate the supported number of images and aspect ratios before encoding.

### 4.2 Encode prompts with image features when required

For a VLM-based model such as Qwen-Image-Edit, the prompt contains image
placeholder tokens. Move processor outputs to the text encoder's device and
dtype before the forward call:

```python
image_inputs = self.processor.image_processor(
    images=condition_images,
    return_tensors="pt",
)
if attention_mask is None:
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
pixel_values = image_inputs["pixel_values"].to(
    device=self.device,
    dtype=self.text_encoder.dtype,
)
image_grid_thw = image_inputs["image_grid_thw"].to(self.device)

encoder_hidden_states = self.text_encoder(
    input_ids=prompt_ids.to(self.device),
    attention_mask=attention_mask.to(self.device),
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    output_hidden_states=True,
)
```

Apply the same condition images to positive and negative prompt encoding when
the upstream CFG implementation does so. If the model uses a text-only
encoder, do not add image placeholders merely to copy Qwen's format. Preserve
the upstream prompt contract and provide an explicit condition-image transport
path.

### 4.3 Prepare condition latents and metadata

Use the upstream pipeline's preprocessing and VAE helpers when possible. The
rollout adapter must retain enough information to reproduce transformer inputs
during training. For Qwen-Image-Edit this includes:

- Packed condition image latents.
- Target and condition spatial shapes used by 2D RoPE.
- Positive and negative prompt embeddings and masks.

Do not infer non-square spatial dimensions from packed sequence length alone.
Carry explicit width/height or shape metadata when the transformer needs it.

### 4.4 Concatenate during denoising and crop predictions

The rollout and training paths must construct the same transformer input:

```python
latent_model_input = torch.cat([latents, condition_image_latents], dim=1)
target_seq_len = latents.shape[1]

noise_pred = self.transformer(
    hidden_states=latent_model_input,
    timestep=timestep / 1000,
    img_shapes=img_shapes,
    ...,
)[0]
noise_pred = noise_pred[:, :target_seq_len]
```

Match the upstream pipeline's timestep scaling, dtype casts, CFG formula, and
prediction sign exactly. The scheduler should receive fp32 predictions and
latents when the matching T2I FlowGRPO adapter does so.

### 4.5 Return condition fields in `custom_output`

Return every rollout-owned condition tensor and metadata field needed by
`prepare_condition`, in addition to the standard FlowGRPO trajectory fields.
Actor-owned metadata such as `sp_size` does not belong in `custom_output`. Use
a null-safe CPU conversion because log-probabilities and negative prompt
embeddings can be absent during validation or when CFG is disabled:

```python
def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


return DiffusionOutput(
    output=_maybe_to_cpu(image),
    custom_output={
        "all_latents": _maybe_to_cpu(all_latents),
        "all_log_probs": _maybe_to_cpu(all_log_probs),
        "all_timesteps": _maybe_to_cpu(all_timesteps),
        "prompt_embeds": _maybe_to_cpu(prompt_embeds),
        "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
        "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
        "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
        "condition_image_latents": _maybe_to_cpu(condition_image_latents),
        "img_shapes": img_shapes,
    },
)
```

The diffusion server and agent loop move tensor fields into the training
`TensorDict` and preserve Python values as non-tensor batch data. Keep names
identical between rollout `custom_output` and training `prepare_condition`.

## Step 5: Add Data Preparation and a Launch Recipe

Ship a converter and a complete training command under
`examples/flowgrpo_trainer/`. The parquet should contain at least:

| Column | Type | Purpose |
|---|---|---|
| `prompt` | list of chat messages | Positive editing instruction in the upstream prompt format. |
| `negative_prompt` | list of chat messages | Negative CFG prompt when CFG is enabled. |
| `images` | list of `{"bytes": ...}` dictionaries | Condition images decoded by the multimodal dataset path. |
| `data_source` | string | Dataset/reward identifier passed to the configured scorer and used for metric grouping. |
| `reward_model` | dictionary | Carries reward style and ground truth/instruction. |
| `extra_info` | dictionary | Optional sample provenance and reward metadata. |

The prompt template, image placeholders, and processor calls must agree. A
placeholder/image-count mismatch usually fails in the VLM encoder; a prompt
template mismatch can silently change policy behavior.

Use the [Qwen-Image-Edit training guide](../start/qwen_image_edit.md) as a
complete example of raw data conversion, parquet schema, launcher overrides,
and model-specific constraints.

## Step 6: Test the Integration

Add CPU tests for the model-specific contracts:

- Both registry lookups resolve the new architecture.
- Missing condition images fail with a clear error.
- `prepare_condition` reads the exact `custom_output` key names.
- Condition injection changes the transformer sequence as expected.
- Transformer predictions are cropped back to the target sequence length.
- Positive and negative CFG paths receive identical condition tensors.
- Spatial metadata covers both target and condition tokens.
- Sequence-parallel alignment rejects incompatible token counts.
- Supported square and non-square aspect ratios behave as documented.

Add a direct special E2E script under `tests/special_e2e/` that runs at least
one training step with a tiny checkpoint and synthetic condition images. It
must exercise rollout and training, not only upstream inference.

Follow the repository's current GPU-smoke selection policy. If the new E2E is
added to `tests/gpu_smoke/run_gpu_smoke_tests.sh`, add both its
`run_selected_test` call and selection-map entry. Do not add an unused test ID.

## Final Checklist

- [ ] The model runs in both diffusers and vLLM-Omni before adapter work starts.
- [ ] The package contains `__init__.py`, a training adapter, and a rollout adapter.
- [ ] Both adapters use the exact `model_index.json::_class_name` and algorithm key.
- [ ] `verl_omni/pipelines/__init__.py` imports the package.
- [ ] The training adapter inherits `DiffusionI2IModelBase` for I2I dispatch.
- [ ] `prepare_condition` returns a non-empty condition dictionary.
- [ ] Rollout uses `condition_image_latents`, not the reserved `image_latents` key.
- [ ] Rollout and training use identical condition concatenation/injection.
- [ ] Transformer output is reduced to the target latent shape before scheduler use.
- [ ] Positive and negative CFG paths use the intended condition tensors.
- [ ] Timestep scaling, dtype handling, scheduler, and CFG match upstream inference.
- [ ] Spatial/position metadata includes all target and condition tokens.
- [ ] The data converter keeps prompt placeholders and condition-image count aligned.
- [ ] The example launcher documents model, data, GPU, reward, and resolution overrides.
- [ ] CPU contract tests and a one-step E2E test pass.
