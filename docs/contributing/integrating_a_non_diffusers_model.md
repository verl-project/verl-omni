(integrating_a_non_diffusers_model)=
# How to Integrate a Non-Diffusers Model for FlowGRPO Training

Last updated: 06/15/2026.

This guide walks you through integrating a **non-diffusers model** — a
standalone `nn.Module` that does **not** inherit from `diffusers.ModelMixin`
and is **not** loaded through `diffusers.AutoModel.from_pretrained` — into
VeRL-Omni so it can be trained end-to-end with the **FlowGRPO** algorithm.

Non-diffusers models manage their own architecture, configuration format,
weight-loading logic, and (optionally) internal text processing (token
embedding inside the forward pass rather than relying on pre-computed
prompt embeddings). BAGEL-7B-MoT is the reference implementation.

If your model is a standard diffusers model, use
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md)
instead. This guide extends the contracts defined there and focuses on
what is **different** for non-diffusers models.

We use the **BAGEL-7B-MoT** integration
([`verl_omni/pipelines/bagel_flow_grpo/`](../../verl_omni/pipelines/bagel_flow_grpo/__init__.py))
as the worked example throughout.

---

## TL;DR

A new non-diffusers model needs **four files in one new package** plus
**three registry hooks**:

```
verl_omni/pipelines/<model>_flow_grpo/
├── __init__.py                       # re-exports adapters + model class
├── <model>_model.py                  # nn.Module subclass of NonDiffusersModelBase
├── diffusers_training_adapter.py     # subclass of DiffusionModelBase
└── vllm_omni_rollout_adapter.py      # subclass of upstream vllm-omni Pipeline
```

The **training adapter** (`diffusers_training_adapter.py`) follows the
same `DiffusionModelBase` contract as standard diffusers models, but
overrides `build_module()` to use your custom `from_pretrained()` path
instead of `diffusers.AutoModel`. The **rollout adapter** is identical
in structure to the diffusers case.

The **model module** (`<model>_model.py`) is the new piece: a standalone
`nn.Module` that subclasses `NonDiffusersModelBase` and provides:

- `from_pretrained(model_path, torch_dtype)` — classmethod for weight loading
- `forward(**kwargs)` — the generation forward pass
- `_no_split_modules` — FSDP sharding hints
- Optional gradient checkpointing support

---

## When to Use NonDiffusersModelBase

Use `NonDiffusersModelBase` when **any** of the following is true:

1. **The upstream model is not a `diffusers.ModelMixin`** — it may be a
   Hugging Face `PreTrainedModel`, a standalone PyTorch module, or a vllm-omni
   ported model.
2. **The model handles text embedding internally** — it takes raw token IDs
   (not pre-computed embeddings) in its forward pass.
3. **The model has a non-standard config format** — no `model_index.json`,
   non-standard config keys, or multiple sub-configs (e.g. BAGEL has separate
   `config.json`, `llm_config.json`, `vit_config.json`).
4. **The model requires custom weight-loading logic** — e.g. remapping
   checkpoint keys from `language_model.model.*` to local parameter names,
   or loading from `ema.safetensors` instead of `diffusion_pytorch_model.safetensors`.
5. **The model needs custom FSDP sharding** — different layer class names
   (`BagelMoTLayer` vs `QwenImageTransformerBlock`) or different parameter
   grouping.

---

## Mental Model

The training side for non-diffusers models follows the same two-context
architecture as diffusers models, but the **module** is built through a
different path:

```text
  ┌─────────────────────────────────┐                ┌──────────────────────────────┐
  │ Rollout worker (vllm-omni)      │   trajectory   │ Trainer worker (FSDP)         │
  │                                 │ ─────────────▶ │                              │
  │ BagelPipeline (upstream)        │  latents,      │ BagelForTraining              │
  │  └─ forward() + SDE loop        │  log_probs,    │  (NonDiffusersModelBase)      │
  │                                 │  prompt ids    │  └─ forward(hidden_states,    │
  │ BagelPipelineWithLogProb        │                │     timestep,                 │
  │  └─ wraps with SDE scheduler    │                │     text_token_ids, ...)      │
  │  └─ fills prompt text from IDs  │                │                              │
  └─────────────────────────────────┘                └──────────────────────────────┘
```

The key difference from the diffusers path:

| Aspect | Diffusers model | Non-diffusers model |
|---|---|---|
| Module loading | `diffusers.AutoModel.from_pretrained()` | `MyModel.from_pretrained(model_path)` |
| Module base class | `ModelMixin` (from diffusers) | `NonDiffusersModelBase` (from verl-omni) |
| `build_module()` | Return `None` → default AutoModel path | Return `MyModel.from_pretrained(...)` |
| Text input to `forward()` | Pre-computed embeddings `(B, L, D)` | Token IDs `(B, L)` |
| `prepare_model_inputs()` | Builds dict with `prompt_embeds` key | Builds dict with `text_token_ids` key |
| Config | `model_index.json` → `_class_name` | `config.json` → custom struct (e.g. `BagelTrainingConfig`) |

---

## Prerequisites

Before you start, the new model must already be supported upstream by:

- **vllm-omni** — provides the rollout-side `<Name>Pipeline`. Your
  rollout adapter inherits from this class. The pipeline must be capable of
  running diffusion (text-to-image, or whatever modality you are training).
- **A downloadable checkpoint** — the model weights (`.safetensors`) and
  config files must be available locally or on Hugging Face Hub.

Unlike the diffusers path, **diffusers does not need to support the model**
— that is the whole point of the non-diffusers path. However, you must
port or reimplement the transformer architecture locally (see Step 2).

If the model is not yet supported in vllm-omni, upstream it there first.
Nothing below will work without vllm-omni rollout support.

---

## Step 1 — Understand the Upstream Pipeline and the Rollout→Training Contract

Read the vllm-omni pipeline's `forward()` method and answer the same
questions as in the diffusers guide, plus these non-diffusers-specific ones:

1. **How does the upstream pipeline process text?** Does it call
   `self.bagel.prepare_prompts(prompts=[text], tokenizer=self.tokenizer)`?
   Does it expect raw text strings or token IDs? BAGEL currently expects
   text strings (see the `_ensure_bagel_prompt_text` workaround in the
   rollout adapter).
2. **What is the model's forward signature?** BAGEL's training-side forward
   takes `(hidden_states, timestep, text_token_ids, latent_pos_ids, ...)`,
   which is different from diffusers convention.
3. **How does the model handle CFG?** BAGEL uses a 3-branch scheme: text
   unconditional (`text_token_ids=None`), image unconditional, and full
   conditional. Training must replicate this exactly.
4. **What is the checkpoint format?** BAGEL uses `ema.safetensors` with
   key prefixes like `language_model.model.layers.*`. The training module
   must remap these to local parameter names.
5. **What are the layer class names?** BAGEL uses `BagelMoTLayer`. FSDP
   needs this for `_no_split_modules`.
6. **What are the special token IDs?** BAGEL uses `<|im_start|>`,
   `<|im_end|>`, `<|vision_start|>`, `<|vision_end|>` for sequence
   construction. These must match the data preprocessing exactly.

---

## Step 2 — Port the Model Architecture

Create `verl_omni/pipelines/<model>_flow_grpo/<model>_model.py`. This is
the most involved step. You are porting the transformer architecture from
the upstream model into a standalone `nn.Module` and subclassing
`NonDiffusersModelBase`.

### 2.1 Subclass `NonDiffusersModelBase`

```python
from verl_omni.pipelines.non_diffusers_model_base import NonDiffusersModelBase

class MyModelForTraining(NonDiffusersModelBase):
    _no_split_modules = ["MyTransformerLayer"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: MyTrainingConfig):
        super().__init__()
        self.config = config
        # ... build layers, embeddings, etc.
```

`NonDiffusersModelBase` provides for free:

| Feature | How to use |
|---|---|
| **LoRA/PEFT injection** | Inherits `add_adapter()`, `load_lora_adapter()`, `set_adapter()`, `disable_adapters()`, `enable_adapters()` |
| **Gradient checkpointing** | Set `_supports_gradient_checkpointing = True` and wrap layer calls with `self._checkpointed_call(fn, *args)` |
| **FSDP sharding** | Set `_no_split_modules` to your layer class names |
| **Checkpoint persistence** | Inherits `save_pretrained()` (saves `model.safetensors` + `config.json`) |

### 2.2 Implement `from_pretrained`

```python
@classmethod
def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16) -> MyModelForTraining:
    config = MyTrainingConfig.from_model_path(model_path)
    ckpt_path = os.path.join(model_path, "ema.safetensors")  # or whatever the checkpoint file is
    from safetensors.torch import load_file
    state_dict = load_file(ckpt_path)

    model = cls(config)
    mapped = _map_checkpoint_to_training(state_dict, config)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    model = model.to(torch_dtype)
    return model
```

Key points:

- You control the entire loading logic. No `diffusers.AutoModel` involved.
- Remap checkpoint keys to match your local parameter names
  (see BAGEL's `_map_checkpoint_to_training()`).
- Handle dtype conversion yourself.

### 2.3 Implement `forward`

The forward signature is **model-dependent**. BAGEL takes
`(hidden_states, timestep, text_token_ids, latent_pos_ids, **kwargs)`.
Your model may take different arguments. The only constraint is that
`prepare_model_inputs()` in the training adapter must build a dict whose
keys match this signature exactly.

For gradient checkpointing, wrap layer calls:

```python
def forward(self, hidden_states, timestep, text_token_ids, latent_pos_ids, **kwargs):
    # ...
    for layer in self.layers:
        def _layer_fn(seq, *args, _layer=layer, **kw):
            return _layer(seq, *args, **kw)
        sequence = self._checkpointed_call(_layer_fn, sequence, ...)
    # ...
    return (velocity,)
```

Return a **tuple** `(velocity,)` — the FSDP engine expects a single-element
tuple.

### 2.4 Implement the Config

Create a `@dataclass` config class with a `save_pretrained()` method and a
`from_model_path()` classmethod:

```python
@dataclass
class MyTrainingConfig:
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    # ...

    def save_pretrained(self, save_directory: str):
        output_path = os.path.join(save_directory, "config.json")
        with open(output_path, "w") as f:
            json.dump(asdict(self), f, indent=4, sort_keys=True)

    @classmethod
    def from_model_path(cls, model_path: str) -> MyTrainingConfig:
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            raw = json.load(f)
        return cls(
            hidden_size=raw.get("hidden_size", 3584),
            # ...
        )
```

The config is saved alongside weights in `save_pretrained()`.

---

## Step 3 — Write the Training Adapter

Create `verl_omni/pipelines/<model>_flow_grpo/diffusers_training_adapter.py`.
This follows the same `DiffusionModelBase` contract as the diffusers guide
({doc}`integrating_a_diffusion_model`), with one key difference:
**override `build_module()`** to use your custom loading path.

### 3.1 Override `build_module`

```python
@DiffusionModelBase.register("OmniMyModelForConditionalGeneration", algorithm="flow_grpo")
class MyModelDiffusion(DiffusionModelBase):
    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        logger.info("Loading MyModelForTraining from %s", model_config.local_path)
        return MyModelForTraining.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)
```

When `build_module()` returns a non-`None` value, the FSDP engine uses it
directly. When it returns `None` (the default), the engine falls back to
`diffusers.AutoModel.from_pretrained`.

### 3.2 Implement `prepare_model_inputs`

Unlike diffusers models that receive `prompt_embeds` and
`prompt_embeds_mask` from the agent loop, non-diffusers models typically
read **raw token IDs** from the micro-batch `TensorDict`. BAGEL stores
pre-tokenized IDs in `micro_batch["bagel_prompt_ids"]`:

```python
@classmethod
def prepare_model_inputs(cls, module, model_config, latents, timesteps,
                         prompt_embeds, prompt_embeds_mask,
                         negative_prompt_embeds, negative_prompt_embeds_mask,
                         micro_batch, step):
    B = latents.shape[0]
    device = latents.device

    hidden_states = latents[:, step]
    timestep = timesteps[:, step]

    # Read pre-tokenized IDs from the data pipeline (NOT from prompt_embeds).
    text_token_ids, text_attention_mask = cls._token_ids_to_batch(
        micro_batch["my_prompt_ids"], device
    )

    model_inputs = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "text_token_ids": text_token_ids,
        "text_attention_mask": text_attention_mask,
        # ... any other model-specific keys
    }

    # Unconditional: set text_token_ids=None (model handles it internally)
    negative_model_inputs = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "text_token_ids": None,
        # ...
    }

    return model_inputs, negative_model_inputs
```

Key contract: the `prompt_embeds` / `negative_prompt_embeds` arguments
are **still received** from the agent loop (they are always padded
tensors of the standard format), but non-diffusers models may **ignore**
them in favor of token IDs from `micro_batch`. BAGEL ignores
`prompt_embeds` entirely and reads `bagel_prompt_ids` instead.

### 3.3 Implement `forward_and_sample_previous_step`

This follows the same pattern as the diffusers guide. BAGEL's
implementation is more involved because of the 3-branch CFG:

1. Forward gen branch: `module(**model_inputs)`
2. Forward text-unconditional branch (if CFG active): `module(**negative_model_inputs)`
3. Forward image-unconditional branch (if `cfg_img_scale > 1.0`): call with
   `cfg_img_v_t = forward_gen(cfg_img_inputs)`
4. Combine via `_combine_cfg()` and call `scheduler.sample_previous_step()`

The return signature is always ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.

### 3.4 Implement the Scheduler

Non-diffusers models use `FlowMatchSDEDiscreteScheduler` just like
diffusers models. BAGEL adds a small twist: the scheduler is configured
with shifted sigmas via `setup_bagel_sigmas()`. Place the sigma setup
logic in a shared `common.py` so both the training and rollout adapters
use the same schedule.

---

## Step 4 — Write the Rollout Adapter

Create `verl_omni/pipelines/<model>_flow_grpo/vllm_omni_rollout_adapter.py`.
This is nearly identical to the diffusers guide — subclass the upstream
vllm-omni pipeline and wrap with the SDE scheduler.

The key non-diffusers difference is in how **prompts** are handled. The
verl-omni agent loop ships token IDs in `req.prompts[0]["prompt_token_ids"]`,
but the upstream vllm-omni pipeline may expect text strings in
`req.prompts[0]["prompt"]`. BAGEL currently includes a workaround
(`_ensure_bagel_prompt_text`) that decodes token IDs to text before calling
`super().forward()`. This is a known limitation (see the `TODO` in the
code) and should be removed once vllm-omni's `BagelPipeline` accepts token
IDs natively.

The SDE window adapter (`_BagelSchedulerAdapter`) is also BAGEL-specific
because the upstream BAGEL scheduler convention uses 4-argument `step()`
calls. Most models will not need this adapter — the standard
`FlowMatchSDEDiscreteScheduler` works directly.

---

## Step 5 — Wire Up Registries and Package

### 5.1 `__init__.py`

Export all three classes so the `@register(...)` decorators run on import:

```python
from .<model>_model import MyModelForTraining
from .diffusers_training_adapter import MyModelDiffusion
from .vllm_omni_rollout_adapter import MyModelPipelineWithLogProb

__all__ = ["MyModelForTraining", "MyModelDiffusion", "MyModelPipelineWithLogProb"]
```

### 5.2 Register in `verl_omni/pipelines/__init__.py`

```python
from .my_model_flow_grpo import *  # noqa: F401, F403
__all__ += my_model_flow_grpo.__all__
```

### 5.3 Architecture String

The architecture string passed to both `@DiffusionModelBase.register()` and
`@VllmOmniPipelineBase.register()` must be consistent. For non-diffusers
models, there is no `model_index.json` to auto-detect from, so users must
pass the architecture explicitly on the CLI:

```bash
+actor_rollout_ref.model.architecture=OmniMyModelForConditionalGeneration
```

---

## Step 6 — Add a Data Preprocessor

The data preprocessor for non-diffusers models must match the tokenisation
used by the upstream pipeline **and** produce the token-ID format that the
training adapter expects.

BAGEL's preprocessor (`examples/flowgrpo_trainer/data_process/bagel_ocr.py`)
does the following:

1. Loads the BAGEL tokenizer from the model checkpoint.
2. Constructs the user prompt text (system prompt + OCR text).
3. Tokenizes in BAGEL native format:
   `<|im_start|>` + `tokenizer.encode(user_text)` + `<|im_end|>`.
4. Stores the token IDs as `bagel_prompt_ids` (an `np.ndarray` of `int64`).

The training adapter reads `micro_batch["bagel_prompt_ids"]` directly —
no runtime tokenisation. This design avoids the decode→re-encode round-trip
that would otherwise be needed if the training adapter received text strings.

---

## Step 7 — Add a Smoke Test

Follow the same pattern as the diffusers guide (Step 6 of
{doc}`integrating_a_diffusion_model`), but with these additions:

1. The dummy data must include the token-ID field (e.g. `bagel_prompt_ids`)
   that the training adapter expects.
2. The architecture override must be passed explicitly:
   `+actor_rollout_ref.model.architecture=OmniMyModelForConditionalGeneration`.

---

## Reference: BAGEL Implementation Checklist

The BAGEL integration is the canonical non-diffusers example. Use this
checklist to verify your implementation against it:

### Model module (`bagel_model.py`)
- [ ] `BagelTrainingConfig` dataclass with `save_pretrained()` and
  `from_model_path()`
- [ ] `BagelForTraining(NonDiffusersModelBase)` with:
  - [ ] `_no_split_modules = ["BagelMoTLayer"]`
  - [ ] `_supports_gradient_checkpointing = True`
  - [ ] `forward()` with gradient checkpointing via `_checkpointed_call()`
  - [ ] `from_pretrained()` loading from `ema.safetensors` with key remapping
  - [ ] Token embedding, timestep embedding, VAE projection, position embedding
  - [ ] MoT dual-pathway attention (text `*_proj` + gen `*_moe_gen`)
  - [ ] SOI/EOI boundary token handling

### Training adapter (`diffusers_training_adapter.py`)
- [ ] `@DiffusionModelBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")`
- [ ] `build_module()` returns `BagelForTraining.from_pretrained(...)`
- [ ] `build_scheduler()` and `set_timesteps()` with shifted sigmas
- [ ] `prepare_model_inputs()` reads `bagel_prompt_ids` from micro-batch
- [ ] `forward_and_sample_previous_step()` with 3-branch CFG combining

### Rollout adapter (`vllm_omni_rollout_adapter.py`)
- [ ] `@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")`
- [ ] Subclasses `BagelPipeline` from vllm-omni
- [ ] Wraps scheduler in `_BagelSchedulerAdapter` for 4-arg `step()` convention
- [ ] `_ensure_bagel_prompt_text()` workaround for text-prompt requirement
- [ ] `forward()` sets up SDE window and returns sliced trajectory

### Shared utilities (`common.py`)
- [ ] `setup_bagel_sigmas()` — shared sigma schedule for rollout and training
- [ ] `bagel_time_shift()` — SD3-style timestep shift of `3.0`
- [ ] CFG defaults (`BAGEL_FLOWGRPO_CFG_DEFAULTS`) — consistent between adapters

### Data preprocessor (`bagel_ocr.py`)
- [ ] Tokenizes prompts in BAGEL native format (`<|im_start|>` + encode + `<|im_end|>`)
- [ ] Stores token IDs as `bagel_prompt_ids` (`np.ndarray` of `int64`)
- [ ] Matches the upstream `BagelPipeline.prepare_prompts()` tokenisation exactly

### Wiring
- [ ] `verl_omni/pipelines/bagel_flow_grpo/__init__.py` re-exports all three classes
- [ ] `verl_omni/pipelines/__init__.py` imports `bagel_flow_grpo`
- [ ] Example launch script at `examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh`
- [ ] Deploy config at `examples/flowgrpo_trainer/bagel_deploy_config.yaml`

---

## Common Pitfalls

### 1. Tokenization mismatch between data prep and rollout pipeline

**Symptom**: Reward collapse, poor image quality, or CFG producing blank
images.

**Root cause**: The data preprocessor's tokenization (e.g. `bagel_ocr.py`
wrapping with `<|im_start|>` / `<|im_end|>`) differs from what the upstream
vllm-omni pipeline does internally (e.g. `prepare_prompts` may add or strip
markers).

**Fix**: Read the upstream pipeline's prompt preparation code carefully and
ensure your preprocessor produces token sequences that would result in
identical KV cache state.

### 2. Config misalignment between checkpoint and training module

**Symptom**: `max_latent_size` mismatches causing position embedding shape
errors (`mat1 and mat2 shapes cannot be multiplied`).

**Root cause**: The training module's config (e.g. `max_latent_size=32`)
may differ from the actual checkpoint's latent position embedding size
(e.g. 1024 positions = 32×32 grid).

**Fix**: In `from_pretrained()`, detect the actual position embedding size
from the checkpoint and adjust the config accordingly. See BAGEL's
`from_pretrained()` for the `max_latent_size` auto-detection from
`latent_pos_embed.pos_embed.shape[0]`.

### 3. Architecture registration not triggering

**Symptom**: `NotImplementedError: No diffusion model registered for
(architecture='OmniBagelForConditionalGeneration', algorithm='flow_grpo')`.

**Root cause**: The package is not imported, so the `@register(...)`
decorators never run.

**Fix**: Ensure the package is imported from
`verl_omni/pipelines/__init__.py` and that the `__init__.py` inside your
package re-exports the decorated classes.

### 4. Scheduler sigma mismatch between rollout and training

**Symptom**: Large KL divergence even at step 1 (before any weight update),
or `pg_clipfrac` consistently near 100%.

**Root cause**: The sigma schedule set up on the rollout side (via
`setup_bagel_sigmas()`) differs from the training side. Common causes:
different `num_inference_steps`, different shift values, or forgetting to
call `scheduler.set_timesteps()`.

**Fix**: Share the sigma setup logic in `common.py` and call it from both
adapters with the same parameters.

### 5. MoT routing masks misaligned

**Symptom**: NaN loss or zero-gradient for certain layers.

**Root cause**: Non-diffusers models with MoT architecture (like BAGEL) route
tokens through different pathways (text vs generation). The text/latent masks
must be correct for every sequence length combination.

**Fix**: Verify `text_mask` and `latent_mask` cover all positions exactly
once and sum to `(B, L_total)`. SOI/EOI tokens should be on the text pathway
(they are not latent patches). Use the BAGEL implementation as a reference.

---

## When to Use the Diffusers Path Instead

Even if your model has a standalone implementation, consider using the
standard diffusers path ({doc}`integrating_a_diffusion_model`) if:

- The model is also available as a `diffusers.ModelMixin` on Hugging Face.
- The model uses standard prompt embeddings (not internal text processing).
- The model's config is compatible with `model_index.json`.
- You want automatic sigmas, CFG, and other infrastructure that diffusers
  provides for free.

The non-diffusers path trades off some of the diffusers automation for full
control over weight loading, architecture, and text processing. Use it when
you need that control, or when diffusers simply does not support your model.
