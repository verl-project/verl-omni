(integrating_a_non_diffusers_model)=
# How to Integrate a Non-Diffusers Model for FlowGRPO Training

Last updated: 06/15/2026.

This guide walks you through integrating a **non-diffusers model** — a
standalone `nn.Module` that does **not** inherit from `diffusers.ModelMixin`
and is **not** loaded through `diffusers.AutoModel.from_pretrained` — into
VeRL-Omni so it can be trained end-to-end with the **FlowGRPO** algorithm.

Non-diffusers models manage their own architecture, configuration format,
and weight-loading logic — none of which go through diffusers APIs. BAGEL-7B-MoT
is the reference implementation.

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

Use `NonDiffusersModelBase` when **diffusers cannot load the model**.
Everything else (custom configs, weight loading, FSDP sharding)
can be handled through the standard ``DiffusionModelBase`` path by
overriding ``build_module()``.

---

## Mental Model

The training side for non-diffusers models follows the same two-context
architecture as diffusers models, but the **module** is built through a
different path:

```text
  ┌─────────────────────────────────┐                ┌──────────────────────────────┐
  │ Rollout worker (vllm-omni)      │   trajectory   │ Trainer worker (FSDP)        │
  │                                 │ ─────────────▶ │                              │
  │ <Name>Pipeline (upstream)       │  latents,      │ <Name>ForTraining            │
  │  └─ forward() + SDE loop        │  log_probs,    │  (NonDiffusersModelBase)     │
  │                                 │  prompt ids    │  └─ forward(...)             │
  │ <Name>PipelineWithLogProb       │                │                              │
  │  └─ wraps with SDE scheduler    │                │                              │
  │  └─ handles prompt format       │                │                              │
  └─────────────────────────────────┘                └──────────────────────────────┘
```

The key difference from the diffusers path:

| Aspect | Diffusers model | Non-diffusers model |
|---|---|---|
| Module loading | `diffusers.AutoModel.from_pretrained()` | `MyModel.from_pretrained(model_path)` |
| Module base class | `ModelMixin` (from diffusers) | `NonDiffusersModelBase` (from verl-omni) |
| `build_module()` | Return `None` → default AutoModel path | Return `MyModel.from_pretrained(...)` |
| Config | `model_index.json` → `_class_name` | `config.json` → custom struct (e.g. `BagelTrainingConfig`) |
| Architecture registration | Auto-detected from `model_index.json` | Explicit: `+actor_rollout_ref.model.architecture=...` |

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

1. **How does the upstream pipeline process text?** Does it expect raw text
   strings or token IDs? If it expects strings, you may need a decode
   workaround in the rollout adapter (see the BAGEL integration's
   `_ensure_bagel_prompt_text` for an example).
2. **What is the model's forward signature?** Non-diffusers models define
   their own forward convention — it will not match the diffusers
   `(sample, timestep, encoder_hidden_states)` pattern. Document the
   signature; ``prepare_model_inputs()`` must produce matching kwargs.
3. **How does the model handle CFG?** If it supports classifier-free
   guidance, identify the negative-branch parameters (e.g. `None` for
   text-conditioning) so training can replicate the CFG logic.
4. **What is the checkpoint format?** Note the weight file name, key prefix
   conventions, and any architectural details that must be remapped.
5. **What are the layer class names?** List them — FSDP needs these for
   `_no_split_modules` so it can wrap the model correctly.
6. **What are the special token IDs?** If the model uses boundary tokens
   (start-of-image, end-of-image, etc.), identify whether they come from
   config or the tokenizer. The data preprocessor must produce consistent
   token sequences.

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
  (see the BAGEL integration's ``bagel_model.py`` for an example).
- Handle dtype conversion yourself.

### 2.3 Implement `forward`

The forward signature is **model-dependent**. For example, an image
generation model might take ``(hidden_states, timestep, text_token_ids,
latent_pos_ids, **kwargs)``. The only constraint is that
``prepare_model_inputs()`` in the training adapter must build a dict
whose keys match this signature exactly.

For gradient checkpointing, wrap layer calls:

```python
def forward(self, *args, **kwargs):
    # ...
    for layer in self.layers:
        sequence = self._checkpointed_call(layer, sequence, ...)
    # ...
    return (output,)
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

This classmethod receives the training module, model config, latents,
timesteps, prompt embeddings (plus masks), and the micro-batch from the
trainer engine, returning a pair of model-kwargs dicts (positive and
negative).  See {doc}`integrating_a_diffusion_model` for the full
signature.  Note: non-diffusers models often ignore the ``prompt_embeds*``
parameters and read token IDs directly from ``micro_batch`` instead
(see the BAGEL integration's ``_prompt_token_ids_to_batch`` for an example).

### 3.3 Implement `forward_and_sample_previous_step`

This follows the same pattern as the diffusers guide. If your model
supports classifier-free guidance, implement multi-branch forwarding
(e.g. conditional + unconditional) and combine the outputs before calling
``scheduler.sample_previous_step()``. The BAGEL integration demonstrates
a 3-branch CFG with sigma-interval gating as a reference.

The return signature is always ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.

### 3.4 Implement the Scheduler

Non-diffusers models use ``FlowMatchSDEDiscreteScheduler`` just like
diffusers models. If your model needs custom sigma schedules (e.g.
time-shifted sigmas), place the setup logic in a shared ``common.py``
so both the training and rollout adapters use the identical schedule.
See the BAGEL integration's ``setup_bagel_sigmas`` for a worked example.

---

## Step 4 — Write the Rollout Adapter

Create `verl_omni/pipelines/<model>_flow_grpo/vllm_omni_rollout_adapter.py`.
This is nearly identical to the diffusers guide — subclass the upstream
vllm-omni pipeline and wrap with the SDE scheduler.

Key considerations for non-diffusers models:

**Prompt format.** The verl-omni agent loop ships token IDs in
``req.prompts[0]["prompt_token_ids"]``, but the upstream vllm-omni
pipeline may expect text strings. If needed, add a decode workaround
in your adapter's ``forward()`` (see the BAGEL integration's
``_ensure_bagel_prompt_text`` for an example).

**Scheduler adapter.** Some upstream pipelines use non-standard
``step()`` argument conventions. If the standard
``FlowMatchSDEDiscreteScheduler`` is incompatible, wrap it in a
lightweight adapter that reshapes inputs and outputs to match the
pipeline's expectation. Most models will not need this —
``FlowMatchSDEDiscreteScheduler`` works directly with the standard
interface.

**SDE window.** ``forward()`` must set up an SDE window (selecting a
subset of denoising steps), compensate for any vllm-omni version-specific
step-count quirks, and slice the trajectory to return only the
windowed steps.

---

## Step 5 — Wire Up Registries and Package

### 5.1 `__init__.py`

Export the adapter classes so their `@register(...)` decorators run on import.
The model module is typically imported by the training adapter directly and
does not need to be in the public API, but including it is fine:

```python
from .diffusers_training_adapter import MyModelDiffusion
from .vllm_omni_rollout_adapter import MyModelPipelineWithLogProb

__all__ = ["MyModelDiffusion", "MyModelPipelineWithLogProb"]
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

The data preprocessor must match the tokenisation used by the upstream
pipeline.  For most models the agent loop's ``prompts`` tensor is enough.

If the upstream pipeline processes prompts differently from the default
chat template (e.g. it has its own tokenization path), the data
preprocessor must produce token sequences consistent with what the
pipeline expects.  The BAGEL integration demonstrates this pattern: the
preprocessor uses a model-specific ``tokenize_<model>_prompt()`` wrapper
to match the pipeline's ``prepare_prompts`` output, storing the result as
a pre-tokenized column that the training adapter reads via
``_prompt_token_ids_to_batch()``.
(See ``examples/flowgrpo_trainer/data_process/`` for reference implementations.)

---

## Step 7 — Add a Smoke Test

Follow the same pattern as the diffusers guide (Step 6 of
{doc}`integrating_a_diffusion_model`), but with these additions:

1. The dummy data must include the ``prompt`` chat messages (standard batch
   ``prompts`` / ``attention_mask`` from the agent loop).
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
- [ ] `prepare_model_inputs()` reads ``prompts`` and ``attention_mask``
      from micro-batch (standard tensors, no extra field needed)
- [ ] `forward_and_sample_previous_step()` with 3-branch CFG combining

### Rollout adapter (`vllm_omni_rollout_adapter.py`)
- [ ] `@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")`
- [ ] Subclasses `BagelPipeline` from vllm-omni
- [ ] Wraps scheduler in `_BagelSchedulerAdapter` for 4-arg `step()` convention
- [ ] SDE `step()` passes batched `(1, tokens, C)` tensors so log-probs match training
- [ ] `_ensure_bagel_prompt_text()` workaround for text-prompt requirement
- [ ] `forward()` sets up SDE window, vllm-omni 0.22 timestep compensation, returns sliced trajectory

### Shared utilities (`common.py`)
- [ ] `setup_bagel_sigmas()` — shared sigma schedule for rollout and training
- [ ] `bagel_time_shift()` — SD3-style timestep shift of `3.0`
- [ ] CFG defaults (`BAGEL_FLOWGRPO_CFG_DEFAULTS`) — consistent between adapters

### Data preprocessor (see ``examples/flowgrpo_trainer/data_process/``)
- [ ] Stores prompts in standard chat-message format (``prompt`` key)
- [ ] (BAGEL only) Pre-tokenizes captions via ``tokenize_bagel_prompt()`` and the training
      adapter reads them via ``_prompt_token_ids_to_batch()``

### Wiring
- [ ] `verl_omni/pipelines/bagel_flow_grpo/__init__.py` re-exports the two adapter classes
      (``BagelDiffusion`` and ``BagelPipelineWithLogProb``)
- [ ] `verl_omni/pipelines/__init__.py` imports `bagel_flow_grpo`
- [ ] Example launch script at `examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh`
- [ ] Deploy config at `examples/flowgrpo_trainer/bagel_deploy_config.yaml`

---

## When to Use the Diffusers Path Instead

If diffusers can load the model, use
{doc}`integrating_a_diffusion_model`.  Override ``build_module()`` there
for any custom loading you need.
