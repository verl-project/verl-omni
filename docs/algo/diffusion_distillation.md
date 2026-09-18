# Diffusion Distribution Matching: DMD2 Runtime

Last updated: 09/18/2026.

## Scope

This page documents the reusable runtime for **distribution-only DMD2**. The
runtime trains a few-step flow-matching student against a frozen real-score
teacher and a trainable fake-score model. It supplies differentiable student
sampling, an explicit student-to-fake-score update cycle, named-LoRA ownership,
EMA, numerical-skip handling, complete checkpoints and semantic student export.

The parent runtime slice intentionally registers no concrete architecture. This
integration supplies the `(QwenImagePipeline, dmd2)` adapter, frozen condition
provider, recipe, inference tool and Qwen-specific validation while leaving the
trainer, worker and engine unchanged.

The current contract is intentionally bounded to shape-preserving,
single-latent flow-matching models sampled with deterministic Euler transitions.
It is not original DMD's paired teacher-trajectory/perceptual-regression recipe,
DMD2's optional adversarial objective, a multimodal scheduler abstraction, or an
autoregressive causal-video runtime.

### DMD2 is not the OPD trainer path

| Contract | DMD2 distribution matching | Existing diffusion OPD |
|---|---|---|
| Trainer selector | `algorithm.trainer_type=distribution_matching` | `algorithm.trainer_type=policy_gradient` |
| Data flow | Prompt batches; fresh samples generated inside the training engine | Online rollout trajectories replayed by frozen teachers |
| Configuration | Independent top-level `dmd` group | `distillation` group with `enabled=true` |
| Supervision | Teacher and learned fake-score x0 estimates at re-noised student samples | Teacher reverse-transition means for `distill_kl` |
| Optimization | Independent student and fake-score optimizers | Actor update with the configured OPD loss |
| Rewards / PPO tensors | Not needed | Existing policy-gradient/OPD lifecycle |

Do not enable `distillation.enabled`, actor `use_distill_loss`, or additional KL
objectives for DMD2. See [diffusion OPD](diffusion_opd.md) for that separate
teacher-scheduling contract.

### What `offline` means

`algorithm.sample_source=offline` selects the framework's engine-local execution
path. It does **not** mean that student images are pregenerated. Each update
attempt samples fresh student latents from prompt batches and random noise inside
the FSDP engine, where the selected student forward can retain its autograd
graph. No independent rollout server, reward worker or replay buffer is started.

## Algorithm

For flow corruption and a velocity model using the `noise - clean` convention,

```{math}
x_\sigma = (1-\sigma)\operatorname{sg}(x_g) + \sigma\epsilon,
\qquad
\hat{x}_0 = x_\sigma - \sigma v(x_\sigma,\sigma,c).
```

The same noise, sigma and corrupted input are used for real and fake scoring in
a student attempt. Teacher classifier-free guidance is applied in denoiser
velocity space, before x0 conversion:

```{math}
v_r = v_{\mathrm{neg}} + s(v_{\mathrm{pos}}-v_{\mathrm{neg}}).
```

`dmd.cfg_norm=layer_norm` rescales each last-dimension vector by
`norm(v_pos) / max(norm(v_r), 1e-12)`. `none` disables rescaling and `scalar`
uses one batch-wide ratio clamped above at one. Student and fake-score
predictions remain conditional-only.

For teacher and fake-score x0 estimates `x_r` and `x_f`, the student objective is

```{math}
Z = \max\left(\operatorname{mean}_{\mathrm{nonbatch}}
    |\operatorname{sg}(x_g)-x_r|,\eta\right),
\qquad
g = \operatorname{nan\_to\_num}\left(\frac{x_f-x_r}{Z}\right),
```

```{math}
L_{\mathrm{student}} = \frac{1}{2}\operatorname{mean}
\left(x_g-\operatorname{sg}(x_g-g)\right)^2.
```

Score predictions and `g` are detached, so only the student receives gradients.
The fake-score attempt creates a new no-grad student sample and minimizes

```{math}
L_{\mathrm{fake}} = \operatorname{mean}
\left(v_f(x_\sigma,\sigma,c)
      - \operatorname{sg}(\epsilon-x_g)\right)^2.
```

The corrupted fake-stage input and target are detached from the student. Pure
tensor equations and `DMDLoss` live in the preceding loss/config PR; this runtime
provides their model execution and optimizer ownership.

## How verl-omni Implements DMD2

```text
main_diffusion.TaskRunner
  -> DistributionMatchingRayTrainer(BaseRayDiffusionTrainer)
     -> DMDTrainingWorker(TrainingWorker)
        -> DMDDiffusersFSDPEngine(DiffusersFSDPEngine)
           -> registered architecture adapter
           -> DMDLoss through the existing diffusion_loss dispatcher
```

| Component | Responsibility |
|---|---|
| `DistributionMatchingRayTrainer` | Reuse offline worker/data/resource setup; run the explicit 1:K cycle; publish complete checkpoints and one student artifact |
| `DMDTrainingWorker` | Reuse distributed initialization and mini/microbatch handling; constrain each call to one optimizer attempt |
| `DMDDiffusersFSDPEngine` | Differentiable sampling, score calls, optimizer selection, independent RNG streams, rank-agreed skips, EMA and DMD checkpoint state |
| Registered model adapter | Conditioning, latent geometry, latent packing, transformer inputs, prediction-to-x0 conversion and sampling sigmas |
| `DMDLoss` and `trainer/diffusion/distillation/utils.py` | Registered objective dispatch and pure tensor equations |

The DMD2 trainer remains a thin algorithm-specific subclass because one cycle
contains one student attempt followed by `K` fake-score attempts with separate
optimizer clocks. Policy-gradient and direct-preference `fit()` implementations
require rollout/reward or preference-score tensors and cannot express this update
order without importing unrelated semantics. Dataloader, resources, worker
initialization, FSDP and mini/microbatch machinery are inherited rather than
reimplemented.

### Architecture adapter contract

The engine resolves the stateless `DiffusionModelBase` adapter registered for the
selected `(architecture, dmd2)` pair and requires these methods:

| Method | Contract |
|---|---|
| `build_conditioning_provider(model_config, dmd_config)` | Return a per-run provider whose `encode` yields positive and optional negative detached conditioning |
| `latent_geometry(module, model_config, batch)` | Validate a nonempty homogeneous batch and return the initial latent shape plus architecture metadata |
| `pack_latents(latents)` | Convert canonical generated latents to the transformer's shape |
| `prepare_dmd_inputs(...)` | Build transformer inputs for latents, sigma, conditioning and geometry |
| `forward(...)` | Return a prediction with the same shape as the packed latent |
| `prediction_to_x0(noisy, prediction, sigma)` | Convert architecture-native prediction into canonical fp32 x0 |
| `sampling_sigmas(model_config, dmd_config, device)` | Return a descending `num_inference_steps + 1` Euler sigma grid |

Training adapters remain stateless classmethod/staticmethod registry classes.
Condition encoders and caches are per-run objects returned by the provider hook,
not state on the adapter class.

The runtime currently assumes one scheduler with `num_train_timesteps`, one sigma
per sample broadcast over all latent dimensions, a shape-preserving prediction
and deterministic Euler transitions. An architecture with coupled schedules,
multiple modalities or autoregressive cache commits needs an explicit engine
contract extension; registration alone is insufficient.

### Logical roles on one physical base

| Logical role | Named adapter | Optimized? |
|---|---|---|
| `student` | `default` | Student optimizer |
| `fake_score` | `fake_score` | Independent fake-score optimizer |
| `teacher_score` | `reference`, which disables adapters | Frozen base only |
| `student_ema` | `student_ema` | No optimizer; student EMA only |

Fake-score and EMA adapters start as copies of the student. The engine validates
nonempty, disjoint student/fake-score parameter sets, and gradients on an inactive
optimized role are errors. Shared named adapters are the only delivered storage
layout; independent full modules and separately placed teachers are not enabled.

## Cycle, skip and batching semantics

Each completed cycle consumes one fresh student batch followed by `K` fresh
fake-score batches. Every worker call performs one minibatch, one epoch and at
most one optimizer update.

- `training/global_step` counts completed cycles after all `1 + K` attempts,
  including numerical skips.
- Student and fake-score successful-update counters advance independently.
- A scheduler advances only with its own successful optimizer update.
- All ranks agree on finite gradients before stepping. A skipped update clears
  gradients, advances neither scheduler nor EMA, and is not retried.
- A successful student update applies EMA once the successful student-update
  count reaches `ema_start_step`.
- An all-skipped cycle consumes the finite cycle budget. Training refuses to
  export if either role had no successful update by the end.
- Unexpected exceptions make the trainer terminal. Counters cannot undo an
  already applied optimizer step; restart from the last complete checkpoint.

`data.train_batch_size` is the global batch per attempt and must divide evenly
across data-parallel ranks. Student and fake-score physical microbatch sizes are
independent. Rank-local nondivisible tails use native TensorDict splitting and
sample-weighted means; no synthetic samples are padded. Dynamic batching is not
supported. The architecture adapter may impose additional geometry constraints.

## Configuration

The loss/config PR supplies the top-level `dmd` group. Runtime routing uses:

```yaml
algorithm:
  trainer_type: distribution_matching
  sample_source: offline
actor_rollout_ref:
  model:
    algorithm: dmd2
    model_type: diffusion_dmd_model
```

Student optimizer settings remain at `actor_rollout_ref.actor.optim`; fake-score
settings are under `dmd.fake_score_optim`. The implemented runtime requires LoRA,
FSDP/FSDP2, exactly one actor epoch, colocated execution and synchronous local
checkpoints. FSDP1 additionally requires `use_orig_params=true`.

| DMD field | Default | Runtime meaning |
|---|---|---|
| `fake_update_ratio` | `2` | Fake-score attempts per student attempt |
| `student_micro_batch_size_per_gpu` | `1` | Student physical microbatch per DP rank |
| `fake_score_micro_batch_size_per_gpu` | `1` | Fake-score physical microbatch per DP rank |
| `fake_score_optim` | LR `2e-5`, weight decay `0.001` | Independent existing FSDP optimizer config |
| `teacher_guidance_scale` | `4.0` | Positive teacher CFG scale |
| `cfg_norm` | `layer_norm` | `none`, `layer_norm`, or `scalar` denoiser-space rescaling |
| `negative_prompt` | `" "` | Explicit architecture-provider negative condition |
| `normalization_epsilon` | `1e-6` | Per-sample score-gradient normalization floor |
| `rollout_timestep_shift` | `3.0` | Architecture adapter's sampling-grid shift |
| `score_discrete_steps` | `1000` | Scheduler-sized score grid; zero selects continuous sampling |
| `score_sigma_min`, `score_sigma_max` | `0.02`, `0.98` | Score-sigma bounds |
| `score_timestep_shift` | `3.0` | Discrete score-sampling shift |
| `ema_decay` | `0.999` | Student EMA decay |
| `ema_start_step` | `0` | Successful student-update threshold for EMA |
| `export_role` | `student` | Export `student` or explicit `student_ema` |

## Qwen-Image integration

`QwenImageDMD2` is a stateless `DiffusionModelBase` adapter in
`pipelines/qwen_image_dmd2/diffusers_training_adapter.py`. It reuses the existing
Qwen-Image training adapter for transformer invocation and implements only the
DMD2 hooks required by the parent runtime:

| Hook | Qwen behavior |
|---|---|
| conditioning | Apply the checkpoint's fixed prompt template and prefix removal, or validate detached precomputed embeddings |
| latent geometry | Derive VAE channel/scale metadata from the checkpoint and require homogeneous image dimensions within a physical batch |
| packing | Use the native Qwen 2x spatial packing from normalized VAE latents to `[B,N,D]` |
| model inputs | Reuse Qwen timestep normalization, RoPE lengths and packed transformer kwargs |
| prediction conversion | Convert packed `noise - clean` velocity to fp32 x0 |
| sampling sigmas | Use a fixed once-shifted Euler grid shared with the inference tool |

The teacher applies positive/negative CFG in packed velocity space before x0
conversion. Student and fake-score forwards are conditional-only. For four steps
and `rollout_timestep_shift=3`, the sigma grid is `[1, 0.9, 0.75, 0.5, 0]`.
This intentionally differs from the stock Qwen pipeline's
resolution-dependent shift; training and the supplied generation tool use the
same fixed grid.

### Prompt and batching contract

Prompt parquet uses the existing `RLHFDataset` chat schema, for example:

```python
{"prompt": [{"role": "user", "content": "A red apple on a wooden table"}]}
```

Raw input accepts a string or one text-only user message. The provider rejects
custom system/assistant or multi-message chats rather than applying a different
template. A custom dataset may instead provide detached `[B,L,D]` embeddings and
matching masks. Negative conditioning is required only for student teacher-score
calls; fake-score attempts do not encode it.

Physical batches may contain more than one sample when all image dimensions
match. Different geometries fail before model execution. Microbatch tails use the
parent runtime's TensorDict splitting and sample-weighted reduction. Validated
distributed coverage is SP=1; this integration does not claim sequence-parallel
training support.

### Train, resume and generate

Use the {doc}`Qwen-Image DMD2 example <../examples/qwen_image/dmd2_trainer>`.
The launcher selects 1024x1024, four steps, max sequence length 1024, LoRA
rank/alpha 32, student LR `1e-4`, fake-score LR `2e-5`, teacher CFG 4 with
`layer_norm`, and FSDP2/BF16 by default:

```bash
MODEL_PATH=/path/to/Qwen-Image \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/qwen_dmd2 \
NUM_GPUS=8 TOTAL_TRAIN_STEPS=1000 \
bash examples/dmd2_trainer/qwen_image/run_qwen_image_dmd2_lora.sh
```

The validation parquet is required by shared dataloader construction, but the
launcher disables validation generation. To resume, pass the same model,
optimizer, DMD and data settings plus `trainer.resume_mode=resume_path` and
`trainer.resume_from_path=<checkpoint>`.

The default inference artifact is the student PEFT adapter. Generate with the
recorded fixed schedule and verified base provenance:

```bash
python examples/dmd2_trainer/qwen_image/generate.py \
  --artifact outputs/qwen_dmd2/inference \
  --prompt 'A red apple on a wooden table' \
  --seed 42 --output outputs/apple.png
```

This is a base-dependent LoRA, not a merged standalone pipeline. The inference
tool verifies the manifest, transformer-config hash and adapter checksum before
decoding.

## Checkpoint and Export

A resumable checkpoint and an inference adapter are distinct artifacts:

```text
OUTPUT_DIR/
  global_step_N/
    actor/
      model_world_size_<world>_rank_<rank>.pt
      optim_world_size_<world>_rank_<rank>.pt
      extra_state_world_size_<world>_rank_<rank>.pt
      dmd_state_rank_<rank>.pt
    data.pt
    trainer.pt
  latest_checkpointed_iteration.txt
  inference/
    adapter_model.safetensors
    adapter_config.json
    inference_manifest.json
```

Standard FSDP shards contain the physical model and student optimizer state.
`dmd_state_rank_*` adds fake-score optimizer/scheduler state, successful/skipped
counters and independent initial-noise, rollout-decision, score-sigma and
score-noise generators. `data.pt` stores dataloader state; `trainer.pt` stores
completed cycles, data epoch, driver RNG and a canonical configuration
fingerprint. EMA is model state, not a third optimizer.

Checkpoints are synchronously and atomically published to a shared local
filesystem only after all required rank files and counters validate. Incomplete
or legacy prototype formats are rejected before model mutation. Positive
`max_actor_ckpt_to_keep` prunes only complete older checkpoints in the run's
output root. Resume restores exact saved state, but native mixed-precision
execution is not promised to produce bitwise-identical future updates.

After the finite cycle budget, the trainer exports one finite complete
base-dependent PEFT adapter selected by `dmd.export_role`. Fake-score and teacher
weights are never inference artifacts. The manifest records optimizer counts,
base provenance, transformer-config hash and artifact checksum. A concrete model
integration owns decoded-generation instructions and validates that the exported
adapter reloads into its architecture.

## Limitations

The Qwen-Image integration is limited to base text-to-image generation with
LoRA and SP=1. It does not include automatic validation-replica synchronization,
vLLM-Omni serving, request batching, standalone score transport, full finetuning
or NPU validation.

## References

- [DMD2 paper](https://arxiv.org/abs/2405.14867) and [reference implementation](https://github.com/tianweiy/DMD2).
- [Diffusion OPD](diffusion_opd.md), a separate frozen-teacher transition-supervision path.
- [Direct-preference integration guide](../contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md), which documents reusable offline infrastructure but not DMD2's 1:K update semantics.
