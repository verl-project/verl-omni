# Diffusion Distribution Matching: DMD2

Last updated: 09/11/2026.

## Background and scope

DMD2 trains a few-step generator using a frozen real-score teacher and a trainable
fake-score model that tracks the generator's distribution. This implementation
supports **Qwen-Image text-to-image, distribution-only DMD2** from prompts. It
includes differentiable student sampling, alternating student/fake-score updates,
LoRA EMA, complete training checkpoints, and a student inference artifact.

This is not original DMD's paired teacher-trajectory/perceptual-regression recipe,
and it does not include DMD2's optional adversarial objective. Original DMD, GAN,
CausVid, Self-Forcing, Qwen-Image Edit and other architectures are outside the
current supported path. A successful short run establishes execution, not
paper-level reproduction, convergence or generation-quality improvement.

### DMD2 is not the OPD trainer path

| Contract | DMD2 distribution matching | Existing diffusion OPD |
|---|---|---|
| Trainer selector | `algorithm.trainer_type=distribution_matching` | `algorithm.trainer_type=policy_gradient` |
| Data flow | Prompt batches; samples generated inside the training engine | Online rollout trajectories replayed by frozen teachers |
| Configuration | Independent top-level `dmd` group | `distillation` group with `enabled=true` |
| Supervision | Teacher and learned fake-score x0 estimates at re-noised student samples | Teacher reverse-transition means for `distill_kl` |
| Optimization | Separate student and fake-score optimizers | Actor update with the configured OPD loss |
| Rewards / PPO tensors | Not needed | Existing policy-gradient/OPD lifecycle |

Do not enable `distillation.enabled`, actor `use_distill_loss`, or additional KL
objectives to run DMD2. See [diffusion OPD](diffusion_opd.md) for that separate
configuration and teacher-scheduling contract.

### What `offline` means here

`algorithm.sample_source=offline` selects the framework's **engine-local execution
path**, not offline RL or training on a fixed dataset of pre-generated images.
No independent rollout server or reward workers are started.

The current student generates fresh samples from prompts and random noise during
each training attempt. Sampling runs inside the FSDP training engine so the
selected student forward can retain its autograd graph. Thus sample generation
is online during training, despite the configuration name `offline`; it does not
require precomputed student images or teacher trajectories. Keep
`sample_source=offline` for this implementation.

The supported launcher uses `verl_omni.trainer.main_diffusion`.
`main_diffusion_v1.py` integration is not implemented for this DMD2 path.

## Objectives and gradient boundaries

The Qwen adapter works with normalized image latents packed as `[B,N,D]`, where
`B` is the physical microbatch, `N` is the packed spatial sequence length and `D`
is the packed channel width. The adapter derives geometry from the checkpoint;
the engine does not guess an image layout. Flow corruption and x0 conversion are

```{math}
x_\sigma = (1-\sigma)\operatorname{sg}(x_g) + \sigma\epsilon,
\qquad
\hat{x}_0 = x_\sigma - \sigma v(x_\sigma,\sigma,c),
```

where `sg` denotes stop-gradient and Qwen predicts velocity in the
`noise - clean_latent` convention. The noise, sigma and corrupted inputs are
shared between real/fake scoring within a student attempt.

### Student objective

The teacher runs positive and negative conditioning. Standard CFG is applied in
**packed velocity space**, before x0 conversion:

```{math}
v_r = v_{\mathrm{neg}} + s(v_{\mathrm{pos}}-v_{\mathrm{neg}}).
```

`dmd.cfg_norm=layer_norm` then rescales each last-dimension vector by
`norm(v_pos) / max(norm(v_r), 1e-12)`; this is norm rescaling, not a learned
LayerNorm module. `none` disables it. `scalar` uses one batch-wide norm ratio,
clamped above at 1. Student and fake-score predictions remain conditional-only.

Let `x_r` and `x_f` be the teacher and fake-score x0 estimates. For each sample:

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

Score predictions and `g` are detached. Only the student receives gradients.
The normalizer spans all non-batch dimensions, and the current Qwen objective is
unmasked. Nonfinite gradient elements are counted **before** sanitization; monitor
this count as well as skipped optimizer updates. Corruption, x0 conversion,
normalization and losses use fp32, independently of the transformer compute dtype.

### Fake-score objective

A fake-score attempt generates a new student sample without gradients, re-noises
it, and trains the fake score with ordinary flow denoising MSE:

```{math}
L_{\mathrm{fake}} = \operatorname{mean}
\left(v_f(x_\sigma,\sigma,c)
      - \operatorname{sg}(\epsilon-x_g)\right)^2.
```

Both the corrupted model input and the target are detached from the generator.
Only the fake-score optimizer is stepped; teacher scoring is unnecessary here.
`DMDLoss`, registered as `dmd2` in `diffusion_algos.py`, dispatches these two losses
using `dmd_stage`; it does not require old log-probabilities or advantages.

### Student sampling and score timesteps

For `S` inference steps, start with the linear grid `[1, ..., 1/S, 0]` and apply
`f(sigma) = mu * sigma / (1 + (mu - 1) * sigma)` once. Four steps with
`dmd.rollout_timestep_shift=3` give `[1, 0.9, 0.75, 0.5, 0]`. This deliberately
differs from the stock Qwen pipeline's resolution-dependent shift.

Each microbatch samples a uniform exit index in `[0,S)`, broadcast from rank 0
so sharded ranks execute the same forwards and gradient exit. Earlier Euler
transitions run without gradients. Only the selected student prediction retains
a graph during a student attempt; fake-score attempts retain no student graph.
Inference instead executes the entire fixed grid, without per-step added noise.

Score sigma is sampled separately for each sample:

- With `score_discrete_steps=T`, require `T` to equal the scheduler's training
  grid size. Draw an integer in `[0,T)`, apply `score_timestep_shift` to `t/T`,
  then clamp to `[score_sigma_min, score_sigma_max]`.
- With `score_discrete_steps=0`, draw uniformly within those sigma bounds;
  no timestep shift is applied.

## Runtime and ownership

The implementation follows the existing loss/engine/worker/trainer structure:

```text
main_diffusion.TaskRunner
  -> DistributionMatchingRayTrainer(BaseRayDiffusionTrainer)
     -> DMDTrainingWorker(TrainingWorker)
        -> DMDDiffusersFSDPEngine(DiffusersFSDPEngine)
           -> QwenImageDMD2 + frozen conditioning provider
           -> DMDLoss through the existing diffusion_loss dispatcher
```

| Component | Responsibility |
|---|---|
| `DistributionMatchingRayTrainer` in `trainer/diffusion/ray_diffusion_trainer.py` | Reuse offline initialization, resource pools, dataloaders and profiling; run the explicit 1:K loop; publish complete checkpoints and export |
| `DMDTrainingWorker` in `workers/dmd_worker.py` | Reuse distributed setup, dispatch and mini/microbatch handling; constrain each actor call to one optimizer attempt |
| `DMDDiffusersFSDPEngine` in `workers/engine/fsdp/dmd_impl.py` | Differentiable sampling, score calls, optimizer selection, RNG streams, numerical skips, EMA and DMD-specific checkpoint state |
| `QwenImageDMD2` in `pipelines/qwen_image_distillation/diffusers_training_adapter.py` | Stateless `(QwenImagePipeline, dmd2)` registry adapter: conditioning construction, geometry, packing, model inputs, velocity-to-x0 conversion and sigma grid |
| `QwenImageConditionProvider` | Per-run frozen text encoding and cached negative conditioning; no text-encoder gradients |
| `DMDLoss` and `trainer/diffusion/distillation/utils.py` | Registered objective dispatch and pure tensor equations |

The adapter is a classmethod/staticmethod registry class, not an instantiated
trainer or owner of optimizer state. The engine checks for its required DMD
methods at construction. No separate generic role graph, transport interface or
`DistributionMatchingModelAdapter` mixin is required by this implementation.

### Logical roles on one physical base

| Logical role | Implementation | Optimized? |
|---|---|---|
| `student` | `default` LoRA adapter | Student optimizer |
| `fake_score` | `fake_score` LoRA adapter | Independent fake-score optimizer |
| `teacher_score` | `reference` adapter context, which disables adapters | No; frozen base prediction |
| `student_ema` | `student_ema` LoRA adapter | No optimizer; student EMA only |

Fake-score and EMA adapters start as copies of the student. The engine records
nonempty, disjoint student/fake-score parameter sets and rejects overlap;
gradients on an inactive optimized role are errors. `LoRAAdapterMixin` provides
adapter selection/restoration, copying and EMA. Teacher scoring is not an OPD
teacher worker or an extra reference-KL objective.

Shared adapters are a storage choice, not a mathematical requirement of DMD2,
but they are the **only implemented layout here**. There is no selectable
`shared_base_adapters` / `colocated_independent` role-layout configuration.
Independent full models and separately placed teachers are not enabled.

### Cycles, skips and EMA

For every cycle, consume one fresh student prompt batch followed by `K` fresh
fake-score prompt batches. Each actor call is constrained to one minibatch,
one epoch and at most one optimizer update. `K` therefore counts attempts,
not an assumed number of updates hidden inside an actor call.

- `training/global_step` advances after all `1 + K` attempts complete, including
  numerical skips. It counts completed cycles, not successful student updates.
- Successful optimizer counters advance independently. Schedulers advance only
  with their own successful optimizer step; the fake scheduler's configured
  horizon is `K` times the student's.
- Ranks agree on nonfinite loss/gradient skips before stepping. A skipped role
  clears gradients and advances neither its scheduler nor EMA; it is not retried.
- EMA starts as a student copy. A successful student update applies
  `EMA = decay * EMA + (1 - decay) * student` once its successful-update count
  reaches `ema_start_step`. Earlier updates and skips leave EMA unchanged.
- Even all-skipped cycles consume the finite training budget. If either role
  has zero successful updates at the end, training raises instead of exporting
  a supposedly trained student.
- Unexpected exceptions stop the trainer. A partially applied cycle cannot be
  undone by resetting counters: resume from the last complete checkpoint in a
  new trainer, rather than retrying in-process.

## Configuration and usage

Follow the [GPU and training installation](../start/install.md), then use the
{doc}`Qwen-Image DMD2 example <../examples/qwen_image/dmd2_trainer>`. Commands below
run from the repository root. The launcher supplies the full configuration;
these selectors identify the route:

```yaml
algorithm:
  trainer_type: distribution_matching
  sample_source: offline
actor_rollout_ref:
  model:
    algorithm: dmd2
    model_type: diffusion_dmd_model
```

The existing actor loss-mode interpolation resolves to `dmd2`. Student optimizer
settings remain at `actor_rollout_ref.actor.optim`; fake-score settings are at
`dmd.fake_score_optim`. Both use existing FSDP optimizer configuration and the
engine's supported constant/cosine schedulers.

### DMD configuration reference

All fields below belong to top-level `dmd` (`DiffusionDMDConfig`), not to OPD's
`distillation` group.

| Field | Default | Meaning |
|---|---|---|
| `fake_update_ratio` | `2` | Positive integer fake attempts per student attempt |
| `student_micro_batch_size_per_gpu` | `1` | Student physical microbatch per DP rank |
| `fake_score_micro_batch_size_per_gpu` | `1` | Independent fake-score physical microbatch |
| `fake_score_optim` | LR `2e-5`, weight decay `0.001` | Existing FSDP optimizer config for fake score |
| `teacher_guidance_scale` | `4.0` | Positive teacher CFG scale; student/fake stay conditional-only |
| `cfg_norm` | `layer_norm` | Packed-velocity rescaling: `none`, `layer_norm`, or `scalar` |
| `negative_prompt` | `" "` | Explicit teacher negative text; empty string is valid, null is not |
| `normalization_epsilon` | `1e-6` | Positive lower bound for the per-sample normalizer |
| `rollout_timestep_shift` | `3.0` | Fixed student/inference grid shift, at least 1 |
| `score_discrete_steps` | `1000` | Scheduler-sized discrete grid; 0 selects continuous uniform sampling |
| `score_sigma_min`, `score_sigma_max` | `0.02`, `0.98` | Bounds satisfying `0 < min < max <= 1` |
| `score_timestep_shift` | `3.0` | Discrete score-sampling shift, at least 1 |
| `ema_decay` | `0.999` | EMA decay in `[0,1]` |
| `ema_start_step` | `0` | Nonnegative successful-student-update threshold |
| `export_role` | `student` | One inference artifact; `student_ema` is an explicit alternative |

The launcher additionally sets 1024×1024, four inference steps, max sequence
length 1024, LoRA rank/alpha 32, student LR `1e-4`, and FSDP2/BF16 with native
training attention. These are **launcher overrides**, not all model-config defaults.
It uses no VAE during training; decoding is part of the separate generation tool.

### Prompt data and launch

Use prompt parquet with the existing `RLHFDataset` schema, for example:

```python
{"prompt": [{"role": "user", "content": "A red apple on a wooden table"}]}
```

Raw inputs accept a string or one text-only user message. The condition provider
applies the checkpoint's Qwen system template and prefix removal; custom system
messages and multi-message conversations are rejected. A custom dataset can
instead supply post-collation embeddings `[B,L,D]` with optional `[B,L]` masks;
pre-tokenized inputs require masks and the correct Qwen template prefix.
Positive and negative conditioning must match the batch. No image preference
pairs, teacher trajectories, rewards or precomputed student samples are needed.

```bash
MODEL_PATH=/path/to/Qwen-Image \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/qwen_dmd2 \
NUM_GPUS=8 TOTAL_TRAIN_STEPS=1000 \
bash examples/dmd2_trainer/qwen_image/run_qwen_image_dmd2_lora.sh
```

A validation parquet is still supplied for shared dataloader initialization;
there is no validation-generation replica. The launcher disables validation
before training and sets `test_freq=-1`.

### Batching and diagnostics

`data.train_batch_size` is the global batch **per attempt** and must divide evenly
across DP ranks. `GLOBAL_BATCH_SIZE` defaults to `NUM_GPUS` in the example.
`STUDENT_MICRO_BATCH_SIZE` and `FAKE_MICRO_BATCH_SIZE` independently default to 1.
At SP=1, eight GPUs with global batch 16 and microbatch 2 use physical batch 2
per rank. A rank-local batch of 3 with microbatch 2 instead accumulates `2 + 1`.
Dense tails use native TensorDict splitting and sample-weighted means; no extra
training examples are padded in. Dynamic batching is not supported.

Samples in a physical microbatch must share image geometry. Accumulation and
physical batching may sample different rollout depths; `scalar` CFG additionally
couples samples through its batch-wide norm. Neither is automatically a controlled
performance comparison at fixed effective batch. Validated distributed coverage
uses SP=1; do not infer SP>1 support from the generic configuration surface.

Each attempt retains its own metric prefix: `student/0/...`, `fake_score/0/...`,
`fake_score/1/...`, and so on. Useful suffixes are `dmd/loss`, `dmd/normalizer`,
`dmd/gradient_norm` for the student, `fake_score/loss` for fake denoising, and
`dmd/update_applied`, `dmd/skip_nonfinite`, `dmd/nonfinite`, `dmd/rollout_exit`,
`training/samples` where applicable. Component timers include
`perf/condition_encode_s`, `perf/student_rollout_s`, `perf/teacher_score_s`,
`perf/fake_score_s`, `perf/backward_s`. Top-level
`training/student_optimizer_steps` and `training/fake_score_optimizer_steps`
record cumulative successful updates independently of `training/global_step`.

Loss-like microbatch metrics are sample-weighted; durations and counts are summed.
DP aggregation reports means, not slowest-rank wall time or global count sums.
Peak allocated/reserved memory uses rank maxima, cumulative from worker
initialization. `perf/cycle_s` covers the attempts but excludes checkpointing;
`perf/checkpoint_s` appears only when a checkpoint is saved. Nested host timings
are not additive CUDA kernel costs. Reuse `global_profiler.steps` and
`actor_rollout_ref.actor.profiler` for traces; see [profiling](../perf/profiler.md).

## Checkpoint, resume and inference contracts

A training checkpoint and one inference artifact serve different purposes:

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

The standard FSDP shards save the shared model, including all adapters, the student
optimizer, scheduler and worker RNG. `dmd_state_rank_*` adds the fake optimizer
and scheduler, successful/skipped counters, and separate initial-noise,
rollout-decision, score-sigma and score-noise generator states. `data.pt` holds the
stateful dataloader; `trainer.pt` records completed cycles, data epoch, driver RNG
and a canonical configuration fingerprint. EMA is model state, not a third
optimizer or a separately required inference export.

Checkpoints are synchronous and atomically published on a shared local filesystem
after all-rank file and clock validation. A failed save does not replace the
latest-complete pointer. Positive `trainer.save_freq` saves on its interval and
at the final cycle; disabling saves also disables the final training checkpoint.
Positive `trainer.max_actor_ckpt_to_keep` prunes older checkpoint directories in
that output root, so use a dedicated directory for each run.

Resume defaults to `auto`. To choose a checkpoint, append these overrides to the
same training command, normally with a fresh `OUTPUT_DIR`:

```bash
MODEL_PATH=/path/to/Qwen-Image \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/qwen_dmd2_resumed \
NUM_GPUS=8 TOTAL_TRAIN_STEPS=1000 \
bash examples/dmd2_trainer/qwen_image/run_qwen_image_dmd2_lora.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=outputs/qwen_dmd2/global_step_500
```

Keep the same model, engine, optimizer, DMD and data settings. Version, world size,
role counters, required state and configuration are checked before worker load.
A changed configured training horizon also changes optimizer compatibility.
Legacy generic-distillation checkpoints are rejected, not silently reinterpreted;
there is no automatic format or counter migration. Load training checkpoints only
from trusted sources because optimizer/RNG restoration uses Python serialization.

Exact recovery of saved state does not guarantee bitwise-identical subsequent
updates. Real native/BF16 runs have shown differing gradients on repeated backward
with identical model, inputs and RNG. Do not equate successful resume or exact
checkpoint restoration with deterministic full-model training replay.

### Student inference artifact

After the finite training budget, the default export is **student**. The fake score
and teacher are never exported for generation. `dmd.export_role=student_ema`
selects EMA instead; choose this at run configuration time because DMD settings
participate in the resume fingerprint.

Export validates finite complete adapter parameters and their own PEFT config;
custom LoRA targets must be covered by `model.fsdp_layer_prefixes`. The artifact
is atomically published and will not overwrite an incompatible existing export.
Its manifest records role, cycle/success counts, base model identity, resolved
revision when available, transformer-config hash, weight checksum, resolution,
sequence limit and fixed Euler sampling settings. A config hash is not a base
weight checksum; unversioned local bases have a null revision and must be kept
immutable by the user.

This is a **base-dependent LoRA**, not a merged self-contained pipeline. Reload
and decode using the supplied tool, which checks the artifact and uses the
recorded conditional-only schedule:

```bash
python examples/dmd2_trainer/qwen_image/generate.py \
  --artifact outputs/qwen_dmd2/inference \
  --prompt 'A red apple on a wooden table' \
  --seed 42 --output outputs/apple.png
```

Use `--base-model` if the same base checkpoint has moved. Do not substitute stock
pipeline scheduler/CFG defaults. Automatic CheckpointEngine validation-replica
synchronization, vLLM-Omni serving/request batching, standalone score transports,
full finetuning and NPU validation are not delivered by this path. FSDP1 requires
`use_orig_params=true`; FSDP2 is the example default.

## Validation and further reading

The validation commands in the
{doc}`Qwen-Image DMD2 example <../examples/qwen_image/dmd2_trainer>`
cover CPU/configuration checks, real two-rank tiny-Qwen FSDP1/FSDP2 updates,
unequal microbatch tails, EMA, checkpoint replay, numerical skips and adapter
reload. A separate production smoke exercises Ray routing, checkpoint publication
and export. CPU fake-engine tests alone cannot validate shared-LoRA FSDP behavior;
real checkpoint training and decoded-generation evidence must state the model,
precision, rank count and actual completed steps. Neither a decoded image nor
finite losses establishes a quality gain.

- [DMD2 paper](https://arxiv.org/abs/2405.14867) and [reference implementation](https://github.com/tianweiy/DMD2).
- [Diffusion OPD](diffusion_opd.md): separate frozen-teacher transition supervision.
- [Direct-preference integration guide](../contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md): shared offline infrastructure, not the DMD2 objective or update semantics.
