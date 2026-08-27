# Diffusion On-Policy Distillation

Last updated: 08/26/2026.

## Background

On-policy distillation (OPD) trains a student on states sampled from the
student's own policy, with a teacher providing dense supervision at those
states. Compared with distilling on teacher-generated data, this removes the
train/inference state mismatch: the teacher advises on the trajectories the
student actually visits. Compared with reward-only RL, the supervision is dense
and continuous instead of a sparse outcome score.

For diffusion policies the rollout is a reverse-SDE trajectory. The student
samples images and stores every intermediate state `(all_latents[t],
all_timesteps[t])`. A frozen teacher then replays each stored state through its
own transformer and the shared noise schedule, producing the mean of its
next-step transition distribution, `teacher_prev_sample_mean`. Since both
transitions are Gaussians with the student's std `σ_t`, the per-step KL has a
closed form, implemented by the `distill_kl` loss:

```{math}
\mathrm{KL}\left(p_\theta \,\Vert\, p_\text{teacher}\right)
= \frac{\lVert \mu_\theta - \mu_\text{teacher} \rVert^2}{2 \sigma_t^2},
```

averaged over latent dimensions. This is the diffusion analog of matching
next-token log-probabilities in LLM OPD: the teacher scores the student's
states, and the student moves its transition means toward the teacher's.

The teacher is fused into the actor worker the same way the reference model
is: a third frozen, forward-only engine that replays the student's
trajectories with a different checkpoint. Teacher and reference can coexist,
so distillation can be combined with a KL penalty toward the initial policy.

## Teacher Runtime

`DiffusionTeacherManager` runs teacher scoring as one stage of the training
step, after rewards and before the actor update. Each teacher is a worker
group of frozen, forward-only engines; a batch passes through four moves:

1. **Route.** The batch column named by `teacher_key` maps every sample to a
   teacher. A single-teacher setup skips the column; a missing column or an
   unmatched key raises instead of mis-routing.
2. **Split and pad.** The batch is split into one sub-batch per teacher, and
   each sub-batch is padded by repeating its own rows up to a multiple of the
   teacher's `world_size ×` scoring micro-batch size — so every data-parallel
   rank gets a non-empty shard that divides evenly into forward micro-batches,
   whatever the step's task mix is.
3. **Score.** All sub-batches are dispatched before any result is awaited, so
   teachers score concurrently. Each teacher replays the stored student states
   through its own transformer and returns its per-step transition means.
4. **Reassemble.** Outputs are unpadded, concatenated, and restored to the
   input row order, then merged into the batch as `teacher_prev_sample_mean`.

Routing and padding happen on the driver, before dispatch, so teacher
placement is purely a resource decision: colocated teachers and a standalone
pool see identical inputs and produce identical outputs.

## Configuration Parameters

### `distillation.enabled` (bool)

Whether on-policy distillation is enabled. Default: `false`. When `true`, the
frozen teachers are built (in the actor worker, or on their own resource pool)
and the trainer scores every rollout batch with them before the actor update.

### `distillation.n_gpus_per_node` (int)

Number of GPUs per node in the teacher resource pool. Default: `0`. Only read
when `nnodes > 0`.

### `distillation.nnodes` (int)

Number of nodes in the teacher resource pool. Default: `0`, which colocates
the teachers with the actor on the actor's GPUs. Set to `≥ 1` to give the
teachers their own `teacher_pool` of `n_gpus_per_node × nnodes` GPUs.

**Constraint:** with `nnodes > 0`, the pool size must exactly equal the sum of
`world_size` across all configured teachers, or
`DiffusionDistillationConfig.__post_init__` raises.

### `distillation.teacher_key` (str)

Column of the batch's non-tensor data used to route each sample to the right
teacher in multi-teacher setups. Default: `"data_source"`.

- **Single-teacher**: ignored (everything goes to the sole teacher).
- **Multi-teacher**: the value of `sample[teacher_key]` must match the `key`
  of one of the configured teachers, or `DiffusionTeacherManager` raises.

### `distillation.teacher_models` (dict)

Map of teacher entries. Each value is a
`DiffusionDistillationTeacherModelConfig`.

The single-teacher entry is named `teacher_model` by convention. **Pitfall:**
when adding more named teachers, the `teacher_model` entry is silently popped
— so do **not** keep `teacher_model` as one entry alongside other named
teachers. Either rely on it alone, or rename it (e.g. `teacher_model1`) and
add the others.

```bash
# WRONG: teacher_model is popped, only teacher_model2 is used
distillation.teacher_models.teacher_model.key=ocr
distillation.teacher_models.teacher_model.model_path=/ckpt/ocr_teacher
+distillation.teacher_models.teacher_model2.key=aesthetic
+distillation.teacher_models.teacher_model2.model_path=/ckpt/aesthetic_teacher

# RIGHT: rename the first teacher
+distillation.teacher_models.teacher_model1.key=ocr
+distillation.teacher_models.teacher_model1.model_path=/ckpt/ocr_teacher
+distillation.teacher_models.teacher_model2.key=aesthetic
+distillation.teacher_models.teacher_model2.model_path=/ckpt/aesthetic_teacher
```

### `distillation.teacher_models.<name>.key` (str)

Identifier used to route samples to this teacher in multi-teacher mode. Must
match the value of `sample[distillation.teacher_key]`. Default: `null`
(required for multi-teacher; auto-set to `"default"` for single-teacher).

### `distillation.teacher_models.<name>.model_path` (str)

Local path to the frozen teacher checkpoint. **Required.**

The teacher must be a full pipeline checkpoint from the same pipeline family
as the student (e.g. a fine-tuned Stable Diffusion 3.5 teacher for a Stable
Diffusion 3.5 student) and must resolve to the same scheduler configuration —
the teacher replays the student's trajectories on the student's noise grid,
and worker init raises if the resolved scheduler configs differ. LoRA
checkpoints must be merged before use; the teacher never loads adapters.

### `distillation.teacher_models.<name>.world_size` (int)

Number of GPUs this teacher occupies in the teacher resource pool. Default:
`0`. Only read when `nnodes > 0`; a single teacher auto-fills the whole pool,
multiple teachers must set it explicitly so the sum matches the pool size.
Each teacher's sub-pool is its own worker group, so teachers are scored
concurrently.

### Loss-side switches

The distillation losses and their switches live under
`actor_rollout_ref.actor` (see the [config reference](../examples/config.md)):

- `diffusion_loss.loss_mode=distill_kl` — pure distillation: the KL to the
  teacher is the only objective.
- `use_distill_loss`, `distill_loss_mode`, `distill_loss_coef` — add the
  distillation term on top of the main objective (e.g. `flow_grpo`), the same
  shape as `use_kl_loss`/`kl_loss_coef` for the reference-KL term.

Enabling the teacher without an active distillation loss (or vice versa) is
rejected at startup. The teacher runtime produces `teacher_prev_sample_mean`,
which only `distill_kl` consumes; `distill_fm_mse` has no producer on the
policy-gradient path and is rejected.

### Scoring batch size

The teacher reuses the reference model's scoring configuration:
`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`, falling back to
`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`.

## Usage

Supported scope: the policy-gradient trainer with online sampling and
FSDP/FSDP2 engines. Unsupported combinations raise at startup.

A complete working recipe is
[`examples/diffusionopd_trainer/sd35/run_sd35_medium_ocr_distill.sh`](../examples/diffusionopd_trainer.md):
SD3.5-Medium distills from an OCR-tuned teacher while the OCR reward is
monitored only, showing the student reach the teacher's reward level through
distillation alone.

Pure distillation — the student imitates the teacher, task reward is only
monitored:

```yaml
distillation:
  enabled: true
  teacher_models:
    teacher_model:
      model_path: /ckpt/sd35m-teacher
actor_rollout_ref:
  actor:
    diffusion_loss:
      loss_mode: distill_kl
    use_kl_loss: false
```

Task reward plus distillation, with the reference-KL penalty on as well
(actor, reference, and teacher are three separate model states; keep
`lora_rank: 0`, since a LoRA actor folds the reference into itself):

```yaml
distillation:
  enabled: true
  teacher_models:
    teacher_model:
      model_path: /ckpt/sd35m-teacher
actor_rollout_ref:
  actor:
    diffusion_loss:
      loss_mode: flow_grpo
    use_distill_loss: true
    distill_loss_mode: distill_kl
    distill_loss_coef: 1.0
    use_kl_loss: true
    kl_loss_coef: 0.04
```

### Multi-teacher

Several task-specialised teachers can distil into one student: every sample is
routed by its `data_source` (or whichever column `teacher_key` names) to the
teacher whose `key` matches, scored there, and the per-sample
`teacher_prev_sample_mean` is scattered back into the batch. Each teacher holds
a full copy of its weights, so the memory cost grows with the number of
teachers; with colocated teachers that memory comes out of the actor's GPUs.

```yaml
distillation:
  enabled: true
  teacher_key: data_source
  teacher_models:
    ocr:
      key: ocr
      model_path: /ckpt/sd35m-ocr-teacher
    aesthetic:
      key: aesthetic
      model_path: /ckpt/sd35m-aesthetic-teacher
actor_rollout_ref:
  actor:
    diffusion_loss:
      loss_mode: distill_kl
    use_kl_loss: false
```

### Standalone teacher pool

Teachers can run on their own GPUs instead of the actor's: `nnodes > 0`
allocates a `teacher_pool`, split into one sub-pool per teacher according to
`world_size`. The actor's GPUs are set by `trainer.n_gpus_per_node`, the
teachers' by `distillation.n_gpus_per_node`. Teacher scoring stays a serial
stage of the training step, so a standalone pool does not make it faster than
the same GPUs colocated; it isolates teacher memory from the actor and the
rollout engine's sleep/wake cycle.

```yaml
trainer:
  n_gpus_per_node: 4
distillation:
  enabled: true
  n_gpus_per_node: 2
  nnodes: 1
  teacher_models:
    ocr:
      key: ocr
      model_path: /ckpt/sd35m-ocr-teacher
      world_size: 1
    aesthetic:
      key: aesthetic
      model_path: /ckpt/sd35m-aesthetic-teacher
      world_size: 1
```

## Metrics

- `actor/distill_kl_loss` — per-step KL between student and teacher
  transition means. Under pure distillation it should be clearly positive at
  step one (the teacher's weights differ from the student's) and decrease as
  the student matches the teacher.
- `timing_s/teacher` — wall time of the once-per-step teacher scoring stage,
  reported alongside `timing_s/ref` and the other fit-loop stages.
