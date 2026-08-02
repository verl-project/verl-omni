(diffusion_opd)=
# Online Policy Distillation for Diffusion Models

Last updated: 07/31/2026

Online policy distillation (OPD) trains a diffusion student against a frozen
teacher on the trajectories the student itself explored during rollout. Instead
of one scalar reward per image, every denoising step contributes a dense,
per-dimension target, so the student receives far more signal per sample than a
policy-gradient run does.

Unlike token-level OPD, the exchanged quantities are continuous: transition
means and flow/noise predictions, not token log-probabilities.

## When to use it

Use OPD when you already have a stronger checkpoint of the *same* model family
— a GRPO fine-tune, a larger sibling, a merged LoRA expert — and want a cheaper
or smaller student to reach its behaviour. It is not a replacement for reward
optimisation: the teacher defines the target, so the student cannot exceed it by
distillation alone.

Distillation can run on its own or as an auxiliary term next to a
policy-gradient loss, and it coexists with a KL-to-reference penalty: the
reference policy and the teacher are independent identities with different
lifecycles.

## What is supported

The architecture is modality-general; this implementation is deliberately not.

| | Supported today |
| --- | --- |
| Pipeline / algorithm | `(StableDiffusion3Pipeline, flow_grpo)` |
| Trainer | policy-gradient (`algorithm.trainer_type=policy_gradient`) |
| Sampling | online (`algorithm.sample_source=online`) |
| Loss | `distill_kl`, standalone or auxiliary |
| Backend | FSDP (`fsdp` / `fsdp2`) for both actor and teacher |
| Placement | colocated |
| Teachers | exactly one |

Every other combination is rejected at startup rather than failing later. Two
parts of the contract are SD3-specific: the teacher request requires
`pooled_prompt_embeds`, which the Qwen FlowGRPO rollout does not emit, and
scheduler validation loads a checkpoint's `scheduler/` directory, which Bagel
and Wan do not provide. Generalising needs adapter-owned hooks, not a config
flag.

`distill_fm_mse` is a valid loss mode but has no producer on the
policy-gradient path — that path computes no `noise_pred` at all — so selecting
it with a teacher enabled is rejected. Its producer belongs to the
direct-preference follow-up.

## Quick start

A teacher entry is a path plus a placement. Everything else is inherited from
the actor, derived from the checkpoint, or a frozen default:

```yaml
actor_rollout_ref:
  teacher:
    enabled: true
    models:
      default:
        model:
          path: /ckpt/sd35m-ocr-grpo-merged
    placement:
      mode: colocated
  actor:
    diffusion_loss:
      loss_mode: distill_kl     # pure distillation
```

As an auxiliary term next to a policy-gradient loss instead:

```yaml
actor_rollout_ref:
  actor:
    diffusion_loss:
      loss_mode: flow_grpo
    use_distill_loss: true
    distill_loss_mode: distill_kl
    distill_loss_coef: 1.0
```

Both forms work with `use_kl_loss: true`; the reference policy and the teacher
are separate models and are scored separately.

## How a step runs

```text
rollout            student explores; emits all_latents / all_timesteps
   |
teacher replay     frozen teacher re-evaluates those exact states
   |               -> teacher_prev_sample_mean
   |
actor update       distill_kl compares the two transition means
```

Teacher scoring is batched once per step, after rollout and before the actor
update. It is not a per-timestep RPC: a diffusion request carries whole latent
trajectories plus scheduler conditioning, which would make a chatty protocol
expensive.

Teacher identity is an algorithm-level role, and teacher GPU placement is a
deployment-level config. The teacher is registered as `Role.TeacherModel`
regardless of where it runs; `placement.mode` decides whether it shares the
actor's pool or owns its own.

## Configuration reference

All teacher settings live under `actor_rollout_ref.teacher`. The top-level
`distillation` key belongs to verl's *token* distillation and must stay disabled
on a diffusion run — the actor worker dispatches on it before it dispatches on
modality, so enabling it would silently replace `diffusion_loss` with the token
loss.

| Field | Meaning |
| --- | --- |
| `enabled` | build the teacher worker; must agree with the loss selection |
| `models.<key>.model.path` | the teacher checkpoint (required) |
| `models.<key>.model.trust_remote_code` / `use_shm` | default to the actor's values |
| `models.<key>.model.transformer_subfolder` | defaults to `transformer` |
| `models.<key>.engine.strategy` | `fsdp` or `fsdp2`; **not** inherited from the actor |
| `models.<key>.engine.model_dtype` | defaults to the actor's |
| `models.<key>.engine.micro_batch_size_per_gpu` | falls back to the rollout's, then the actor's |
| `placement.mode` | `colocated` (only mode implemented) |
| `placement.n_gpus_per_node` / `nnodes` | standalone only; setting them under `colocated` is an error |

`models` is a dict so multi-teacher routing can land without a config
migration; exactly one entry is accepted today.

### What the teacher does *not* configure

Anything that defines the replayed trajectory is inherited, not set:
`algo.sde_type`, `algo.noise_level`, the timestep grid, guidance scale, and the
conditioning contract. `distill_kl` compares two Gaussians at the same state and
divides by the student's transition standard deviation, so a teacher-owned
noise level would quietly make the quantity something other than a KL.

There are no offload knobs. A forward-only FSDP module is CPU-offloaded
unconditionally by both engine paths, and the engine's `to()` is a no-op under
`forward_only`, so a knob here could not change behaviour.

## The `teacher_*` batch keys

The seam between the runtime that produces targets and the losses that consume
them:

| Key | Consumed by | Produced |
| --- | --- | --- |
| `teacher_prev_sample_mean` | `distill_kl` | yes |
| `teacher_noise_pred` | `distill_fm_mse` | not on the policy-gradient path |

Targets are tensors, fp32 on CPU, shaped `[batch, steps, ...]`, validated for
shape and finiteness before they join the batch. They are never aliased to
`ref_*` names. The payload shape is revision 1: timestep subsampling or
variable-length windows would add a mask and change loss normalisation, and
would widen this seam deliberately rather than silently.

## What is validated, and when

Failures are grouped by the earliest point at which they are decidable, and all
of them raise rather than warn.

**Before Ray starts** — teacher enabled without a distillation loss, or a
distillation loss with no teacher (both directions); top-level `distillation`
armed on a diffusion run; the v1 trainer; a non-policy-gradient trainer type;
offline sampling; a non-FSDP teacher or actor backend; `standalone` placement;
resource fields set under `colocated`; more than one teacher; `distill_fm_mse`.

**After configs are constructed** — the teacher checkpoint's architecture must
match the actor's and fall inside the supported matrix, and both schedulers,
each built from its own checkpoint and stepped with the same pipeline settings,
must resolve to identical `timesteps` and `sigmas`.

**Per request** — every timestep the rollout visited must land on the shared
scheduler grid; stage 1 already proved the two grids identical.

**At worker init** — a teacher checkpoint that fails to load aborts the run; it
never falls back to actor or reference weights.

Capacity is not among these. There is no checkpoint memory estimator, so an
out-of-memory teacher fails loudly at initialisation rather than being predicted
at startup.

## Frozen-teacher guarantees

Teacher weights load once, never enter an optimizer, never receive actor
checkpoints, and are excluded from student checkpoints. The worker exposes a
parameter checksum so tests can assert the weights are unchanged across
optimizer steps.

Separation is defined by role and provenance, not by weight inequality: a
teacher may legitimately point at the same checkpoint as the reference. What
must never happen is a shared worker slot or a fallback to other weights.

## Limitations and next steps

- Only colocated placement; a standalone `teacher_pool` is a follow-up.
- FSDP only; a VeOmni teacher backend is a follow-up, and is where the
  sequence-parallel field names get unified.
- The v1 trainer does not wire the teacher and rejects the combination.
- One teacher; multi-teacher routing by task or data source is a follow-up.
- `distill_fm_mse` needs a direct-preference producer before it is usable.

## References

- Design discussion: verl-project/verl-omni#293
- Distillation losses: verl-project/verl-omni#300
- Teacher-in-ref prototype and its end-to-end evidence: verl-project/verl-omni#304
- Runtime: `verl_omni/workers/teacher_workers.py::DiffusionTeacherWorker`,
  `verl_omni/trainer/diffusion/teacher_manager.py::DiffusionTeacherModelManager`
- Validation: `verl_omni/trainer/diffusion/teacher_preflight.py::validate_teacher_preflight`,
  `verl_omni/trainer/diffusion/teacher_scheduler_checks.py`
- Config: `verl_omni/workers/config/diffusion/teacher.py`
