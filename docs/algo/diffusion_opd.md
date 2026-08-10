# Diffusion On-Policy Distillation

Last updated: 08/09/2026.

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

## Configuration Parameters

### `distillation.enabled` (bool)

Whether on-policy distillation is enabled. Default: `false`. When `true`, the
actor worker builds the frozen teacher and the trainer scores every rollout
batch with it before the actor update.

### `distillation.teacher_models.teacher_model.model_path` (str)

Local path to the frozen teacher checkpoint. **Required** when enabled.

The teacher must be a full pipeline checkpoint from the same pipeline family
as the student (e.g. a fine-tuned Stable Diffusion 3.5 teacher for a Stable
Diffusion 3.5 student) and must resolve to the same scheduler configuration —
the teacher replays the student's trajectories on the student's noise grid,
and worker init raises if the resolved scheduler configs differ. LoRA
checkpoints must be merged before use; the teacher never loads adapters.

Only the single `teacher_model` entry is supported.

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

Supported scope: the policy-gradient trainer with online sampling, FSDP/FSDP2
engines, and a single teacher. Unsupported combinations raise at startup.

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

## Metrics

- `actor/distill_kl_loss` — per-step KL between student and teacher
  transition means. Under pure distillation it should be clearly positive at
  step one (the teacher's weights differ from the student's) and decrease as
  the student matches the teacher.
- `timing_s/teacher` — wall time of the once-per-step teacher scoring stage,
  reported alongside `timing_s/ref` and the other fit-loop stages.
