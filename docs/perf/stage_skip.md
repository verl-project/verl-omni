(stage_skip)=
# Stage Skip: Cache and Replay Pipeline Stages

Last updated: 10/20/2026

Stage skip replays the output of one RL pipeline stage — reward scoring,
old log-prob, reference outputs, advantage, actor update, weight sync — for
configured global steps instead of running the model forward. It is the
per-stage generalization of rollout skip: rollout skip replays generations,
stage skip replays everything computed *after* them. Both build on verl's
`SkipManager` (`BaseSkip` / `SKIP_REGISTRY` / `@SkipManager.annotate`); the
stage roles, hooks, and configs live entirely in verl-omni.

Typical uses:

- **Deterministic replay of a step.** Enable rollout skip plus
  `old_log_prob` + `reward` caches, rerun the same steps, and compare training
  metrics bit-for-bit against the original run — the prerequisite for
  debugging a divergence against a fixed input.
- **Stage-isolated profiling.** Skip rollout (cache) to measure the training
  phase alone, or skip the colocated reward model to measure trainer-side
  overhead. No code changes, one config block.
- **Divergence bisection.** Freeze reward outputs first, then ref outputs,
  then old log-probs — each flip is one config line.
- **Cheap pipeline bring-up.** Run the full plumbing (advantage, weight sync,
  checkpoint, metrics) with `actor_update` set to `empty` before the first
  expensive GPU run.

## Configuration

Stage skips are configured under the top-level `omni_skip` block of
`diffusion_trainer.yaml` (kept separate from verl's `skip:` block, which verl
parses against its own fixed schema):

```yaml
omni_skip:
  old_log_prob:
    enable: true
    dump_dir: "~/.verl/stage_dump"
    steps: [1, 2, 3]
    action: cache
```

| Role | Stage call | Actions |
|---|---|---|
| `reward` | Colocated reward-model scoring | `cache`, `repeat` |
| `old_log_prob` | Recomputed old log-probs (policy gradient) | `cache`, `repeat` |
| `ref_log_prob` | Reference outputs: ref log-probs (policy gradient) or ref noise preds (direct preference) | `cache`, `repeat` |
| `advantage` | Driver-side advantage computation | `cache`, `repeat` |
| `actor_update` | Actor update | `cache`, `repeat`, `empty` |
| `weight_sync` | Mid-cycle rollout weight refresh (async modes) | `empty` |

- `cache` replays the exact step; `repeat` replays the closest dumped step
  (nearest below, else nearest above); `empty` skips the stage without any
  cache. A step not listed in `steps` always runs for real. On a cache miss
  the stage runs normally and its output is dumped.
- Caches live under
  `{dump_dir}/{experiment_name}_{project_name}/{role}/{step}/`. Rollout skip
  uses its own directory layout under `skip.rollout.dump_dir`, so the two
  dumps do not collide.
- Every skipped call is counted; the trainer logs `skip/<role>/calls` and
  `skip/<role>/steps` at the end of the run and prints a summary.

## What each skipped stage does

- `reward` / `old_log_prob` / `ref_log_prob` / `advantage`: the cached stage
  output is returned to the pipeline exactly as the real stage produced it.
  The `advantage` cache holds the full batch snapshot with advantages — the
  largest of the stage dumps.
- `actor_update` with `empty`: the update is a no-op and the metrics pipeline
  still runs (empty metrics). No checkpoint content is produced from a
  skipped update. `cache`/`repeat` replay the metrics of an earlier real
  update.
- `weight_sync` with `empty`: only the *mid-cycle* refresh in async modes is
  skipped, so colocated rollout replicas keep pre-update weights for the rest
  of the step. Lifecycle syncs (init, step end, validation) always run, so
  the pipeline never stalls — training quality does silently degrade, which
  is why the role requires `ack_stale_weights: true`.

Validation never touches stage caches: validation batches carry a
`validate` marker and bypass every skip, and rollout skip bypasses them the
same way it always has.

## Guardrails

`init_skip_manager` fails the run at startup for configurations that would be
silently inert or unsafe:

- `old_log_prob` + `algorithm.rollout_correction.bypass_mode` — both
  substitute old log-probs.
- `old_log_prob` or `advantage` with
  `algorithm.trainer_type=direct_preference` — those stages never fire.
- `old_log_prob` when the configured loss reads no recomputed log-probs —
  the stage call never fires.
- `weight_sync` without `ack_stale_weights: true`.

## Scope and limitations

- The diffusion V1 (sync / separate_async) and V0 trainers wire the stage
  roles. The omni (AR) stack delegates its trainer to verl and supports only
  verl's `skip:` roles (`rollout`, `async_rollout`, `rollout_tq`).
- The v0 `weight_sync` call sites are lifecycle-only (they wake the rollout
  replicas), so the role has no v0 hook.
- Replay assumes the batch shapes and data order of the cached step match the
  current step; a `repeat` replay against a different batch size fails loudly
  at the union/assert sites.
- Skipped runs are debugging tools: do not use their metrics for convergence
  claims or release numbers.
