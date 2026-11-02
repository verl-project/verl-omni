# Per-Stage Skip: Cache and Replay RL Pipeline Stages

Last updated: 11/02/2026

Stage skip replays the output of one RL pipeline stage for configured global
steps instead of running it: reward scoring, old log-probs, reference outputs,
advantages, the actor update, or the mid-cycle weight sync. It extends rollout
skip, which replays generations, to the stages computed after them. Everything
is built on verl's `SkipManager` (`BaseSkip`, `SKIP_REGISTRY`,
`@SkipManager.annotate`) and every role starts disabled.

Use it to:

- **Replay a training step deterministically.** Freeze rollout, reward and old
  log-probs, then compare training metrics against the original run. This is
  the fixed input you need to debug a divergence.
- **Profile one stage in isolation.** Replay rollout to time the training phase
  alone, or replay the reward model to remove its cost, with no code changes.
- **Bisect a divergence.** With rollout frozen, freeze reward, then reference
  outputs, then old log-probs. Each flip is one config line.
- **Bring up a recipe cheaply.** Run the plumbing around the update (advantage,
  weight sync, checkpoint, metrics) with `actor_update` set to `empty`.

Skipped runs are debugging tools. Do not use their metrics for convergence
claims or release numbers.

## Quick start: SD3.5-Medium on CUDA

[`run_sd35_medium_ocr_replay.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/stage_skip/run_sd35_medium_ocr_replay.sh)
wraps the
[V1 OCR recipe](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh)
and runs it twice with the same arguments. The first pass finds no caches, so it
runs every stage and dumps its output. The second finds them and replays.

### Prerequisites

1. Follow the [installation guide](https://github.com/verl-project/verl-omni/blob/main/docs/start/install.md),
   then install the OCR reward dependency:

   ```bash
   uv pip install -e ".[ocr]"
   ```

2. Accept the licence of the gated `stabilityai/stable-diffusion-3.5-medium`
   model on Hugging Face and log in with `huggingface-cli login`.

3. Get the raw OCR prompts from the
   [Flow-GRPO repository](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr),
   place them under `$WORKSPACE/data/ocr` (`WORKSPACE` defaults to `$HOME`), and
   convert them:

   ```bash
   python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
     --input_dir $WORKSPACE/data/ocr \
     --output_dir $WORKSPACE/data/ocr/sd3
   ```

4. Have three visible GPUs: the base recipe uses 2 for actor and rollout and 1
   for the reward model. Rollout dumps hold every trajectory's latents, so keep
   disk space free under `STAGE_SKIP_RUN_DIR`.

### Run the replay check

From the repository root:

```bash
bash examples/stage_skip/run_sd35_medium_ocr_replay.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `OCR_WORKSPACE` / `WORKSPACE` | `$HOME` | Dataset root of the base recipe |
| `STAGE_SKIP_RUN_DIR` | `$WORKSPACE/stage_skip/sd35_medium_ocr` | Dumps, metrics and checkpoints |
| `NUM_STEPS` | `3` | Training steps per pass |
| `STAGES` | `reward old_log_prob` | `omni_skip` roles to replay; `skip.rollout_tq` is always on |

Hydra overrides after the mode reach every pass. The script refuses to start if
the dump directory already holds a dump, because the record pass would replay
it. Delete the directory or point `STAGE_SKIP_RUN_DIR` somewhere new.

Validation is off (`trainer.test_freq=-1`), checkpoint resume is off, and
`actor_rollout_ref.actor.fsdp_config.full_determinism=True` seeds the LoRA
initialization so that both passes start from the same weights.

### What a correct run shows

The script exits non-zero if any of these fails:

1. **The record pass reaches each stage.** Each role in `STAGES` has at least
   `NUM_STEPS` payloads under `$STAGE_SKIP_RUN_DIR/dump`.
2. **The replay pass replays each stage.** Its metrics report
   `skip/<role>/steps = NUM_STEPS`.
3. **Training reproduces.** Every `actor/*` and `critic/*` metric in
   `metrics/record.jsonl` and `metrics/replay.jsonl` matches exactly at every
   step. The last line reads `REPLAY OK: ...`.

The logs of the replay pass also show `[RolloutTqSkip()] Injected ...` and
`Replaying <role> output at step ...` lines for every step, and its
`timing_s/gen`, `timing_s/reward` and `timing_s/old_log_prob` drop sharply.

A mismatch is reported per step and metric as `record=... replay=...`:

- If `critic/*` differs, rollout or reward did not replay faithfully. Look for
  a missing `Injected` or `Replaying` line first.
- If only `actor/*` differs, the inputs match and the update itself is not
  deterministic. Check that `full_determinism` took effect, that the GPU count
  is unchanged, and whether you added an attention backend (`FA3=1`) that the
  deterministic flags do not cover.

### Profile the training phase

Once a dump exists, run rollout from the cache while everything else runs for
real:

```bash
bash examples/stage_skip/run_sd35_medium_ocr_replay.sh profile
```

Then compare per-stage time between the full pass and the trainer-only pass:

```bash
python3 - "$WORKSPACE/stage_skip/sd35_medium_ocr/metrics" <<'EOF'
import json, sys
for name in ("record", "profile"):
    rows = [json.loads(line) for line in open(f"{sys.argv[1]}/{name}.jsonl")]
    timing = {k: v for r in rows if r["step"] == 2 for k, v in r["data"].items() if k.startswith("timing_s/")}
    print(name, {k: round(v, 2) for k, v in sorted(timing.items())})
EOF
```

## Configuration

Stage roles live in the top-level `omni_skip` block of `diffusion_trainer.yaml`.
They stay out of `skip:` because verl parses that block against its own fixed
schema. A misspelled role fails at startup.

```yaml
omni_skip:
  old_log_prob:
    enable: true
    dump_dir: "~/.verl/stage_dump"
    steps: [1, 2, 3]
    action: cache
  actor_update:
    enable: true
    steps: [2]
    action: empty
```

### Roles

| Role | What it replaces | Actions | Trainers |
|---|---|---|---|
| `reward` | Colocated reward-model scoring | `cache`, `repeat` | V1, V0 |
| `old_log_prob` | Recomputed old log-probs | `cache`, `repeat` | V1, V0 (policy gradient) |
| `ref_log_prob` | Reference log-probs or, for direct preference, reference noise preds | `cache`, `repeat` | V1, V0 |
| `advantage` | Driver-side advantage computation | `cache`, `repeat` | V1 (policy gradient) |
| `actor_update` | The actor update | `cache`, `repeat`, `empty` | V1, V0 |
| `weight_sync` | Mid-cycle rollout weight refresh | `empty` | V1 async modes |

Rollout itself is replayed by verl's roles under `skip:`. Use
`skip.rollout_tq` on the V1 trainer and `skip.rollout` on the V0 trainer. The
omni (AR) stack delegates its trainer to verl and supports only those `skip:`
roles.

### Actions

- `cache` replays the exact step. On a miss the stage runs and dumps its output.
- `repeat` replays the closest complete step (nearest below, else nearest above).
  With no dump at all, the stage runs and dumps.
- `empty` skips the stage with no cache.

A step not listed in `steps` always runs for real. Validation never touches
stage caches: its batches carry a `validate` marker that bypasses every role.

### What a skipped stage does

- `reward`, `old_log_prob`, `ref_log_prob` and `advantage` return the cached
  output as the real stage produced it. The `advantage` cache holds the full
  batch with advantages, so it is the largest stage dump.
- `actor_update` with `empty` no-ops the update and the metrics pipeline runs on
  empty metrics. `cache` and `repeat` replay the metrics of an earlier real
  update, with the weights left unchanged. Neither writes checkpoint content.
- `weight_sync` with `empty` skips the refresh but still wakes the replicas that
  were slept for the colocated reward model, on pre-update weights. Lifecycle
  syncs (init, step end, validation) always run. Training quality silently
  degrades, which is why the role requires `ack_stale_weights: true`.

### Cache layout

```text
{dump_dir}/{experiment_name}_{project_name}/{role}/{step}/{call_idx}/stage_payload.pt
```

`call_idx` is the stage call's ordinal within the step. It is always 0 unless
the V1 separate-async trainer splits a step into `parameter_sync_step`
mini-batches, in which case each call owns a slot, the same scheme as verl's
`rollout_tq`. `repeat` only considers steps whose slots are all dumped. Rollout
dumps use verl's layout under `skip.<role>.dump_dir` and do not collide.

Caches are keyed by experiment, project, role and step, not by the rest of the
config. Delete the directory, or use a new one, after changing anything that
affects a stage's output.

### Counters

The trainer logs `skip/<role>/calls` and `skip/<role>/steps` at the end of the
run and prints a summary, so a skipped run describes itself. On a record pass
both are 0, because a stage that ran and dumped was not skipped.

## Guardrails

`init_skip_manager` fails at startup for configurations that would be silently
inert or unsafe:

| Configuration | Why it is rejected |
|---|---|
| An enabled role with empty `steps` | It could never fire |
| A misspelled role under `omni_skip` | It would be ignored |
| `weight_sync` without `ack_stale_weights: true` | Replicas keep stale weights |
| `weight_sync` on V0 or in `trainer.v1.trainer_mode=sync` | Those trainers have no mid-cycle sync |
| `reward` or `weight_sync` with streaming rewards | Scoring happens inside rollout; use the rollout roles |
| `ref_log_prob` with no reference policy | The stage never runs |
| `old_log_prob` + `algorithm.rollout_correction.bypass_mode` | Both substitute old log-probs |
| `old_log_prob` or `advantage` with `algorithm.trainer_type=direct_preference` | Those stages never run |
| `old_log_prob` when the loss reads no recomputed log-probs | The stage never runs |
| `advantage` on the V0 trainer | V0 computes advantages inline, with no hook |

A dump that fails, such as on a full disk, raises instead of warning. A silently
missing cache would turn a later replay into a real run.

## Limitations

- Stage caches are row-aligned with the batch that produced them, so replay
  them together with rollout (`skip.rollout_tq` or `skip.rollout`). A cached
  stage on fresh generations pairs its outputs with different samples.
- `cache` with the same config and data order keeps rows aligned. `repeat`
  against a different batch size fails loudly at the union and assert sites, but
  a same-sized batch from another step replays silently, so treat `repeat` as a
  plumbing tool rather than a reproduction tool.
- Replaying a stage does not free the memory of its model: the reward model and
  rollout replicas still load, so profile runs need the same GPUs.
- The V1 and V0 `weight_sync` limits above mean the role only applies to async
  modes that also have a colocated reward model.
