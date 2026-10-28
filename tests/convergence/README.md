# L4 Convergence Tests

L4 answers a question the other layers cannot: **does a real training recipe, with
real weights and a real dataset, still converge?** L1 proves CPU APIs behave, L2
proves a tiny random model reaches a couple of steps, and L3 proves that
numerical/perf behaviour on tiny models did not drift. None of those can detect a
recipe that trains without error but no longer learns.

L4 is a **release-readiness gate**, not a pull-request check. It is driven by
declarative recipes and it fails closed: when the required checkpoints, dataset,
or hardware are not present, the case is reported as `skipped` with the exact
unmet requirement, and the release verdict becomes `incomplete`. It never
substitutes a tiny model, a synthetic dataset, or a fraction of the configured
steps and calls the result a convergence pass.

## Contents

| Path | Purpose |
| --- | --- |
| `recipes/*.yaml` | Declarative L4 recipes: launcher, assets, hardware, metrics, tolerances, baseline. |
| `recipe_registry.py` | Schema/semantic validation and precondition checks. |
| `curves.py` | Per-step reward/loss curve extraction from the trainer's console log or `metrics.jsonl`. |
| `compare.py` | Comparability contract plus direction-aware `atol + rtol * |reference|` scoring. |
| `report.py` | Aggregate release-readiness report (`release_readiness.json` / `.md`). |
| `run_convergence.py` | Runner/orchestrator CLI. |
| `run_l4_convergence.sh` | Shell entrypoint used locally and by the workflow. |
| `test_l4_*_on_cpu.py` | L1-collected unit tests for all of the above (no GPU, no checkpoints). |

## Quick start

```bash
# 1. Static check: are the recipes valid, and is this machine even able to host them?
MODE=preflight bash tests/convergence/run_l4_convergence.sh

# 2. Create reviewed baselines. Release owners only, on the release runner.
MODE=baseline bash tests/convergence/run_l4_convergence.sh

# 3. Verify against the stored baselines and write the release report.
MODE=verify bash tests/convergence/run_l4_convergence.sh
```

Useful knobs:

```bash
MODE=verify L4_CASES=sd35_medium_flowgrpo bash tests/convergence/run_l4_convergence.sh
MODE=verify L4_TIMEOUT_MINUTES=30     bash tests/convergence/run_l4_convergence.sh
REGISTRY=/path/to/recipes OUTPUT_ROOT=/path/to/out MODE=verify bash tests/convergence/run_l4_convergence.sh
```

## Recipe anatomy

```yaml
case_id: sd35_medium_flowgrpo     # unique; also the per-case artifact directory
algorithm: flow_grpo              # declared so a baseline cannot be reused across algorithms
precision: bf16                   # part of the comparability contract
release_gate: true                # whether this case can block a release

recipe:
  launcher: examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh
  overrides:                      # appended after the launcher's own arguments
    - data.train_files={dataset:train}
    - actor_rollout_ref.model.path={model:policy}
    - trainer.total_training_steps=100

models:                           # must be real released checkpoints
  policy:
    kind: real
    source: stabilityai/stable-diffusion-3.5-medium
    revision: main
    local_path: /models/stable-diffusion-3.5-medium
    env_path: L4_SD35_MEDIUM_PATH  # environment override, wins over local_path

dataset:
  kind: real
  train: data/ocr/sd3/train.parquet   # relative to dataset.env_root when that is set
  val: data/ocr/sd3/test.parquet
  env_root: L4_OCR_DATA_ROOT

hardware:
  min_gpus: 3
  min_gpu_memory_gb: 40          # free memory each GPU must have at run start
  gpu_architectures: [sm89, sm90, sm100]
  min_free_disk_gb: 200

budget:
  timeout_minutes: 720
  total_training_steps: 100

metrics:
  warmup_steps: 10                 # steps excluded before scoring
  final_window: 10                 # steps averaged for the final value
  min_points: 40                   # must be >= warmup_steps + final_window
  tracked:
    - {name: "critic/rewards/mean", direction: higher, optional: false}
    - {name: "actor/loss", direction: lower, optional: true}

convergence:
  atol: 0.02                       # |current - reference| <= atol + rtol * |reference|
  rtol: 0.05
  min_improvement: 0.0             # required in-run progress (signed: positive is better)
  require_finite: true             # NaN/Inf in the curve fails the case

baseline:
  artifact_name: l4-convergence-baseline
  branch: main
```

Rules the registry enforces:

- `models.*.kind` must be `real`, and a `source` containing `tiny-random` is rejected.
- `dataset.kind` must be `real`.
- `case_id`s must be unique; the whole registry is rejected otherwise.
- `min_points >= warmup_steps + final_window`, at least one required metric, and
  every direction must be `higher` or `lower`.
- At least one recipe must be `release_gate: true`; an all-ungated registry means
  nothing would gate a release.

### Override placeholders

Overrides are resolved before the launcher is executed, so no recipe embeds a
machine-specific absolute path:

| Placeholder | Resolves to |
| --- | --- |
| `{model:<name>}` | `env_path` environment value, else the model's `local_path`. |
| `{dataset:<key>}` | `dataset.<key>`, joined with `dataset.env_root` when set. |
| `{output_root}` | The runner's `--output-root`. |
| `{case_dir}` | `<output_root>/current/<case_id>`, for checkpoints and rollout dumps. |

An unknown placeholder is a hard error: silently leaving a path unresolved would
let a recipe train the wrong model.

## Statuses and evidence levels

Every case writes `current/<case_id>/result.json` with a `status` and an
`evidence_level`. Only `passed` **with** `compared` evidence can make a release
ready.

| Status | Meaning |
| --- | --- |
| `passed` | Real run finished, curve scored, baseline matched within tolerance. |
| `failed` | Real run finished but regressed (or the launcher exited non-zero). |
| `incomparable` | The run's contract differs from the baseline (weights, data, algorithm, precision, scoring window, or GPU shape). Never treated as "no regression". |
| `timeout` | The run exceeded `budget.timeout_minutes`; partial logs and curve are kept. |
| `skipped` | A declared precondition is unmet (assets or hardware). No training was started. |
| `invalid` | Recipe valid but the run produced no required metric, or no reviewed baseline exists yet. |
| `not_run` | `preflight` mode: preconditions were checked, no run was requested. |
| `baseline_created` | `baseline` mode stored a baseline; this does not verify convergence. |

| Evidence level | Meaning |
| --- | --- |
| `static` | Recipe parsed, preconditions checked. No training evidence. |
| `run` | A real run happened and a curve was extracted. |
| `compared` | The curve was compared against a reviewed baseline. |

## Release verdict

`release_readiness.json` / `release_readiness.md` aggregate the gated cases:

| Verdict | Meaning |
| --- | --- |
| `ready` | Every gated case is `passed` with `compared` evidence. |
| `blocked` | At least one gated case ran and failed. |
| `incomplete` | At least one gated case is missing, skipped, timed out, invalid, incomparable, or baseline-only. |

`incomplete` exits non-zero. That is intentional: a release gate that cannot
produce evidence must not report success. Until the L4 runner is attached to a
production cluster with the real assets, the weekly workflow is expected to be
red with `incomplete` rather than green with no evidence.

## Baselines

```text
outputs/l4_convergence/
|-- baseline/<case_id>/baseline.json     # reviewed reference: contract + curve summary
|-- current/<case_id>/result.json        # this run's status, contract, curve, comparison
|-- current/<case_id>/train.log          # captured launcher stdout/stderr
|-- release_readiness.json
`-- release_readiness.md
```

A baseline stores the run contract next to the curve summary. A later run whose
weights, dataset, algorithm, precision, scoring window, step budget, or GPU shape
differ is reported as `incomparable` instead of being quietly scored against the
wrong reference. The commit SHA is recorded but is deliberately excluded from
comparability, because the commit under test is normally exactly what changed.

A run never becomes its own baseline: `baseline` mode always reports
`baseline_created`, which is `incomplete` for release purposes.

In GitHub Actions the baselines live in the `l4-convergence-baseline` artifact
(retained 90 days), keyed by branch, mirroring the L3 nightly policy.

## Failure triage

1. `skipped` — read `preconditions.reason` in `result.json`; it names the exact
   missing checkpoint, dataset shard, or GPU capacity. Attach the missing
   resource, do not loosen the check.
2. `invalid` with "no reviewed baseline" — run `MODE=baseline` on the release
   runner first and review the produced curve.
3. `incomparable` — read `comparison.mismatches`; a baseline may only be used for
   the exact contract it was created from.
4. `failed` — read `comparison.metrics[*].reasons` and
   `current/<case_id>/train.log`. Confirm the regression is real and intended
   before refreshing any baseline.
5. `timeout` — check whether the budget or the runner is the problem; the partial
   curve and log are preserved.

## Adding a recipe

1. Pick an existing example launcher under `examples/`; do not fork a new training
   script for L4.
2. Declare the real checkpoints, the real dataset shards, and the hardware shape
   the recipe genuinely needs. Do not lower `min_gpu_memory_gb` to make a case run
   on unsuitable hardware. `local_path` is only checked for existence, so also note
   in the recipe when a checkpoint is **gated** (for example
   `stabilityai/stable-diffusion-3.5-medium`) or needs a package that is not on
   PyPI: the precondition cannot verify access, and `skipped` will otherwise look
   like a missing download rather than a missing entitlement.
3. Declare the convergence metric, its direction, the warmup/window, and a
   tolerance you can defend. `atol` carries the near-zero case; `rtol` carries the
   scale.
4. Run `MODE=preflight` to confirm the recipe is well-formed.
5. Create the first baseline with `MODE=baseline` on the release runner, review the
   curve and its `env`/`contract` provenance, then rely on `MODE=verify`.

### Tightening `min_improvement`

`min_improvement` ships as `0.0`: no reviewed baseline exists yet, and an invented
floor is indistinguishable from a real one. Once a release owner has produced and
reviewed a baseline, derive the floor from it in one auditable step:

```bash
python3 -m tests.convergence.derive_thresholds \
    --recipe tests/convergence/recipes/<case>.yaml \
    --baseline outputs/l4_convergence/baseline/<case>/baseline.json \
    --retention 0.5 --write
```

The proposal is `retention * baseline_improvement` on the recipe's driving metric
(the first required tracked metric, or `--metric`), and `--write` rewrites exactly
that one line. The tool **refuses** and exits non-zero when the baseline did not
improve, did not measure an improvement, or lacks the driving metric, so a
threshold can never be tightened from absent evidence. `retention` is validated to
be in `(0, 1]`.

### Recipe pitfalls

All of these were hit while validating this layer on real hardware. They are
config and resource traps rather than harness bugs, and they cost real GPU hours:

- **Do not null out `trainer.total_epochs`.** The diffusion trainer evaluates
  `len(train_dataloader) * trainer.total_epochs` before it consults
  `trainer.total_training_steps`, so `total_epochs=null` raises a `TypeError`
  during dataloader setup. Leave `total_epochs` alone and set
  `trainer.total_training_steps` alone; that is what L3 does too.
- **Do not pass `+` for a key that already exists** (and vice versa). A `+key=`
  override fails when the key is already in the config, and a bare `key=` fails
  for a key that is not. Use `++` when you are unsure; verify the knob against the
  launcher's own config before adding it.
- **DLO is the wrong offload mechanism on a shared device.** With
  `enable_distributed_layerwise_offload=true`, `wake_up()` re-maps the engine's
  whole saved CUDA VMM pool at once and raises
  `CUDA Error: out of memory at cumem_allocator.cpp:163` during
  `actor_rollout_update_weights`. Leaving DLO off and letting the launcher's
  `layered_summon=True` stream weights layer-wise avoids that spike.
- **Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** It conflicts
  with vLLM-omni's cumem sleep/wake path: the worker OOMs at
  `cumem_allocator.cpp:163` while gigabytes of the device are still reported free.
- **The reward `enable_resource_pool=True` pool is disjoint from the actor pool.**
  The GPU budget is `trainer.n_gpus_per_node + reward.reward_model.n_gpus_per_node`;
  if the two sum to more than the devices given, the trainer fails with
  `Total available GPUs N is less than total desired GPUs M`. Either budget the
  extra card or drop the pool and accept that the reward engine's reservation
  lands on a rank card.
- **`text_encoder_tp_size` must be 1 or the tensor parallel size.** An
  intermediate value is rejected, so TP=3 leaves the whole text encoder on one
  rank; check the encoder's own head count before assuming a TP degree is usable.
- **A free-memory floor that is too low converts `skipped` into a CUDA OOM.** At
  TP=2 the Qwen-Image diffusion worker rank alone needs ~36.9 GiB on a 44.40 GiB
  L20, and that footprint is weight-dominated: cutting the denoise steps, pinning
  the diffusion engine's `max_num_seqs`, and halving `train_batch_size` did not
  change the failure at all. Declare the real per-card requirement, and remember
  that the check measures *free* memory, so a co-tenant makes a card unusable long
  before its nominal capacity is reached.
- **`gpu_memory_utilization` cannot fix a sleep/wake OOM.** Level-1 sleep re-maps
  the engine's **weights** pool (`allocator.wake_up(tags=["weights"])`), whose size
  is set by the model, while `gpu_memory_utilization` only sizes the KV/workspace
  reservation. A failing wake therefore does not respond to that knob at all.
- **Budget the colocated actor and rollout engine together, not separately, and do
  not assume more tensor parallelism closes the gap.** The two must be resident at
  the same instant on the same card when the rollout stage wakes. Measured for
  Qwen-Image on a 44.40 GiB L20: the actor keeps **24.32 GiB** resident at TP=2 and
  **31.80 GiB** at TP=4 (`After FSDP … memory used/total (GB)`), while the rollout
  engine needs **33.75 GiB per rank** before sleep — a figure that is effectively
  TP-invariant, because it is the engine's own transformer plus its pipeline
  buffers, VAE and text encoder rather than the actor's parameters. The demand at
  the wake is therefore 58-65 GiB, and the run dies with
  `wake_up failed on a stage: CUDA Error: out of memory at cumem_allocator.cpp:163`.
  TP=4 made the actor figure *worse* while the engine stayed the same, so TP=1/2/3/4
  all fail here; no batch, canvas, prompt, or engine-slots knob moves it. Verify the
  two residency numbers before choosing a card, and note that dropping the reward
  judge entirely does not help, because the actor and the rollout engine are the
  conflict.
- **The actor is not fully released when the rollout stage wakes.** That is the
  proximate cause above: at the wake the failing message names the *actor* process,
  e.g. `GPU 1 has a total capacity of 44.40 GiB of which 2.69 MiB is free. Process
  1642…`. The stack sets `self.rollout.sleep_level = 1` (a Level-1 sleep, which
  keeps memory) and warns `Setting the sleep level to 1 may cause a memory
  overflow`. A Level-2 sleep, or placing the rollout engine outside the actor's
  pool, is what would actually fit this recipe on a 44.40 GiB card.
- **`param_offload` is a two-sided trade, and neither setting is free.** At TP=4,
  `param_offload=True` is what fits the *device*, because the launcher's
  layer-wise summon keeps only a shard on card (shards loaded 9/9 in 6 s, and each
  worker logged `Sleep Level 1: physically freed 33.64 GiB`) — but the offload then
  needs roughly **317 GB of host RAM**. `param_offload=False` relieves host RAM
  (peak ~146 GiB) and instead fills the card: all four ranks fail at orchestrator
  init with `torch.OutOfMemoryError … Tried to allocate 108.00 MiB … of which
  66.00 MiB is free`. Size the container for the setting you choose, and remember
  that Ray kills workers above **95 % of the container cap**, so a TP=4 offload run
  needs a cap of ~380 GiB or more.
- **Re-check the capacity gate immediately before each mode, not once at startup.**
  A `baseline` run is long enough that a window open at container-pick time can
  close before the case starts. Pre-warming a 54 GB page cache took 40 minutes, and
  during it a co-tenant grew from 12.4 GB to 31.5 GB on one selected card, so the
  harness correctly reported `preconditions unmet: … capable=4 idle=3 of 4 visible`
  only after all the setup work had been spent.

## CI wiring

`.github/workflows/l4_weekly_convergence.yml` runs weekly, on manual dispatch, and
on pull requests labelled `L4-convergence-ci`. It keeps read-only permissions,
uses the same dynamic-runner and baseline-artifact mechanics as L3, and uploads
current results, logs, and the release report even when the gate fails.

L4 needs production-class hardware. The current GPU CI runner cannot host the
declared recipes; that is why the preconditions are declared per recipe and why a
`skipped` case fails the gate instead of silently passing it.
