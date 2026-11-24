# Check pinned diffusion consistency artifacts

Last updated: 11/20/2026

## Capture on the real trainer path

The optional `verl_omni.tools.consistency_capture` library is installed through
Ray's existing worker-process setup hook. The normal trainer entrypoint and
nightly CI remain unchanged.

For a pipeline integration, first screen the existing
`rollout_corr/logprob_abs_diff_mean`, `rollout_corr/logprob_abs_diff_max` and
`rollout_corr/logprob_abs_diff/ts_*` metrics. Numerical differences alone do
not identify a pipeline bug. In the default recomputation path, rollout
log probabilities are not the old-policy values consumed by the loss;
bypass/on-policy modes have a different consistency requirement.

To collect evidence under the recipe's actual precision, LoRA and microbatch
settings, keep its normal `python3 -m verl_omni.trainer.main_diffusion_v1`
command and existing Hydra overrides. Append:

```bash
actor_rollout_ref.rollout.calculate_log_probs=true \
+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=verl_omni.tools.consistency_capture.install_trainer_capture \
+ray_kwargs.ray_init.runtime_env.env_vars.DIFFUSION_CAPTURE_DIR=/absolute/fresh/run/captures \
"+ray_kwargs.ray_init.runtime_env.env_vars.DIFFUSION_CAPTURE_STEPS='1'"
```

Use the normal PPO/SDE recomputation path, with bypass disabled. The capture
step list must remain a string in Ray's `env_vars`. Start a fresh Ray job with
these settings; an already initialized Ray runtime keeps its existing job
configuration. This add-only override applies to a recipe with no existing
setup hook; Hydra rejects adding the same key twice. It does not compose
callbacks. If a recipe already owns a callback, its owner must explicitly add
`install_trainer_capture()` to that callback and omit only the
`worker_process_setup_hook` override. Keep both `DIFFUSION_CAPTURE_*` env-var
overrides above; without a capture directory the installer intentionally does nothing.

For multi-node runs, use the same capture path on storage accessible to every
worker and the driver, or copy every worker's event directories into one replay
input root after collection. Node-local roots contain only that host's events;
the reader cannot detect probes left on another host. Keep the same capture-root
basename on every host, because it identifies the run in each event record.

The explicit Ray runtime environment forwards these variables and installs the
observer in the task-runner and engine workers. It does not select a model, change seeds, force
FP32, disable LoRA, or change the microbatch. With no capture directory,
the observer is not installed. The current integration covers diffusion
PPO/SDE old-log-probability recomputation through the shared Diffusers
engine; bypass and direct-preference algorithms do not provide that pair.

The driver saves every row and timestep of the rollout/recomputed pair.
Engine records save the consumed microbatch's log probabilities, raw
timesteps, and current/next sigma values when the observer
recognizes the resolved FlowMatch SDE adapter in its own timestep-unit map.
The map records the units of `all_timesteps` consumed by the scheduler, before
transformer-input scaling; the training adapters have no diagnostic fields.
SD3 FlowGRPO uses training-step units. Adapters using sigma units (such as BAGEL),
unrecognized adapters and out-of-tree replacements retain numerical comparison
without sigma diagnosis. The observer does not infer units from observed timestep/sigma values.
Record completion is written after the detached CPU payload and its hash.
Temporary batch metadata and engine context are restored on exceptions.
Capturing selected steps adds observation and I/O cost; it is not a
throughput benchmark.

Replay the resulting events on CPU, with the report outside the input root:

```bash
python3 tools/consistency_checking/replay_diffusion_trainer_capture.py \
  --root /absolute/fresh/run/captures \
  --output /absolute/fresh/run/report.json
```

The reader reuses the standalone checker's full `(N, T)` numeric comparison.
It reports existing-style difference metrics by raw timestep and distinguishes
`numerical_mismatch` (exit 3) from a consumed
`scheduler_sigma_timestep_mismatch` (`logic_mismatch`, exit 2).
Only the declared FlowMatch raw-timestep/sigma relationship supports that
logic classification. Incomplete event directories fail instead of passing. The reader requires at
least one driver pair and one engine probe, but cannot attest that every
expected probe was emitted.

For an integrator's negative control, temporarily perturb one consumed
scheduler sigma before recomputation, restore it in `finally`, and recompute
the same batch again before any optimizer update. Require the report to name
the affected raw timestep, sigma index and current/next field, then verify the
restored result matches the baseline. This tests a known logic fault, rather
than labelling ordinary kernel/dtype differences as a scheduling bug. Captures
do not independently attest model weights or prove a naturally occurring bug.

## Escalate to module-level localization

For a mismatch that needs module or operator evidence, use the existing
[training/inference consistency workflow](https://github.com/verl-project/verl-omni/blob/main/.agents/skills/train-infer-consistency/SKILL.md)
introduced by [#620](https://github.com/verl-project/verl-omni/pull/620).
It routes collection and analysis to the MindStudio skills instead of
maintaining a separate model-specific module map in this recipe.

The two capture formats serve different purposes:

| Artifact | Evidence and consumer |
| --- | --- |
| Shared trainer `record.json` / `payload.pt` and replay report | Saved log-probability pairs, timestep/sigma inputs and event identities; consumed by the CPU replay reader above. |
| Paired training/rollout MSProbe dumps | Module/operator tensors for the existing collection and analysis workflow; needed for module-level localization. |

The shared events are screening and pairing context for the investigation;
they do not contain complete module dumps and cannot replace either dump input
to the analysis script. SD3.5 remains a validated reference recipe, not an
automatic cross-backend module mapper.

1. Use the replay report's step, batch, rank and affected timesteps to select
   the diagnostic window. Retain the launch configuration and pairing evidence;
   event IDs and payload hashes alone do not establish equal inputs or weights.
2. Follow the existing collection workflow to obtain or validate paired MSProbe
   dumps for the same sample, branch and computation. Record both backend/source
   versions, input and weight identities, and tensor layouts.
3. Let the analysis workflow's agent propose `output_0_key_mapping.json` from
   those dumps and current implementations. Validate corresponding semantic
   boundaries and layouts, including fused/split operations; leave unsupported
   or ambiguous correspondences unresolved rather than forcing a name match.
4. Run that workflow's deterministic comparison and root-cause scripts on the
   accepted mapping and tensor data. The agent explains the resulting evidence;
   it does not decide tensor equality or turn an unverified mapping into a pass.

Recheck the mapping when model or backend implementations change. Reuse the
existing workflow to maintain it; this recipe adds no second mapping registry
or automatic fused/split transformation engine.

## Standalone pinned artifact checker

`tools/consistency_checking/check_diffusion_consistency.py` checks a saved rollout and actor
log-probability comparison on CPU. It accepts a caller-defined manifest and an
independently hashed list of artifact bytes. It runs no model or rollout and
does not require the training framework to import for `--help`.

This is an artifact contract for a producer, not a trainer API. Collect the
two role records from the same run, policy, sample and trajectory. Each record
needs the actual raw scheduler timestep and the three forward-input hashes.
Collect a `Module.*` JSON dump with at least one nonempty module object for
each record, and save a dense (strided) CPU float32
`comparison.pt` containing exactly `actor_log_probs` and `rollout_log_probs`,
each shaped `(N, T)`. `N` is the number of paired keys; `T` is positive. The
manifest's comparison keys give the tensor row order explicitly.

## Input layout and command

All manifest and pin-list paths are relative to an input root. For a two-role,
one-pair example, `manifest.json` is:

```json
{
  "version": 1,
  "expected": {
    "run_id": "run-one",
    "policy_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "global_step": 1,
    "sde_type": "sde",
    "noise_level": 0.7,
    "timestep_atol": 0.0
  },
  "records": [
    {
      "run_id": "run-one", "policy_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "global_step": 1, "sample_id": "sample-0", "trajectory_index": 0,
      "branch": "cond", "timestep_unit": "raw_scheduler", "timestep": 812.5,
      "noise_level": 0.7, "sde_type": "sde", "role": "rollout", "phase": "rolloutgenerate",
      "hidden_states": {"sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "dtype": "torch.float32", "shape": [1, 2]},
      "prompt_embeds": {"sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "torch.float32", "shape": [1, 2]},
      "pooled_projections": {"sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "dtype": "torch.float32", "shape": [1, 2]},
      "dump_path": "dumps/rollout.json"
    },
    {
      "run_id": "run-one", "policy_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "global_step": 1, "sample_id": "sample-0", "trajectory_index": 0,
      "branch": "cond", "timestep_unit": "raw_scheduler", "timestep": 812.5,
      "noise_level": 0.7, "sde_type": "sde", "role": "actor", "phase": "actorlog_prob",
      "hidden_states": {"sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "dtype": "torch.float32", "shape": [1, 2]},
      "prompt_embeds": {"sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "torch.float32", "shape": [1, 2]},
      "pooled_projections": {"sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "dtype": "torch.float32", "shape": [1, 2]},
      "dump_path": "dumps/actor.json"
    }
  ],
  "comparison": {"path": "comparison.pt", "keys": [["sample-0", 0, "cond"]]},
  "atol": 0.00001,
  "rtol": 0.001
}
```

The example hashes are placeholders. A producer must put actual SHA256 values
in the records. The `expected` block declares one pairing context; it is
included in the pinned manifest. `atol` and `rtol` are required finite,
nonnegative thresholds chosen by the caller. The script does not set a model
specific tolerance or weight identity. The checker evaluates closeness in
float64 to avoid overflow of finite float32 differences; the comparison remains
`abs(actor - rollout) <= atol + rtol * abs(rollout)`.

Write `pins.sha256` with exactly one line for the manifest, comparison tensor,
and every unique referenced dump. Each line is a lowercase 64-character SHA256,
two spaces, a root-relative path, and a final newline. For the example:

```text
<manifest-sha256>  manifest.json
<comparison-sha256>  comparison.pt
<rollout-dump-sha256>  dumps/rollout.json
<actor-dump-sha256>  dumps/actor.json
```

Replace each angle-bracket placeholder with the actual lowercase digest.
Record the SHA256 of the **pin file itself** independently from the producer
artifacts. Supply that digest as `--pins-sha256`; computing it from the current
file immediately before checking gives no independent tamper evidence.

```bash
python tools/consistency_checking/check_diffusion_consistency.py \
  --root /absolute/saved-pair \
  --manifest manifest.json \
  --pins pins.sha256 \
  --pins-sha256 "$TRUSTED_PINS_SHA256" \
  --output /absolute/new-report.json
```

The report path must be fresh and outside the input root. A numerical mismatch
writes `status: numerical_mismatch` and exits nonzero. Missing or invalid bytes,
metadata, pairing, or tensor shape fail without an admitted report. The reader
checks all pinned bytes before `torch.load(..., weights_only=True)` and uses
those same verified bytes for JSON and tensor parsing. It accepts relocated
root folders because every input path is relative to the root.

## Collect a new public SD3.5 producer pair

The [public SD3.5 capture recipe](../../tools/consistency_checking/sd3/README.md)
provides source-byte inventory, CPU admission, separate real rollout and actor
processes, and external producer pins before this replay bridge. The raw
MSProbe captures cover the first sample's first transformer forward per role.
The saved batch and comparison tensors cover both prompts and two SDE steps.
The default public producer uses no rollout offload and targets an 80 GB GPU.
For the opt-in DiT-only layerwise rollout mode, set
`SD35_ROLLOUT_OFFLOAD=dit-layerwise` before both `admit` and `capture`; the
admission token binds the mode. See the recipe for the measured RTX 3090
memory usage and validation scope.

## Replay an existing SD3.5 producer pair

`tools/consistency_checking/sd3/replay.py` bridges the existing two-prompt,
two-step SD3.5 producer artifacts to the checker above. It does not collect a
new capture or run a model. Prepare `producer-pins.sha256` outside the capture
root, with root-relative entries
containing exactly these eight original files: `rollout/manifest.json`,
`rollout/actor_batch.pt`, `actor/comparison.json`, `actor/comparison.pt`, one
`record.json` under each role's `captures/<role>/<capture-id>/`, and the
corresponding `dump/step0/rank0/dump.json` for each record. Record the SHA256
of the pin list independently before replay; a digest freshly calculated from
the same untrusted capture provides no independent evidence.

For an already trusted capture, prepare the list once with the producer pin
utility. Put the output outside the capture root. Record the printed SHA256
digest independently from the producer artifacts and retain it for replay.

```bash
python tools/consistency_checking/sd3/pin_producer.py \
  --root /absolute/existing-capture \
  --output /absolute/producer-pins.sha256
```

```bash
python tools/consistency_checking/sd3/replay.py \
  --root /absolute/existing-capture \
  --pins /absolute/producer-pins.sha256 \
  --pins-sha256 "$TRUSTED_PRODUCER_PINS_SHA256" \
  --output /absolute/new-replay-output
```

The output directory must be fresh and outside the input root. It contains
`case/` with derived checker inputs, `report.json` from the public checker,
and `audit.json` with the overall replay status, full log-probability
comparison, source pin digest, and selected tensor row.
Original producer files are read only. Producer records use nested input
metadata; replay exposes those fields to the checker. The bridge binds the
captured sample to the producer's ordered `request_map`, verifies both saved
rollout tensor copies, and compares all four saved log probabilities. The
public checker validates the captured forward for row 0, step 0, so
`report.json` covers only that pair. A mismatch anywhere in the full comparison
makes replay exit nonzero, even if the captured pair passes.
The pin list uses paths relative to the supplied capture root, so the eight
original bytes can be copied to a differently named root. The pinned producer
metadata still names the original run path and request IDs; the bridge uses
those declarations to interpret legacy absolute dump paths. The recipe
requires the fixed two prompts, seeds 1234 and 1235, and tolerances `1e-5`
and `1e-3`. `audit.json` includes pinned producer identity and provenance;
these declarations are not fresh source, probe, or model-weight validation.
The replay bridge does not attest model weights independently of the pinned
producer declarations; use the public capture recipe to produce fresh bytes.

## Scope of evidence

The gate checks one actor and one rollout record for each key, exact
run/policy/step, conditional branch, SDE/noise, raw timestep gap, input
hashes, nonempty `Module.*` dumps, row-key bijection and declared numerical
tolerances. The hashes and policy are producer declarations bound to an
external pin-list digest. They do not independently attest actual model
weights, execution, or all internal tensors. The tool does not analyze layer
values or establish full training, transport, convergence or performance.

For collection and deeper analysis, see this repository's [profiling guide](profiler.md),
its [Ascend profiling section](profiler.md#7-ascend-npu-profiling-npu), and the
upstream [MSProbe](https://github.com/Ascend/msprobe) and
[MSAgent](https://github.com/Ascend/msagent) projects. Existing model-specific
GPU harnesses and historical measurements are outside this standalone script;
they are not validation evidence for this new reader until its own tests and
saved-artifact replay are run.
