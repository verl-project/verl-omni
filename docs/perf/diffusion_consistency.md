# Check pinned diffusion consistency artifacts

Last updated: 10/31/2026

`scripts/check_diffusion_consistency.py` checks a saved rollout and actor
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
python scripts/check_diffusion_consistency.py \
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
