# Check pinned diffusion consistency artifacts

Last updated: 11/09/2026

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

## Collect a new public SD3.5 producer pair

The [public SD3.5 capture recipe](../../examples/flowgrpo_trainer/sd35/consistency/README.md)
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

`scripts/replay_sd35_consistency.py` bridges the existing two-prompt,
two-step SD3.5 producer artifacts to the checker above. It does not collect a
new capture or run a model. Prepare `producer-pins.sha256` outside the capture
root, with root-relative entries
containing exactly these eight original files: `rollout/manifest.json`,
`rollout/actor_batch.pt`, `actor/comparison.json`, `actor/comparison.pt`, one
`record.json` under each role's `captures/<role>/<capture-id>/`, and the
corresponding `dump/step0/rank0/dump.json` for each record. Record the SHA256
of the pin list independently before replay; a digest freshly calculated from
the same untrusted capture provides no independent evidence.

For an already trusted capture, prepare the list once with Python's standard
library. Set `CAPTURE_ROOT` and `PIN_FILE` to absolute paths; put `PIN_FILE`
outside `CAPTURE_ROOT`. Keep the resulting digest in an independent record.

```python
import hashlib
import os
from pathlib import Path

root = Path(os.environ["CAPTURE_ROOT"])
pin_file = Path(os.environ["PIN_FILE"])
names = ["rollout/manifest.json", "rollout/actor_batch.pt",
         "actor/comparison.json", "actor/comparison.pt"]
for role in ("rollout", "actor"):
    records = list((root / role / "captures" / role).glob("*/record.json"))
    assert len(records) == 1
    names += [str(records[0].relative_to(root)),
              str((records[0].parent / "dump/step0/rank0/dump.json").relative_to(root))]
data = "".join(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
               for name in names).encode()
with pin_file.open("xb") as stream:
    stream.write(data)
print(hashlib.sha256(data).hexdigest())
```

```bash
python scripts/replay_sd35_consistency.py \
  --root /absolute/existing-capture \
  --pins /absolute/producer-pins.sha256 \
  --pins-sha256 "$TRUSTED_PRODUCER_PINS_SHA256" \
  --output /absolute/new-replay-output
```

The output directory must be fresh and outside the input root. It contains
`case/` with derived checker inputs, `report.json` from the public checker,
and `audit.json` describing the source pin digest and selected tensor row.
Original producer files are read only. Legacy nested input records and newer
flat records are normalized to the same checker record. The bridge binds the
captured sample to the producer's ordered `request_map`, verifies both saved
rollout tensor copies, then selects row 0 and step 0. Because the producer
saved only the first forward for each role, the report applies only to that
captured sample and step, even though the original tensors are 2 by 2.
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
