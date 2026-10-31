# Offline microbatch tuner

Last updated: 10/28/2026

`scripts/tune_microbatch.py` compares a finite list of microbatch sizes for one **offline, trusted, single-process worker**. It does not change an active trainer, choose a model, construct a batch, launch Ray or distributed jobs, or perform a warmup itself. The worker owns model setup, any warmup and restore, measured updates, and the numerical witnesses. Use the same frozen workload, initial trainable state, optimizer state and random state for every candidate.

## Plan and workload identity

Save the plan and identity as JSON files. For example:

```json
{"effective_batch":8,"candidates":[1,2,4],"baseline":2,"max_trials":3,"steps":2,"atol":1e-5,"rtol":1e-3}
```

```json
{"data_sha256":"<captured-batch-sha256>","source_revision":"<reviewed-revision>","model_revision":"<reviewed-model-revision>","precision":"fp32"}
```

The identity can contain any nonempty string keys with string values. Record exact source, model, data, precision and policy identifiers relevant to the worker; the controller checks equality and hashes the canonical JSON but cannot attest that the worker loaded those bytes. A worker must verify its own inputs and report the same identity. The plan accepts distinct positive candidate divisors of `effective_batch`, a baseline in that list, positive `max_trials` and `steps`, and finite nonnegative `atol`/`rtol`. The baseline runs first. At most `max_trials` candidates are launched; untried candidates appear in `budget_excluded`. This is a **trial/update budget**, not a wall-clock timeout.

## Run

Choose a trusted worker command that accepts `--request PATH --result PATH`, then provide a new output path:

```bash
python scripts/tune_microbatch.py \
  --plan /path/to/plan.json \
  --workload /path/to/workload.json \
  --output /path/to/new-search \
  -- python /path/to/trusted_worker.py --worker-specific-option value
```

The controller creates the output directory exclusively and starts one fresh direct child per candidate. It appends `--request` and `--result` to the worker command. Each trial has `request.json`, `result.json` when written by the worker, `evaluation.json`, `stdout.txt` and `stderr.txt`. `source.json` records the plan, workload, workload digest and worker argv; `aggregate.json` records trial verdicts, excluded budget and recommendation. Existing output paths fail. The controller does not retry a trial. When interrupted, it settles only its owned **direct child**; it does not claim to stop descendants or distributed work. Use a worker whose runtime is already controlled by its caller; the controller adds no arbitrary workload timeout.

## Worker protocol

The request contains exactly these fields:

| Field | Meaning |
| --- | --- |
| `trial_id` | Zero-based execution index. |
| `microbatch` | Candidate size; baseline is index zero. |
| `accumulation` | `effective_batch // microbatch`. |
| `steps` | Number of measured updates expected from the worker. |
| `effective_batch` | Fixed effective batch from the plan. |
| `workload` | Canonical identity mapping supplied by the caller. |
| `workload_digest` | SHA256 of sorted compact JSON workload bytes. |

The result contains exactly `trial_id`, `microbatch`, `accumulation`, `steps`, `workload`, `workload_digest`, `status`, `reason`, `initial_state_digest`, `update_seconds`, and `witnesses`. The first six fields must equal the request. An `ok` result has an empty `reason`, nonempty initial-state digest, `steps` finite positive timing values, and `steps` witness maps. Each witness has nonempty finite numeric vectors `loss`, `grads` and `params`; signed values are valid. The vectors must represent the same ordered elements in every trial. The controller compares each update's loss, gradients and updated parameters to the baseline elementwise with `math.isclose(abs_tol=atol, rel_tol=rtol)`, and requires the same initial-state digest. An `oom` or `failure` result has a string reason, null initial digest, and empty timings/witnesses.

The worker should perform one warmup update, restore its initial trainable/optimizer/scheduler/random state, then time each measured update and form witnesses after the timed boundary if that is its declared timing contract. The controller enforces the submitted result shape and numerical alignment; it does not verify warmup, restoration, model provenance, timing boundaries or accelerator memory. Keep those in the worker's own diagnostics and validation.

Malformed JSON, missing or extra result fields, nonfinite values, nonzero exits, declared OOM/failure, mismatched workload or initial state, and numerical differences exclude a trial. A failed baseline prevents later candidates from being recommended. The CLI can still exit zero after recording excluded trials, including when `recommendation` is null: **inspect `aggregate.json` and require the intended valid-trial count before acting on a recommendation**. A valid result is an exploratory component comparison for the declared worker; it is not a full training throughput or convergence claim.

The CPU test at `tests/utils/test_offline_microbatch_tuner_on_cpu.py` exercises the protocol with a stdlib fake worker. Its results are fixture proof of controller behavior, not GPU or model evidence.
