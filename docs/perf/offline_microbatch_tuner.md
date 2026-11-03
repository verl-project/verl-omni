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

## Fixed real SD3.5 Medium worker

`scripts/sd35_microbatch_worker.py` provides a bounded **actor update component** recipe using the real `PPODiffusersFSDPEngine`, `BaseEngine.train_batch`, and `diffusion_loss`. It measures the `train_mode` / `train_batch` / learning-rate-scheduler-step boundary after one restored warmup; it does not measure rollout, model loading, end-to-end training throughput, or convergence. This is not a dynamic memory-budget search.

Supply a local SD3.5 Medium snapshot with `model_index.json`, `transformer/config.json`, `scheduler/scheduler_config.json`, and a single `transformer/diffusion_pytorch_model.safetensors`. Supply an independently captured **real actor batch** as a `torch.save` plain dictionary of six finite CPU fp32 tensors: `all_latents` `[2,3,16,32,32]`, `all_timesteps` `[2,2]`, `prompt_embeds` `[2,L,4096]` with `L>0`, `pooled_prompt_embeds` `[2,2048]`, `old_log_probs` `[2,2]`, and nonzero `advantages` `[2,2]`. The worker does not download weights or synthesize rollout data. Capture and inspect these inputs on the authorized host before using them.

First emit the byte-bound identity, then prepare the exact fixed plan. The identity hashes the captured file, the four listed model files, the installed upstream `BaseEngine` file, the local engine/loss/adapter files, and this worker file. Run from the intended source checkout with matching imports. Identity must be regenerated if any of those bytes change.

```bash
python scripts/sd35_microbatch_worker.py --mode identity --model /local/sd35-medium --batch /local/actor_batch.pt --result /tmp/sd35-workload.json
```

Save `/tmp/sd35-plan.json` as:

```json
{"effective_batch":2,"candidates":[2,1],"baseline":2,"max_trials":2,"steps":2,"atol":1e-5,"rtol":1e-3}
```

CPU admission may be run before GPU search with a request matching the identity and either candidate. It checks the scheduler, adapter, `TensorDict`, and `train_batch` signature without loading model weights or initializing CUDA. The standalone `admit` mode also accepts no request and then checks microbatch 1.
Admission requires the captured `all_timesteps` to equal the configured scheduler's first two CPU fp32 timesteps in both batch rows, exactly. On a CPU host with the real local cache and captured batch, set `SD35_MICROBATCH_TEST_MODEL` and `SD35_MICROBATCH_TEST_BATCH` to those paths and run `pytest tests/utils/test_sd35_microbatch_worker_on_cpu.py -q`; the opt-in integration test checks both microbatch candidates and a mismatched timestep. Missing variables skip that test in a portable developer suite. A real-admission acceptance run must supply both variables and report zero skips.

```bash
python scripts/sd35_microbatch_worker.py --mode admit --model /local/sd35-medium --batch /local/actor_batch.pt --result /tmp/sd35-admission.json
python scripts/tune_microbatch.py --plan /tmp/sd35-plan.json --workload /tmp/sd35-workload.json --output /new/sd35-search -- python scripts/sd35_microbatch_worker.py --mode run --model /local/sd35-medium --batch /local/actor_batch.pt
```

Run the search in a runtime with exactly one visible CUDA device and world size one. The worker owns a one-rank NCCL group and destroys it on exit. Each fresh child starts from the same seeded LoRA, optimizer, scheduler, and random state; the warmup update is restored before two timed updates. It retains gradient references inside the engine train context and materializes full sorted gradient/parameter witnesses after timing. A failure or OOM cannot produce an `ok` result. Each success writes a separate diagnostics sidecar before its exclusive protocol result. Inspect `aggregate.json`: require **two valid trials** and a nonnull `recommendation` before using the comparison. CPU unit tests only cover contract seams, not real SD3.5 execution.
