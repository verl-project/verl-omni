# L3 Nightly Regression

These regressions run 20 training steps on local tiny-random weights with one
deterministic prompt repeated across the batch. Qwen-Image cases reuse an OCR
prompt. Qwen3-Omni and MiniCPM-o build a tiny checkpoint and text-only or image-audio prompts before training. Each job checks two things:

- Steps in `DEBUG_DUMP_STEPS` default to `1,2` and write debug dumps for
  precision comparison.
- Steps after `PERF_SKIP_STEPS` default to `2` and are used for timing,
  throughput, and memory metric comparison.

Shared hooks, metric collection, dump comparison, dataset generation, and the
v1 trainer entrypoint live in this directory. Each case directory only owns
that algorithm's Hydra overrides. The runner is `tests/nightly/run.py`.
Qwen-Image cases start the default `verl_omni.trainer.main_diffusion_v1`.
Qwen3-Omni and MiniCPM-o export
`NIGHTLY_TRAINER_MODULE=verl_omni.trainer.main_omni` so the same runner starts
the omni v1 entrypoint. 
The v0 trainer is not used.

## Supported models and algorithms

| Model | Case | Algorithm | Trainer | GPUs | Script |
| --- | --- | --- | --- | --- | --- |
| Qwen-Image | `qwen_image_flowgrpo` | FlowGRPO, OCR reward | `main_diffusion_v1` | 1 | `tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh` |
| Qwen-Image | `qwen_image_diffusion_nft` | DiffusionNFT, OCR reward, old-adapter rollout | `main_diffusion_v1` | 1 | `tests/nightly/qwen_image_diffusion_nft/run_qwen_image_diffusion_nft.sh` |
| Qwen-Image | `qwen_image_online_dpo` | Online DPO, OCR reward | `main_diffusion_v1` | 1 | `tests/nightly/qwen_image_online_dpo/run_qwen_image_online_dpo.sh` |
| Qwen3-Omni | `qwen3_omni_thinker_gspo` | GSPO LoRA, GSM8K rule reward | `main_omni` | 1 | `tests/nightly/qwen3_omni_thinker_gspo/run_qwen3_omni_thinker_gspo.sh` |
| MiniCPM-o 4.5 | `minicpmo_4_5_thinker_gspo` | GSPO LoRA, AVQA choice reward | `main_omni` | 1 | `tests/nightly/minicpmo_4_5_thinker_gspo/run_minicpmo_4_5_thinker_gspo.sh` |

Case-specific precision tensors and ordering notes are in each case README.
The workflow baseline check and run loop both read `ALL_L3_CASES` in
`.github/workflows/l3_nightly.yml`. Add a case there and to the `case` input
options. The script path is `tests/nightly/<case>/run_<case>.sh`.

## CI Status

This directory is wired to `.github/workflows/l3_nightly.yml`. The workflow runs
the cases sequentially in strict nightly mode every day at 22:00
Asia/Shanghai and can also be triggered manually.

Use these tests as a regression signal for numerical drift and performance
changes. They are not part of the fast pull-request CI loop.

Workflow trigger modes:

- Scheduled runs use `nightly` mode and require an existing baseline artifact.
- `workflow_dispatch` supports `mode=nightly` or `mode=baseline`, and a `case`
  choice of `all`, `qwen_image_flowgrpo`, `qwen_image_diffusion_nft`,
  `qwen_image_online_dpo`, `qwen3_omni_thinker_gspo`, or
  `minicpmo_4_5_thinker_gspo`. Scheduled runs and pull requests ignore `case`
  and run every case.
- To trigger a baseline run, open the workflow in GitHub Actions, choose
  **Run workflow**, and set `mode=baseline`. Set `case` to one case to refresh
  only that subdirectory.
- Adding the `L3-nightly-ci` label to a pull request triggers strict `nightly`
  mode for that PR.

## Requirements

The runner must have:

- 1 GPU by default, or set `NUM_GPUS` to match the runner.
- An installed `verl_omni` GPU environment with the rollout and training
  dependencies needed by the case. Omni cases also need the audio stack used
  by the nightly workflow (`librosa`, `soundfile`, `av`, `audioread`).
- Local tiny-random policy weights. Qwen-Image cases also need the OCR reward
  model. Omni cases build their own tiny policy under `POLICY_MODEL_PATH`.
- Enough local disk space for debug dumps, metrics, and logs under
  `OUTPUT_ROOT`.

Expected local model defaults:

- Qwen-Image policy: `~/models/tiny-random/Qwen-Image`
- Qwen-Image OCR reward: `~/models/tiny-random/qwen3-vl`
- Qwen3-Omni policy: `~/models/tiny-random/Qwen3-Omni`
- MiniCPM-o policy: `~/models/tiny-random/MiniCPM-o-4_5`

## Run

`BOOTSTRAP_MISSING_BASELINE=1` is the default. A first run with no baseline
copies the current artifacts into that case's `baseline/` directory and passes.
An existing baseline is compared and is not replaced. Set
`BOOTSTRAP_MISSING_BASELINE=0` for a strict comparison.

Diffusion case example:

```bash
NUM_GPUS=1 \
MODEL_PATH=/path/to/tiny-random/Qwen-Image \
REWARD_MODEL_PATH=/path/to/tiny-random/qwen3-vl \
OUTPUT_ROOT=/path/to/debug_dumps \
bash tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh
```

Omni case example:

```bash
NUM_GPUS=1 \
POLICY_MODEL_PATH=/path/to/tiny-random/Qwen3-Omni \
OUTPUT_ROOT=/path/to/debug_dumps \
bash tests/nightly/qwen3_omni_thinker_gspo/run_qwen3_omni_thinker_gspo.sh
```

Strict comparison adds `BOOTSTRAP_MISSING_BASELINE=0` to any of those case
commands. Replace the script path with other case scripts when needed.

## Baselines

The default layout gives every case its own subdirectory:

```text
outputs/debug_dumps/
|-- baseline/
|   |-- qwen_image_flowgrpo/
|   |-- qwen_image_diffusion_nft/
|   |-- qwen_image_online_dpo/
|   |-- qwen3_omni_thinker_gspo/
|   `-- minicpmo_4_5_thinker_gspo/
|-- current/
|   |-- qwen_image_flowgrpo/
|   |-- qwen_image_diffusion_nft/
|   |-- qwen_image_online_dpo/
|   |-- qwen3_omni_thinker_gspo/
|   `-- minicpmo_4_5_thinker_gspo/
`-- logs/
    |-- qwen_image_flowgrpo/
    |-- qwen_image_diffusion_nft/
    |-- qwen_image_online_dpo/
    |-- qwen3_omni_thinker_gspo/
    `-- minicpmo_4_5_thinker_gspo/
```

GitHub Actions stores every case inside one shared `l3-nightly-baseline`
artifact.

`L3_REFRESH_BASELINE` defaults to `0`. Set it to `1` to delete only this case's
baseline subdirectory after training succeeds, then copy `current/` in as the
new baseline. The run prints `BASELINE BOOTSTRAPPED` and does not compare
against the deleted baseline. Sibling case directories stay in place. The
workflow sets this in `mode=baseline` after downloading the current unified
baseline artifact. The `case` input limits which scripts run, so a single-case
baseline run leaves the other downloaded directories unchanged and uploads the
full tree. If no artifact exists yet, a single-case run uploads only that case;
a later nightly fails before the GPU runner when a selected case has no
baseline artifacts. A scheduled nightly keeps `L3_REFRESH_BASELINE` at `0` with
`BOOTSTRAP_MISSING_BASELINE=0`, so a missing baseline, a missing dump, or a
metric regression fails the job.

In GitHub Actions, baseline mode uploads:

- `l3-nightly-baseline`, retained for 90 days.
- `l3-nightly-reports-<run_id>`, retained for 30 days.

Nightly mode downloads the latest non-expired `l3-nightly-baseline` artifact for
the configured `baseline_branch` input, then runs strict comparison. If that
artifact is missing, or the selected case has no `payload.pt` and `metrics.json`
inside it, the job fails before the GPU runner is created. It uploads
current dumps as `l3-nightly-current-<run_id>` and always uploads logs and
reports as `l3-nightly-reports-<run_id>`.

To refresh the production baseline:

1. Review the expected change and choose the branch whose baseline should be
   refreshed.
2. Trigger `l3_nightly` manually with `mode=baseline`. Choose one case to
   refresh only that subdirectory, or leave `case=all`. A single-case run with
   an existing artifact keeps the other cases. With no artifact yet, it uploads
   only the selected case.
3. Inspect the uploaded reports and baseline artifact before relying on later
   nightly runs. Publish that artifact only from a green run: a failed baseline
   job can refresh the cases that finished and leave later cases unchanged.

Each case writes these comparison outputs under its `current/<case>/` directory:

- `metrics.json` contains aggregated timing, throughput, and memory metrics.
- `dump_compare.json` contains precision comparison results.
- `metrics.jsonl` contains step-level debug metrics from the run.
- `logs/<case>.log` contains the console log.

## Comparison Details

Performance comparison reads step-level metrics from `metrics.jsonl`, or falls
back to parsing the console log. It compares only steps after `PERF_SKIP_STEPS`,
which defaults to `2`, and compares the mean value for each comparable metric.
A key is compared when the baseline summary contains it. A baseline key missing
from the current run fails the comparison.

Performance tolerance:

- `PERF_THRESHOLD`, default `0.15`, meaning a 15% relative-change threshold.
- `timing_s/update_weights` uses a fixed 20% threshold. Weight synchronization is
  sensitive to vLLM-Omni, LoRA, IPC, free-cache, and layered summon details, and
  is usually a small part of total step time, so this wider tolerance avoids
  noisy failures while still catching material regressions.

Performance metrics:

- Lower is better: `perf/time_per_step`, `timing_s/gen`,
  `timing_s/old_log_prob`, `timing_s/prepare_actor_batch`, `timing_s/ref`,
  `timing_s/reward`, `timing_s/update_actor`, and `timing_s/update_weights`.
- Higher is better: `perf/mfu/actor`, `perf/mfu/actor_infer`, and
  `perf/throughput`.

FlowGRPO reports `timing_s/old_log_prob` and `perf/mfu/actor_infer`. DiffusionNFT
and online DPO do not recompute old log-probs, so those two keys are absent.
`perf/mfu/actor` is compared for the Qwen-Image cases. The GSPO omni cases also
collect language-model MFU, but `collect_metrics.py` warns that those
model/algorithm pairs are outside the diffusion MFU allowlist.

A lower-is-better metric fails when current is more than the metric threshold
above baseline. A higher-is-better metric fails when current is more than the
metric threshold below baseline. Other collected metrics, including
`perf/total_num_images`, `timing_s/step`, and derived `timing_per_image_ms/*`
values, are kept in the report but are not compared against the baseline.

Precision comparison recursively compares all tensors in matching `payload.pt`
debug dumps under `current/<case>/` and `baseline/<case>/`. Missing files,
missing tensors, and shape mismatches fail the comparison.

Qwen-Image dumps cover driver-forward, actor-forward, and LoRA-gradient
payloads for `DEBUG_DUMP_STEPS`, which defaults to `1,2`. The nightly dataset
writes `extra_info.repeat_index` on each repeated prompt. Online DPO and
DiffusionNFT draw noise or timesteps in actor-batch order. That batch arrives
in per-run `uuid4` uid order, so the nightly hook sorts it onto `repeat_index`
before `prepare_actor_batch`. Dump comparison realigns rows by the same key.
Online DPO keeps the top and bottom reward of each prompt. The nightly random
reward model returns stable scores, so chosen and rejected stay in the same
order across runs. FlowGRPO binds `log_probs`, latents, and timesteps at
rollout; comparison-time `repeat_index` alignment is enough for that case.
Missing or mismatched keys fall back to raw file order with a
`row order not realigned` warning. Normal training does not load the hook and
stays in uid order.

For Qwen-Image, actor-forward tensors are every tensor returned by the
diffusion FSDP engine forward/backward batch output. LoRA-gradient tensors are
every available LoRA parameter gradient under `gradients.<parameter_name>`.
Optional actor-forward-step tensors are included only when
`DEBUG_DUMP_FORWARD_STEPS=1`. Named batch tensors differ by algorithm; see the
case README.

Qwen3-Omni and MiniCPM-o dump `actor_update_batch` and `lora_gradients`. The
actor mini-batch includes `old_log_probs`, `ref_log_prob`, and
`rollout_log_probs` when present. Rollout is greedy so the repeated prompt
keeps stable token ids; a token-id change still fails when it moves those log
probs or the LoRA gradients. Case READMEs list the exact payload fields.

Dumped but never failing for Qwen-Image:

- `responses` is rollout pixel output: uint8 images, and rollout sampling is
  not bit-exact, so pixel-level differences are expected.
- `advantages`, `sample_level_scores`, `sample_level_rewards`, `reward_prob`,
  and `returns` are derived from the rollout image through the OCR reward.
  vLLM-Omni request packing (`max_num_seqs` > 1) changes the pre-window ODE
  batch shape, which flips individual OCR scores by ~1/255 pixel drift.

For the GSPO omni cases, `batch.responses` and `batch.advantages` stay
informational for the same shared comparator reason. Rule or choice reward
scores are compared.

These informational tensors are measured and reported but cannot fail the run.
They carry `informational: true` in `dump_compare.json` and are listed under
`informational_tensors`; the log prints
`Informational tensors (measured, never fail)`. A shape mismatch on these
tensors still fails.

Precision thresholds:

- `PRECISION_ATOL`, default `1e-3`, elementwise absolute-error threshold used
  for `frac_abs_over_atol`.
- `PRECISION_MEAN_ATOL`, default `1e-4`, maximum mean absolute error.
- `PRECISION_RMSE_ATOL`, default `1e-3`, maximum RMSE.
- `PRECISION_P99_ATOL`, default `2e-3`, maximum 99th percentile absolute error.
- `PRECISION_MAX_FRAC_ABS_OVER_ATOL`, default `2e-2`, maximum fraction of
  elements whose absolute error exceeds `PRECISION_ATOL`.
- `PRECISION_MIN_COS_SIM`, default `0.99`, minimum cosine similarity.

Each tensor report includes `numel`, `mean_abs_err`, `rmse`, `p99_abs_err`,
`frac_abs_over_atol`, and `cos_sim`. A tensor fails when any aggregate metric
exceeds its threshold or cosine similarity falls below its floor, except for the
informational tensors listed above.

## Failure Triage

When the job fails:

1. Check `logs/<case>.log` first for environment, model-loading, Ray, CUDA, or
   OOM failures.
2. Check `dump_compare.json` for tensor-level precision drift on the configured
   `DEBUG_DUMP_STEPS`.
3. Check `metrics.json` for post-warmup performance regressions after
   `PERF_SKIP_STEPS`.
4. If the change is expected, rerun once on the same fixed runner, review the
   new `current/` artifacts, and replace `baseline/<case>/` only after
   confirming the drift is intentional.
