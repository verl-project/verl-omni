---
name: profile
description: "Route a verl-omni performance investigation to the right tool and capture a usable trace. Use when profiling FlowGRPO / diffusion training or rollout — choosing between nsys, torch.profiler, torch_memory snapshots, MFU comparison, or RL-Insight dashboards, and profiling one lightweight step instead of a full run."
---

# Profile a run

`docs/perf/profiler.md` owns the config surface (`global_profiler` +
per-role `actor_rollout_ref.{actor,ref,rollout}.profiler`), the six copy-paste
recipes, and the lightweight-footprint recipe. Open it — do not work from a
remembered procedure. This skill routes you to the right tool and adds the
cross-cutting decisions the guide leaves implicit.

## Step 0 — Pick the tool by the question

| The question you are answering | Tool | Where it is documented |
| --- | --- | --- |
| Where does the step's wall-clock go? (phase overlap, Python control flow, rank straggler) | `nsys` | `docs/perf/profiler.md` recipes 4, 4a |
| Which ops/kernels dominate, and CPU vs CUDA? | `torch` (`torch.profiler`) | `docs/perf/profiler.md` recipes 1, 2 |
| What is holding GPU memory / who OOMs? | `torch_memory` | `docs/perf/profiler.md` recipe 3 |
| Is config B more compute-efficient than A? | MFU (no profiler) | `docs/perf/diffusion_mfu.md` — read `perf/mfu/actor`, **relative** only |
| Live dashboards across replicas / TransferQueue during a long run? | RL-Insight | `docs/start/rl_insight.md` |

A profiler answers "where is the time/memory in this step". MFU answers "how
efficient is this config vs another on the same setup" — it is a metric, not a
trace, and it over-estimates LoRA (it counts the full DiT forward+backward).

## Step 1 — Profile ONE lightweight step, never a full run

A full FlowGRPO step trace is hundreds of MB and slow to open. Every `examples/`
recipe forwards `"$@"` to the same `diffusion_trainer` config and Hydra resolves
duplicates last-wins, so **append** footprint overrides instead of editing the
script — shrink `rollout.n`, `pipeline.num_inference_steps`, resolution, and
batch (see the guide's lightweight recipe; it cut a step 616 s → 70 s).

Always pin these, or profiling is silently skipped:

```bash
trainer.total_training_steps=1 trainer.save_freq=-1 trainer.test_freq=-1 \
trainer.resume_mode=disable global_profiler.steps=[1]
```

The last step force-triggers save/validation when `save_freq`/`test_freq` > 0,
and a leftover checkpoint auto-resumes past the profiled step. For continuous
`nsys` captures, step 2 is the steady-state sample (step 1 carries profiler
startup, the last step closes the window).

## Step 2 — Enable the profiler on the process that owns the phase

Each phase runs in a different process; enabling the wrong `*.profiler` yields an
empty trace:

- actor train / backward → `actor_rollout_ref.actor.profiler`
- generation → `actor_rollout_ref.rollout.profiler` (a separate vLLM-Omni server;
  `tool_config.torch.discrete=True` is **required** — it rejects continuous mode)
- reward model → `reward.reward_model.rollout.profiler`
- ref log-prob → `actor_rollout_ref.ref.profiler`

These keys already exist in the composed config, so override with plain
`key=value` — a `+key=value` append fails with "An item is already at ...".

## Gotchas (each has bitten a real run)

- **V1 trainer**: `nsys` step-scoped controller capture
  (`capture-range=cudaProfilerApi`) is not supported by
  `verl_omni.trainer.main_diffusion_v1`; only `main_diffusion` drives the
  step-based start/stop lifecycle.
- **`nsys` output path**: `*.nsys-rep` files land under
  `/tmp/ray/session_latest/logs/nsight/` (fixed by Ray), not `save_path`; only
  `torch` / `torch_memory` traces honor `global_profiler.save_path`.
- **Report hygiene**: reports may embed env vars such as `HF_TOKEN` unless
  `discard-environment` is set — scrub before sharing.
- **Do not hand-roll** `torch.profiler` / timers inside adapter or pipeline code.
  Workers are wrapped with `verl.utils.profiler.DistProfiler` and driven around
  each profiled step; an ad-hoc timer is fine for a throwaway local check but
  must never reach a PR.

## Further reading

- `docs/perf/profiler.md` — authoritative config surface and recipes.
- `docs/perf/diffusion_mfu.md` — MFU reporting and adding an estimator.
- `docs/start/rl_insight.md` — online observability dashboards.
