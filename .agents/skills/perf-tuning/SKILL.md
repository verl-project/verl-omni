---
name: perf-tuning
description: "Turn a verl-omni speed-up request or a performance-sensitive Python/PyTorch change into a measured, regression-safe workflow: profile before changing code, apply the repo's hot-path practices, then compare a matched baseline vs candidate on both the affected stage and end-to-end step time. Use when asked to make training faster, when reviewing a hot-path change, or when judging whether an optimization actually helped."
---

# Perf-tuning a Python hot path

This skill is a workflow router, not a reference. The authoritative details
live elsewhere — link to them, do not copy them:

- [code-style rule](../../rules/code-style.md) § Performance Patterns — the
  Python/PyTorch hot-path conventions checklist.
- [tuning guide](../../../docs/perf/tuning_guide.md) — stage-by-stage **config**
  tuning (rollout batching, actor MFU, async reward, OOM triage).
- [profile skill](../profile/SKILL.md) + `docs/perf/profiler.md` — trace
  collection for the trainer/mode actually being run.
- [self-review skill](../code-review/SKILL.md) — the Performance review
  category that consumes your evidence at PR time.

## Step 0 — Classify the request

| The request | Route |
| --- | --- |
| "make training faster", bottleneck unknown | [profile](../profile/SKILL.md) first — record the **measured** dominant phase; never guess it from code shape (a slow-looking rollout is often reward-bound) |
| faster via knobs only (batching, offload, pools, poll cadence) | [tuning guide](../../../docs/perf/tuning_guide.md); no code change, no microbenchmark needed |
| change Python/PyTorch on a hot path (trainer, worker, agent loop, reward, rollout adapter, TransferQueue data path) | this skill |

A "hot path" is anything on the per-step critical path of training:
generation, reward scoring, log-prob / actor update, weight sync, trajectory
and TransferQueue traffic. Identify the phase's owner process before
changing anything — the same optimization means different code in the trainer
controller vs a rollout server.

## Step 1 — Check the change against the hot-path conventions

Before writing the candidate, walk the [code-style](../../rules/code-style.md)
§ Performance Patterns list against the touched code. The recurring verl-omni
risks, each with its cheap fix:

- **Host/device sync in hot loops** — `.item()`, `.cpu()`, `.tolist()`,
  `print(tensor)` per element/per frame. Keep them at true boundaries
  (logging, serialization); hoist or batch otherwise.
- **Python per-element loops over tensors** — vectorize, preserving sample
  boundaries, ordering, and loss normalization; a batched op that reorders
  rows silently corrupts training.
- **Repeated copies / per-item serialization** — one batched
  transfer/`select_fields` read instead of N roundtrips; the TransferQueue
  dump read is field-projected for exactly this reason (see the tuning
  guide's TransferQueue section).
- **Implicit dtype/device** — set both explicitly; implicit promotion hides a
  silent copy or a dtype mismatch that only surfaces numerically.
- **Unmeasured caches / `empty_cache()` / alternate kernels** — do not add
  them speculatively; measure first (Step 2), and give any long-lived cache a
  bound or an owner.

## Step 2 — Matched baseline vs candidate measurement

The protocol that makes the comparison meaningful:

- **Same everything except the change**: source revision (baseline = the
  candidate's commit minus only this change), workload shape, hardware,
  topology, warmup policy, and relevant runtime settings (batch sizes, engine
  concurrency, thread caps). Any difference invalidates the comparison.
- **Steady state, repeated**: discard warmup steps, take repeated
  measurements of the same window, and report the **distribution** (e.g.
  min/median/max), not the single best step. Profiling skill Step 2's
  warmup-then-record pattern applies to timing runs too.
- **Both levels, always**: the targeted phase **and** the full training step.
  A microbenchmark or stage-local win alone must not be reported as a
  training speedup — extra synchronization, copies, allocations, or RPC
  traffic elsewhere in the step can eat it (and has).
- **Memory and correctness alongside**: peak memory at the affected phase,
  plus the applicable numerical evidence (loss/reward curves unchanged,
  log-prob or debug-dump equivalence, CPU test suite). Throughput gained by
  wrong numbers is not gained.

## Step 3 — Report in this shape

Hypothesis → baseline → change (what and why this fixes the measured
bottleneck) → exact commands/config for both runs → hardware + workload →
phase measurement → full-step measurement → memory → correctness evidence →
limitations → keep/revert recommendation. If a field is unknown, say so
rather than filling it with an estimate.

Worked in-repo references: #653 with its follow-up #678 (TransferQueue
put/read overhead — batched per-group writes, field-projected dump reads,
poll cadence; #678 reports the matched gen/step timings) and the tuning
guide's CPU-thread cap (a config-level fix that reports both stage and
full-step numbers).

## Boundaries

Not a campaign controller or auto-tuner — experiment campaigns belong to the
experiment-lifecycle work (#810) when it lands; this skill covers one
hypothesis at a time. Config-knob tuning without code changes stays in the
tuning guide. At review time, hand the Step 3 report to the self-review
skill's Performance category instead of re-arguing the numbers.

<!--
MAINTAINER GUIDE — keep this skill a workflow router: the conventions live
in .agents/rules/code-style.md, the knobs in docs/perf/tuning_guide.md, the
tools in the profile skill. Recheck the referenced sections and PR examples
when the perf docs or the experiment-lifecycle skill (#810) land.
-->
