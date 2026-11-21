# Phases

Five phases, one branch, one reloop. A phase is a **stage of the loop**, not a
category of change: `OPTIMIZATION` is where every hypothesis lives, whether it is a
config knob, a code patch or a delegated operator replacement. Keeping the phases few
matters more than labelling a change correctly — the round already says what it changed.

```
PRELUDE ─► ENABLEMENT* ─► OPTIMIZATION ⟲ ─► SWEEP ─► CLOSE
                             ▲              │
                             └─ new cycle ──┘

* ENABLEMENT runs only when the baseline does not run end to end.
⟲ OPTIMIZATION repeats once per hypothesis, until no idea is worth a round.
```

Every move, in one table — this is the whole state machine:

| You just finished | Go to | When |
| --- | --- | --- |
| — | `PRELUDE` | a session starts here |
| `PRELUDE` | `ENABLEMENT` | the baseline does not run end to end |
| `PRELUDE` | `OPTIMIZATION` | the baseline ran — the normal case |
| `PRELUDE` | `CLOSE` | the baseline cannot be measured and the user stops |
| `ENABLEMENT` | `OPTIMIZATION` | it runs now |
| `ENABLEMENT` | `CLOSE` | it cannot be made to run |
| one `OPTIMIZATION` round | `OPTIMIZATION` | another hypothesis is worth a round (`start` alone is enough) |
| `OPTIMIZATION` | `SWEEP` | no hypothesis is left, or three rounds without progress |
| `SWEEP` | `OPTIMIZATION` | budget remains and headroom exceeds the noise floor — **new cycle** |
| `SWEEP` | `CLOSE` | budget spent, target reached, or converged |
| `CLOSE` | — | terminal; nothing is appended afterwards |

`session.py phase --to <name>` records a move; the adapter rejects a jump it cannot
justify (for example `prelude → sweep`). `SWEEP → OPTIMIZATION` starts a new cycle: the
counter increments, round ids become `c02-…`, and the current stack carries over.

## PRELUDE — anchor the baseline

- **Do:** run the user's launcher unmodified, `repeats` times (one round each), record
  the objective and take one lightweight profile to see where the step time goes.
- **You may change:** nothing. Rounds here carry `change: none`.
- **Exit:** an anchor exists (median of the repeats) and you know which `timing_s/*`
  bucket dominates.
- **Common mistake:** anchoring from one run. Two or three repeats are what make the
  noise floor meaningful; a single run cannot separate a 3 % gain from drift.

## ENABLEMENT — only if the baseline does not run

- **Enter when:** the baseline fails on data, environment, launcher, memory or a
  missing dependency — the training equivalent of "the model does not serve".
- **You may change:** whatever makes it run — a path, a smaller micro-batch, a device
  flag, a missing dataset.
- **Grading differs here:** a round is `keep` when it runs and the accuracy floor holds,
  not when it is faster. A speed number from a shrunken workload is not a gain.
- **Exit:** the launcher completes end to end, or the session closes with `stopped`.

## OPTIMIZATION — the loop body

Every hypothesis goes here. Two familiar styles, both just rounds:

- **Config rounds** — follow the repo's order in `docs/perf/tuning_guide.md` (§1 layout,
  §2 rollout throughput, §3 actor memory and precision, §4 symptom checklist) and change
  the whitelisted knob that addresses the dominant bucket.
- **Code rounds** — hand the hot spot to a specialized skill (`ascend-fusion-op-replace`,
  a profiler analysis, `train-infer-consistency`), apply the diff or finding, then
  **re-measure end to end yourself**. The change must be revertible (`git apply -R` or
  equivalent) and its own numerical or unit check must pass first.

Optional convention, not a rule: prefix `change` with `config:` or `code:` when it makes
the report easier to read (`config:max_num_seqs=128`, `code:experts-fusion`).

- **Exit:** no remaining idea is likely to beat the noise floor, or three consecutive
  rounds fail to beat the incumbent, or the budget is spent.
- **Common mistakes:** two changes in one round (the gain is unattributable); trusting a
  claimed speedup (an agent's prediction, a kernel author's number and a microbenchmark
  are evidence, never a `keep` — write the claim next to the measured number in
  `notes.md`, since that gap is what the KB learns from).

## SWEEP — validate the stack, do not extend it

- **Do:** re-measure the accumulated stack at 2–3 operating points the user cares about
  (for example `train_batch_size`, `max_response_length`, rollout concurrency). One
  round per point, `change: none`.
- **You may change:** nothing. A regression here sends you back to `OPTIMIZATION` with
  the offending stack entry under suspicion.
- **Skip:** when the validated gain has not moved since the previous sweep — record no
  round and reloop or close directly.
- **Exit:** every point measured, or one point proves the gain does not generalise.

## CLOSE — stop, report, learn

- **Do:** `session.py close --reason <budget|converged|target|plateau|stopped> --detail "…"`,
  then write `kb/recipes.jsonl` (winning stack, workload, numbers) and `kb/lessons.md`
  (what failed and why).
- **Stop reasons:** `budget` (rounds or wall clock), `converged` (nothing left worth a
  round), `target` (the mission target reached by a kept stack), `plateau` (three rounds
  without progress), `stopped` (the user, or a hard blocker).
- **Exit:** `close.md` plus the KB rows exist and `session_breakdown.json` is current.

## How a verdict is decided

| Situation | Verdict |
|---|---|
| No measurement, or the run was truncated | no verdict — fix the run, then measure |
| The run crashed | `crash` + `why: crash_*`, then retry once at smaller footprint or move on |
| Better than the incumbent by **more than the noise floor**, and the accuracy floor holds | `keep` |
| Differs by less than the noise floor | `revert` + `why: within_noise` — the stack must not accumulate luck |
| Worse | `revert` + `why: regressed` |
| A user constraint is violated (reward, loss, consistency) | `revert` + `why: constraint`, whatever the speed number says |

Noise floor = `max(1 % of the anchor, 2 × sd of the baseline repeats)`; it is derived in
`state.json`, so compare against it instead of eyeballing two numbers. A retry of the
same idea is a new round with the same `change` string, so `rounds.tsv` shows the repeats.
