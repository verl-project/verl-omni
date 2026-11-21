---
name: experiment-campaign
description: "Run a bounded optimisation session against a user-defined objective (speed, memory, consistency or quality) on verl-omni runs. Use when the task is iterative measurement rather than a one-shot fix: anchor a stock baseline, change one thing at a time, keep or revert on re-measured evidence, sweep the accumulated stack, and record what was learned. The session moves through PRELUDE → ENABLEMENT → OPTIMIZATION → SWEEP → CLOSE (enablement only when the baseline does not run), and closes when the budget, the target or convergence says so. This skill is the coordinator only: it measures and decides; diagnosis is delegated to specialized skills."
---

# Experiment campaign

One session is one closed loop: **input → optimise → validate → learn**.

```text
PRELUDE ─► ENABLEMENT* ─► OPTIMIZATION ⟲ ─► SWEEP ─► CLOSE
                             ▲              │
                             └─ new cycle ──┘

* ENABLEMENT runs only when the baseline does not run end to end.
⟲ OPTIMIZATION repeats once per hypothesis, until no idea is worth a round.
```

You are the coordinator. You run the benchmarks, keep the log, and decide
keep-or-revert. You do **not** analyse traces or invent diagnoses: hand those to a
specialized skill (`profile`, `ascend-fusion-op-replace`, `train-infer-consistency`, …)
and consume its finding. A predicted speed-up never decides a keep — only a
measurement you ran does.

`rounds.tsv` is the only history. `state.json`, `session.md` and
`session_breakdown.json` are derived from it; when they disagree, re-render.

## Minimum path

```bash
session.py init  --dir sessions/<id> --id <id> \
    --mission "metric=timing_s/step,direction=lower,target=150" \
    --max-rounds 12 --round-timeout 25m --launcher "bash /path/launcher.sh"

session.py start --dir sessions/<id> --phase prelude --change none          # → c01-pre-001
#   run the launcher under `timeout`, keep the log and the extractor output
session.py result  --dir sessions/<id> --round c01-pre-001 --value 166.5 --why measured
session.py verdict --dir sessions/<id> --round c01-pre-001 --verdict baseline --why measured

session.py start  --dir sessions/<id> --phase optimization --change "max_num_seqs=128"
session.py result --dir sessions/<id> --round c01-opt-002 --value 155.9 --noise 8.5 --why improved
session.py verdict --dir sessions/<id> --round c01-opt-002 --verdict keep --notes "why this helped"

session.py close --dir sessions/<id> --reason target --detail "155.9 <= 150? no: budget"
```

`render` (also run automatically by every command) rewrites `state.json`,
`session.md` and `session_breakdown.json`, and prints the state including
`next_action` — read that instead of guessing where you are.

## Session directory

```text
sessions/<id>/
├── state.json              derived: phase, cycle, anchor, noise floor, stack, next action
├── manifest.json           environment fingerprint (host, cwd, launcher, extractor)
├── session.md              the one human entry point (regenerated)
├── session_breakdown.json  machine-readable outcome for downstream tools
├── close.md                stop reason and what was kept
├── kb/                     across-session memory: recipes.jsonl, lessons.md
└── rounds/<round>/         change.yaml · command.txt · result.json · verdict.json · notes.md · artifacts/
```

## Four rules

1. **Measure before you change, and measure again after.** A round is a change plus
   the number it produced. No baseline, no optimisation.
2. **One change per round; only `keep` enters the stack.** A `revert` rolls the change
   back, so the next round starts from the last kept stack, not from a failed one.
3. **The coordinator judges.** Numbers from an agent, a kernel author or a prediction are
   evidence, not a verdict. A `keep` also needs the accuracy floor to hold.
4. **Bounded.** Rounds, wall clock and per-round timeout are set at `init`. When the
   budget is spent, the target is reached, or improvement stops, run `close` — and write
   what you learned into `kb/`.

## Phases

| Phase | Purpose | You may change |
|---|---|---|
| `prelude` | Anchor the baseline (repeat it) and profile once to see where the time goes | nothing — measurement only |
| `enablement` | Only if the baseline does not run: data, environment, launcher, memory | what makes it run; no gain is credited |
| `optimization` | The loop body: one hypothesis per round, config or code or a delegated operator replacement | one change per round, code only behind a revertible diff |
| `sweep` | Re-measure the accumulated stack across 2–3 operating points | nothing — validation only |
| `close` | Stop reason, final numbers, KB write-back | nothing |

Entry conditions, allowlists and exit criteria per phase: `references/phases.md`.
Log columns, round files, derived state and the breakdown schema:
`references/records.md`.

## Exit codes

`0` ok · `1` usage (bad path/args) · `2` header drift · `3` invalid field ·
`4` refused by the ledger's ordering rules (result before start, second verdict,
illegal phase move, anything after `close`). On `4`, fix the order — do not work
around it.

## Anchors

- `references/phases.md` — what each phase may touch, and how keep/revert is decided.
- `references/records.md` — `rounds.tsv`, `state.json`, round files, `session_breakdown.json`, `kb/`.
- `templates/session-init.sh` — the six commands above, ready to copy.
- `templates/kb-recipe.jsonl` — one row of the recipe knowledge base.
- Domain knowledge lives elsewhere: `docs/perf/tuning_guide.md` (what to tune, in which
  order), `docs/start/metrics.md` (metric meaning), `docs/perf/profiler.md` (profiling),
  and the `profile` / `train-infer-consistency` skills.

<!--
MAINTAINER GUIDE — Keep this file the coordinator's manual: loop, six commands,
directory, four rules, phase table. Detail belongs in the two references. Recheck when
any of these change: the commands in adapters/session.py, the phase order or allowlists,
the log columns, or docs/perf/tuning_guide.md's decision order (the optimization phase
points at it). The four rules are the invariant set; a fifth means an existing rule moved.
-->
