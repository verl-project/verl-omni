# Records

Everything the session knows is in `rounds.tsv`. The other files are derived views, so
a disagreement is resolved by re-rendering, never by editing them.

## `rounds.tsv` (append-only)

Ten columns, one row per event. The header must match this order byte for byte;
the adapter refuses a log written by another schema instead of migrating it.

| Column | Meaning |
|---|---|
| `ts` | ISO 8601 with offset |
| `round` | `c<cycle>-<phase>-<seq>`, e.g. `c01-opt-002` (empty for `phase`/`close` rows) |
| `phase` | `prelude` `enablement` `optimization` `sweep` `close` |
| `event` | `start` · `result` · `verdict` · `phase` · `close` |
| `change` | the one change the round tries (`none` for baseline/sweep) |
| `value` | the measured objective |
| `noise` | the noise floor used for this verdict, when known |
| `verdict` | `baseline` · `keep` · `revert` · `crash` (only on `verdict` rows) |
| `quality` | `ok` · `smoke` · `degraded` |
| `why` | short reason code: `improved`, `within_noise`, `regressed`, `constraint`, `crash_timeout`, `measured`, … |

Round ids: `pre` prelude · `ena` enablement · `opt` optimization · `swp` sweep.
Sequence numbers are per cycle, so a reloop restarts at `c02-opt-001`.

**Ordering enforced by the adapter** (exit 4 otherwise):

- one open round at a time: a new `start` is refused until the current round has a verdict;
- `result` requires a `start`, and is recorded once per round;
- `verdict` requires a `result`, and is recorded once per round — verdicts are not rewritten;
- `keep`/`baseline` require a measured `value`;
- `phase` moves must follow the phase graph (`references/phases.md`);
- `close` is terminal: nothing else may be appended afterwards.

`append` holds an exclusive lock and `fsync`s, so a crash mid-write leaves either the
whole row or none of it.

## Round directory

`rounds/<round>/` carries the evidence a reviewer needs without reading the log:

| File | Written by | Content |
|---|---|---|
| `change.yaml` | `start` | phase + the change string |
| `command.txt` | `start` | the exact command that ran (launcher, overrides, timeout) |
| `result.json` | `result` | value, noise, quality, why, evidence pointer, timestamp |
| `verdict.json` | `verdict` | verdict, why, timestamp |
| `notes.md` | you | 3–8 lines: hypothesis, what the numbers said, why keep/revert |
| `artifacts/` | you | logs, profiles, traces — raw evidence, referenced from `result.json` |

## `state.json` (derived)

The anchor comes from the rounds whose phase is `prelude`. A baseline re-measured later
(a drift control during `sweep`) is counted in `anchor.controls` and never moves it —
otherwise every later verdict would be judged against a number that shifted underneath it.

| Field | Meaning |
|---|---|
| `session_id` | from `init --id` |
| `phase` | current phase |
| `cycle` | 1, and +1 per `sweep → optimization` reloop |
| `mission` | `metric`, `direction` (`lower`/`higher`), optional `target` |
| `budget` | `max_rounds`, `per_round_timeout` |
| `anchor` | `value` (median of the **prelude** baseline rounds), `sd`, `measurements`, `rounds`, `controls` |
| `noise_floor` | `max(1 % × |anchor|, 2 × sd)` — fixed by the prelude, so a later verdict is judged against the same number |
| `best` | best **kept** value (the anchor if nothing was kept) |
| `target_reached` | whether `best` meets `mission.target` in the mission direction |
| `stack` | kept rounds only, in order: `{round, change, value}` |
| `rounds` / `crashes` | decided rounds and crashes |
| `current` | the open round id, or empty |
| `next_action` | `measure_baseline` · `finish_round` · `run_round` · `close` · `done` |
| `stop` | `{reason, detail}` once closed |

`next_action` is the first true row of: closed → `done`; a round is open →
`finish_round`; phase is `prelude` without an anchor → `measure_baseline`; budget spent
or target reached → `close`; otherwise `run_round`.

## `session_breakdown.json` (derived, for downstream tools)

```json
{
  "schema_version": "verl-omni.session_breakdown.v1",
  "session_id": "…",
  "phases": {"order": ["prelude", "enablement", "optimization", "sweep", "close"],
             "entered": ["prelude", "optimization"],
             "current": "close", "cycles": 1},
  "outcome": {"baseline": 166.5, "baseline_sd": 1.45, "noise_floor": 2.9,
              "final": 155.9, "target_reached": false, "rounds": 3, "crashes": 0,
              "stop": {"reason": "budget", "detail": "…"}},
  "rounds": [{"round": "c01-opt-002", "phase": "optimization", "change": "max_num_seqs=128",
              "value": "155.9", "verdict": "keep", "why": "improved"}]
}
```

The schema version is part of the contract: add fields, do not repurpose them.

## `kb/` — what survives the session

- `recipes.jsonl`, one JSON object per line: the winning stack, the workload, the
  measured numbers, and the session id. This is what a later session warm-starts from.
- `lessons.md`, short bullets: what failed and **why**. A rejected idea with a reason is
  worth more than a winning number, because it stops the next session repeating it.

Both are read at `prelude` (so the first hypothesis starts from known ground) and
written at `close`. Keep entries short and numeric; a lesson without a number is a mood.

## Exit codes

`0` ok · `1` usage (missing `--dir`, existing session, bad mission) · `2` header drift ·
`3` invalid field (unknown event/verdict/quality, non-numeric value, tab in a field) ·
`4` refused by ordering (result before start, second verdict, illegal phase move,
anything after `close`).
