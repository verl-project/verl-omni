---
name: self-review
description: "Review your own verl-omni branch against the project rubric before opening or updating a PR. Use before submitting a contribution, or whenever asked to self-review. Report-only: it groups findings by severity, gives a READY / NEEDS CHANGES verdict, and never edits files. Mirrors the checks a maintainer applies, over your whole diff."
---

# Self-review

Catch issues before a maintainer does, over your **whole** PR diff. You are
already on the branch with the project conventions available, so: get the diff →
review it against the rubric and the guides for the areas you touched → report →
iterate with the contributor until it is ready, then remind them to put the
notes on the PR.

Report-only — do **not** edit, commit, or push as part of reviewing. Once the
contributor makes fixes, load [commit-and-pr](../commit-and-pr/SKILL.md) for the
commit/PR conventions and [run-cpu-tests](../run-cpu-tests/SKILL.md) for the
tests you cite.

## 1. Get the diff

Identify the remote that points to `verl-project/verl-omni` (do not assume it is
named `origin`), then diff the whole branch against its base:

```bash
git remote -v
git fetch <upstream-remote> main
git diff <upstream-remote>/main...HEAD        # whole branch, not just the last commit
```

If the branch trails `main` and the diff looks polluted with unrelated merged
files, scope to your own commits instead: `git log <upstream-remote>/main..HEAD
--oneline`, then `git show <commit>`.

## 2. Read the rules for what you touched

Review against the project rules, not a remembered copy. `AGENTS.md` is the
top-level contract; then read the guide for each area you changed:

| Area | Guide |
| --- | --- |
| config dataclasses / generated yaml | `.agents/rules/config.md` |
| code style / `# Copied from` | `.agents/rules/code-style.md` |
| diffusion pipelines / adapters | `.agents/rules/pipelines.md`, `docs/contributing/integrating_a_diffusion_model.md` |
| diffusion algorithm (policy-gradient vs direct-preference) | `docs/contributing/integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md`, `docs/contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md` (let [add-pipeline](../add-pipeline/SKILL.md) classify which one) |
| reward scorers | `.agents/rules/reward.md` |
| tests | `.agents/rules/testing.md`, `docs/contributing/testing_guide.md` |
| recurring traps | `docs/contributing/common_pitfalls.md` |
| CI / GPU smoke | `docs/contributing/ci_cd.md`, `docs/contributing/gpu_smoke_tests.md` |

## 3. verl-omni rubric

Beyond generic correctness, these project-specific traps are the ones a generic
review misses — each has bitten a real PR and none is obvious from the diff
alone:

- **Silent field loss / wire compatibility** — the rollout request/output path
  threads private engine keys (`prompt_token_ids`, the dual-written
  `multi_modal_data`, `extra_fields`). Keep valid requests byte-identical on the
  wire and fail closed on conflicting/unsupported fields rather than dropping or
  overwriting them.
- **Shape guessing** — flag new `ndim==5` / `shape[1]==3`-style modality or
  layout inference; the declared media contract (`DiffusionIOSpec`, `media_kind`)
  should drive it instead.
- **Sanity gates the CPU job skips** — changed `config` dataclasses need the
  regenerated `_generated_*.yaml` (`scripts/generate_trainer_config.sh`);
  `check_dataproto_usage.py`, `check_device_api_usage.py` (no literal `cuda` /
  `nccl` / `.cuda`), `check_docstrings.py`, `check_license.py`, and
  `validate_structure.py` each run as their own job.
- **Pin-induced false failures** — a red test may be a local `verl` /
  `vllm-omni` pin mismatch, not the diff. Confirm the env matches
  `.github/*_pin.txt` before treating it as a finding.
- **Don't overstate GPU evidence** — "reached engine init" is not "end-to-end
  passed"; cite only what a run actually completed.
- **Surgical diff** — every changed line should trace to the stated goal; flag
  unrelated refactors, reformatting, and orphaned dead code (unused imports,
  variables, functions left by your own change).

## 4. Report

- **Blocking** — numbered. Each: title → explanation → `file.py:line` → impact.
  Cite the rule, e.g. *Per `.agents/rules/config.md`: regenerate the yaml.*
- **Non-blocking** — same format, lower severity: raise with the reviewer rather
  than guess at now.
- **Dead code (advisory)** — a short table: `path:line` · Likely-dead / Used ·
  reason.
- **Summary** — a short synthesis and a verdict (**READY** / **NEEDS CHANGES**),
  spelling out what to **fix before submitting** vs what to **leave for the
  actual review**.

Be concrete, cite the rule, review the whole diff, and don't invent issues or
flag pure style the formatter already enforces.

### Evidence rule (mandatory)

Self-review is prone to confident guessing. Every finding and every claim must
be grounded, or explicitly marked as ungrounded:

- Tag each finding with its evidence basis: **[verified]** (you read the exact
  `file.py:line` and it shows the problem), **[likely]** (inferred from the diff
  but not confirmed at the source), or **[unchecked]** (rubric item you did not
  actually inspect). A `file.py:line` with no matching read is not `[verified]`.
- **Never claim a check passed that you did not run.** State the exact command
  and its result, or write "not run". Do not report "CPU tests pass", "ruff
  clean", or "config regenerated" from assumption.
- **Separate verified vs unverified in the summary.** List what actually ran
  (command + outcome) apart from what remains unverified (e.g. GPU end-to-end,
  untested pipelines). A **READY** verdict must name what it does *not* cover.
- If you could not open a file or run a check, say so plainly rather than
  producing a plausible-sounding finding. An honest gap beats a fabricated one.

## 5. Iterate, then share

Expect several rounds: the contributor fixes findings, you review again. Keep
going until the verdict is **READY** — only the *leave for the actual review*
items should reach the reviewer unresolved. End by reminding the contributor to
put the final summary in the PR description or a comment; it saves the reviewer a
round-trip. Never commit the review notes as part of the diff.
