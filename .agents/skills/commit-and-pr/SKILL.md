---
name: commit-and-pr
description: "verl-omni commit message + PR conventions and the mandatory contribution policy. MUST load before any git commit or PR creation -- enforces the [{modules}] {type}: {description} title format, commit trailers, duplicate-work checks, and AI-assistance disclosure."
---

# Commit & PR Conventions

How to commit and open PRs in `verl-project/verl-omni`. **Load this before any
commit or PR.** The rules here come from `AGENTS.md` / `CLAUDE.md` and are
enforced by maintainers — breaching them can result in banning.

## When to Use

- Any `git commit` in this repo.
- Any PR creation (including delegated/agent commits).
- Whenever you infer a module/type label for a change.

## Step 0: Contribution policy (fail-closed)

Before proposing a PR, run the duplicate-work checks:

```bash
gh issue view <issue_number> --repo verl-project/verl-omni --comments
gh pr list --repo verl-project/verl-omni --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl-omni --state open --search "<short area keywords>"
```

- If an open PR already addresses it → **do not open another.**
- If your approach differs materially → explain the difference in the issue.
- **No low-value busywork PRs** (single typo, one style tweak, one mutable
  default). Mechanical cleanups are acceptable only bundled with substantive work.
- If the work is duplicate/trivial → **stop** and report what's missing.

## PR title format

```
[{modules}] {type}: {description}
```

- **modules** (comma-separated if several). The gate is
  `tests/special_sanity/check_pr_title.py`, not `AGENTS.md` — it accepts the official
  `AGENTS.md` set (`vllm_omni`, `diffusion`, `omni`, `rollout`, `trainer`, `reward`,
  `model`, `algo`, `fsdp`, `ray`, `worker`, `data`, `cfg`, `ckpt`, `doc`, `ci`,
  `tests`, `docker`, `misc`) **plus** `training_utils`, `single_controller`,
  `recipe`, `perf`, `env`, `tool`. Prefer the official set; the extras are real
  (`recipe` for `examples/` work, `perf` for perf changes). Anything outside the
  validator's list fails the check.
- **type**: `feat`, `fix`, `refactor`, `chore`, `test`.
- `[BREAKING]` if it breaks any API (CLI args, config, signatures) — placed
  **immediately before the module bracket**: `[BREAKING][cfg] refactor: ...`, *not*
  `[cfg] [BREAKING] ...` (the validator only strips a leading `[BREAKING]`).
- For a stacked/multi-part PR series, prefix `[N/N]` (single digits only), e.g.
  `[1/N][omni] feat: ...`. Combined with `[BREAKING]` the order is
  `[N/N][BREAKING][module]`.

### Module inference from changed paths

| Path                                   | Module        |
| -------------------------------------- | ------------- |
| `verl_omni/pipelines/`                 | `diffusion`   |
| `verl_omni/pipelines/qwen3_omni/`      | `omni`        |
| `verl_omni/trainer/`                   | `trainer`     |
| `verl_omni/trainer/config/`, `*/config/` | `cfg`       |
| `verl_omni/utils/reward_score/`, `verl_omni/reward_loop/` | `reward` |
| `verl_omni/workers/rollout/`           | `rollout`     |
| `verl_omni/workers/`                   | `worker`      |
| `verl_omni/models/`                    | `model`       |
| `verl_omni/utils/vllm_omni/`           | `vllm_omni`   |
| `docs/`                                | `doc`         |
| `tests/`                               | `tests`       |
| `.github/`, CI                         | `ci`          |
| `docker/`                              | `docker`      |

### Title examples

```
[reward] feat: add pickscore visual reward
[diffusion, cfg] feat: add teacher-anchored distillation losses for OPD
[omni] fix: correct attention mask for Qwen3-Omni text+image inputs
[tests] test: cover qwen-image DPO adapter guidance branching on CPU
[BREAKING][cfg] refactor: rename guidance_scale to cfg_scale
```

Real history to match style: `[trainer, algo, cfg] feat: ...`,
`[diffusion, cfg, tests] feat: ...`, `[1/N][omni] feat: ...`.

## Commit messages

Body explains **why**, not what. Wrap ~72 chars. For AI-assisted commits, add an
explicit disclosure line in the **body** and attribution **trailers** (this is
the actual repo convention — see recent history):

```
[reward] feat: add pickscore visual reward

Add a PickScore-based scorer routed via default_compute_score_image so
flow-GRPO runs can optimize human-preference reward.

AI assistance (<your tool name>) was used for this change.

Co-authored-by: <your tool name>
Signed-off-by: Your Name <your.email@example.com>
```

The `Co-authored-by` trailer names the assisting tool **actually used** —
substitute your own tool name, do **not** copy `Claude Code` verbatim unless
you are Claude Code. `Signed-off-by` with a real name/email is required.

## PR description (AI-assisted work — mandatory)

The PR body **must** include:

1. Why this is **not** duplicating an existing PR (cite your Step 0 checks).
2. **Test commands run and their results.**
3. A clear statement that **AI assistance was used**.
4. A note that a **human submitter** has reviewed every changed line — pure
   code-agent PRs are not allowed.

## Pre-commit

Ensure hooks pass before committing (`pre-commit install` once). If you touched
config dataclasses, run `scripts/generate_trainer_config.sh` and commit the
regenerated `_generated_*.yaml` (see [config rule](../../rules/config.md)).

## Common Mistakes

- ❌ Title with a module not in `check_pr_title.py`, or an invalid type.
- ❌ `[BREAKING]` after the module bracket (`[cfg] [BREAKING]`) instead of before it.
- ❌ Opening a PR without the duplicate-work checks.
- ❌ Omitting the AI-assistance disclosure or trailers.
- ❌ Committing a stale `_generated_*.yaml`.
