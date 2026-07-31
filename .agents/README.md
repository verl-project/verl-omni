# `.agents/` — Agent rules & skills for verl-omni

Repo-local guidance for AI-assisted contributions, complementing the mandatory
contribution policy in [`AGENTS.md`](../AGENTS.md) / [`CLAUDE.md`](../CLAUDE.md).

Every claim in these files was checked against the tree it describes — the
pre-commit hook scripts, the registry declarations, the manager call sites, the CI
workflow — not against prose in the docs. Where the repo and its own documentation
disagree, these files record the repo and say so.

## Division of labour

- **[`docs/contributing/`](../docs/contributing/) is authoritative for procedures.**
  Seven `integrating_*` guides, each ending in a final checklist, plus a testing
  guide and a symptom-first pitfalls reference. Nothing here restates them.
- **`skills/`** are **routers and deltas**: they classify the task, name the guide
  that owns it, and add only what the guide does not cover. Invoked as `/<skill>` or
  auto-loaded by description.
- **`rules/`** carry **contracts and invariants**: the signature a manager calls,
  what a pre-commit hook actually greps for, which filename suffix CI selects. They
  load automatically for files matching their `paths:` frontmatter.

A fact belongs in exactly one place. When a guide and a file here would say the same
thing, the guide wins and the file links to it.

## `rules/`

| Rule                              | Applies to                                                    | Key point |
| --------------------------------- | ------------------------------------------------------------- | --------- |
| [code-style](rules/code-style.md) | everywhere                                                    | what each pre-commit hook enforces, how sparsely this tree comments, and the four reuse mechanisms to use instead of copying |
| [pipelines](rules/pipelines.md)   | `verl_omni/pipelines/**`                                      | dispatch is by registry key and registration is an import side effect; training adapters are never instantiated |
| [reward](rules/reward.md)         | `verl_omni/utils/reward_score/**`, `verl_omni/reward_loop/**` | scorers are selected by config, not by the `data_source` dispatcher; managers call by keyword |
| [config](rules/config.md)         | `verl_omni/trainer/config/**`, `verl_omni/workers/config/**`  | inherit verl's `BaseConfig`, declare `_mutable_fields`, regenerate the YAMLs |
| [testing](rules/testing.md)       | `tests/**`, `test_*.py`                                       | placement is the only commit gate — the `_on_cpu` suffix that decides what CI runs is not enforced |

## `skills/`

| Skill                                                | Use for                                                        |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| [commit-and-pr](skills/commit-and-pr/SKILL.md)       | commit trailers, `[{modules}] {type}:` titles, duplicate-work checks, AI-assistance disclosure |
| [add-pipeline](skills/add-pipeline/SKILL.md)         | routing a model / algorithm integration to the right guide under `docs/contributing/` |
| [add-reward-score](skills/add-reward-score/SKILL.md) | a new reward scorer plus the config overrides that select it    |
| [run-cpu-tests](skills/run-cpu-tests/SKILL.md)       | what the CPU job does that `testing_guide.md`'s local commands don't |

`commit-and-pr` holds the authoritative module list; other files link to it rather
than duplicating it.

## Other agent tools

The files live here. `.claude/` and `.codex/` are symlinks to the `skills/` and
`rules/` directories in this one, so each tool finds the same content at the path it
looks for:

```
.claude/skills -> ../.agents/skills      .codex/skills -> ../.agents/skills
.claude/rules  -> ../.agents/rules       .codex/rules  -> ../.agents/rules
```

Whole directories, not per-file links — a skill added under `.agents/` shows up in
both without anyone remembering to link it. Add content only here; `CLAUDE.md` →
`AGENTS.md` is the same arrangement one level up.

## Maintaining these files

- Verify before you edit. Read the enforcement script, count the occurrences, read
  the source — a convention with zero occurrences in the tree is not a convention.
- Prefer a command over a snapshotted number, so the file cannot silently go stale.
- Each skill ends with a `MAINTAINER GUIDE` comment naming what invalidates it.
- Editing agent instructions is itself governed by
  [`docs/contributing/editing-agent-instructions.md`](../docs/contributing/editing-agent-instructions.md).
