---
name: self-review
description: "Review your own verl-omni branch against the project rubric before opening or updating a PR. Use before submitting a contribution, or whenever asked to self-review. Report-only: covers technical purpose, code quality, goal completeness and validation evidence, with a READY / NEEDS CHANGES verdict. Never edits files."
---

# Self-review

Review the **whole** change, not only its latest commit. Keep the report
proportional to the diff; a small fix does not need a technical essay.

Report-only — do **not** edit, commit, or push as part of reviewing. Once the
contributor makes fixes, load [commit-and-pr](../commit-and-pr/SKILL.md) for the
commit/PR conventions and [run-cpu-tests](../run-cpu-tests/SKILL.md) for tests.

## 1. Establish scope

Identify the intended base and its remote; a stacked PR may not target `main`.
Record the exact base and head SHAs, then inspect the whole three-dot diff:

```bash
git remote -v
git fetch <base-remote> <base-branch>
git rev-parse <base-ref> HEAD
git diff <base-ref>...HEAD
```

Do not replace this with a last-commit review. A two-dot comparison against newer
`main` includes main-only changes; check the merge base before calling those
regressions. Use individual commits for history, not to omit parts of the final
diff. For local uncommitted work, also inspect `git diff --cached` and `git diff`.
Re-check affected evidence if the reviewed head changes.

## 2. Read the applicable rules

`AGENTS.md` is the top-level contract. Read the current guides, not a remembered
copy. Formatting and project-specific conventions belong to these sources:

| Area | Guide |
| --- | --- |
| code style, runtime boundaries, shell recipes | `.agents/rules/code-style.md` |
| config dataclasses / generated YAML | `.agents/rules/config.md` |
| diffusion pipelines / adapters | `.agents/rules/pipelines.md`, `docs/contributing/integrating_a_diffusion_model.md` |
| diffusion algorithm | [add-pipeline](../add-pipeline/SKILL.md) selects the policy-gradient or direct-preference guide |
| reward scorers | `.agents/rules/reward.md` |
| tests | `.agents/rules/testing.md`, `docs/contributing/testing_guide.md` |
| recurring traps | `docs/contributing/common_pitfalls.md` |
| CI / GPU smoke | `docs/contributing/ci_cd.md`, `docs/contributing/gpu_smoke_tests.md` |

## 3. Review and report in four parts

### A. Purpose and technical background

Summarize the problem, the mechanism being changed, and why the change is needed.
Trace the relevant callers and consumers: e.g. config → allocation → worker,
rollout → replay → loss, or adapter export → binding → forward. Check whether an
existing implementation already solves it before recommending another abstraction.

### B. Code quality and correctness

Inspect changed lines **and their execution context**, using the code-style rules.
Classify findings by subject; these categories are not severity levels:

| Category | Inspect |
| --- | --- |
| Correctness | tensor shape/dtype/device, gradients, wire compatibility, concurrency and resource lifetime |
| Performance | host/device synchronization, allocation, hot-loop overhead and measured workload equivalence |
| Maintainability | ownership, reuse, imports, naming and unnecessary abstractions |
| Style | clarity, useful comments/types and consistency beyond automated formatting |
| Process | reproducible tests, dependency compatibility and evidence for the claimed scope |

Mark an issue **blocking** or **non-blocking** from its impact, independently of
category. Missing critical validation can block; a naming preference usually
cannot. Do not invent findings to avoid saying the code is sound, or impose
source-line limits, blanket bans on framework hooks, or formatter-only nits.

Project-specific checks:

- **Wire compatibility:** for protocol-preserving refactors, keep valid
  `prompt_token_ids`, `multi_modal_data` and `extra_fields` intact end to end.
  Unsupported or conflicting fields must not disappear silently.
- **Media contracts:** use `DiffusionIOSpec` / `media_kind` rather than guessing
  modality from `ndim` or a dimension equal to three.
- **Distributed paths:** verify actual replica/rank allocation, sample ordering
  and loss normalization, not just a parsed config or a printed parallel degree.
- **Weight sync:** trace export, name/shape/scaling conversion, binding and use.
  A tensor count or active adapter ID alone cannot prove value-correct binding.
- **Surgical scope:** flag unrelated cleanup and orphaned code caused by this
  change; do not demand refactoring pre-existing debt unrelated to the goal.

Each finding needs: **severity + category + evidence tag → `path:line` → triggering
input/path → impact and why → concrete fix or next verification step**.

### C. Goal completeness

Compare the final diff with the issue and PR description. List unmet requirements,
unsupported configurations, prerequisite PRs and changed defaults/compatibility.
An intentional limitation is not a proven bug, but must not be presented as
implemented or validated support. Check that examples and docs use the same
contract as the code.

### D. Validation and accountability

Separate what you ran from author-reported evidence and untested paths. Choose the
required layer from the testing guide; CPU tests, GPU trainer completion,
performance and convergence are different claims, not interchangeable badges.

- Tag findings **[verified]** (read exact source or reproduced), **[likely]**
  (inferred, with the missing check named), or **[unchecked]** (not inspected).
  Keep unchecked items as coverage gaps, not confirmed defects.
- Give the exact command and result, or **not run**. Associate test logs with the
  tested SHA/config; a previous head's green check is not current-head evidence.
- Check `.github/*_pin.txt` and the interpreter/worker source binding before
  attributing an import or runtime failure to the patch. Compare with the base
  when needed; a dependency mismatch is not automatically a code regression.
- Read the automated gates' actual scope. CPU selection does not replace the
  config-doc, generated-config, device-API, DataProto or other sanity checks.
- "Engine initialized" is not a completed GPU e2e. For timing claims identify
  the baseline, workload, hardware, warmup and all measured samples; distinguish
  actor-only from whole-training time and disclose contention.
- Assess concrete problems such as swallowed errors, unnecessary fallbacks or
  unsupported claims; **do not infer AI authorship from code style**. Follow
  `AGENTS.md` for disclosure and human accountability, and never check off human
  review on someone else's behalf.

End with **READY / NEEDS CHANGES**, blocking actions, and residual risks. READY
means no blockers for the stated scope, not proof of untested GPU behavior or
convergence. If required evidence is missing, explain why it blocks. If there
are no findings, say what supports that conclusion rather than just "LGTM".

## 4. Handoff, not automatic publication

Iterate after the contributor fixes findings. Keep review notes out of the code
diff. Suggest updating the PR description when scope or evidence changes.

Publishing follows the accountability rules in `AGENTS.md`: draft review
comments and replies for the user, and post only what they asked for and
approved word for word.

When asked to suggest reviewers, use `docs/community/governance.md` for ownership
and `.github/CODEOWNERS` for path routing (last matching rule wins). Suggest at
most two relevant owners; do not automatically tag them.

## Template provenance

The four-part review structure is adapted from
[Code Review Style Guide](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32).
Project rules take precedence: categories are separate from severity, abstraction
is contextual, and evidence replaces speculative AI-code detection.

<!--
MAINTAINER GUIDE — Keep rule details in .agents/rules/ and procedures in docs/.
Recheck this rubric when review permissions, wire contracts or CI gates change.
-->
