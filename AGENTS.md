# Agent Instructions for verl-omni

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl-omni`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl-omni --comments
gh pr list --repo verl-project/verl-omni --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl-omni --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# GPU (two steps — engine stack first, then rollout + train)
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install -e ".[vllm-omni,train,dev]"

pre-commit install
```

See [`docs/start/install.md`](docs/start/install.md) for optional extras.

### PR title format

Titles must follow `[{modules}] {type}: {description}`.

Valid modules: `vllm_omni`, `diffusion`, `omni`, `rollout`, `trainer`, `reward`, `model`, `algo`, `fsdp`, `ray`, `worker`, `data`, `cfg`, `ckpt`, `doc`, `ci`, `tests`, `docker`, `misc`.

Valid types: `feat`, `fix`, `refactor`, `chore`, `test`.

Add `[BREAKING]` prefix if the PR breaks any API (CLI arguments, config, function signatures).

### Commit messages

Add attribution using commit trailers:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [verl project](https://github.com/verl-project/verl)'s [`AGENTS.md`](https://github.com/verl-project/verl/blob/main/AGENTS.md), which was itself adapted from the [vLLM project](https://github.com/vllm-project/vllm).
