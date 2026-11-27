---
paths:
  - "**/*"
---

# Code Style Rules

Code conventions and the automated pre-commit checks that enforce their
mechanically detectable parts. Formatting is handled by the tools below.

## Automated gates (run before every commit)

`pre-commit install` wires these up. What each hook **actually** enforces — read off
`.pre-commit-config.yaml` and `tests/special_sanity/`, not off the hook names:

| Hook                       | What it actually enforces                                                  |
| -------------------------- | -------------------------------------------------------------------------- |
| `ruff` / `ruff-format`     | Lint + format, `line-length = 120` (scope below)                           |
| `check-pr-lint`            | Incremental AST contracts, example recipes, comments, docs, and Ruff ([pr_lint_checks.md](../../docs/contributing/pr_lint_checks.md)) |
| `mypy`                     | Errors enabled only in configured module overrides; global `ignore_errors = true` |
| `check-license`            | Accepted copyright text in tracked `.py` files; does not validate the full license header |
| `autogen-trainer-cfg`      | `_generated_*.yaml` matches flattened Hydra source configs |
| `check-docstrings`         | Presence (not style) of docstrings, in a hardcoded file list — see below     |
| `check-naming-conventions` | **Spelling only** — two project names, see below                          |
| `validate-structure`       | Test files must live in `tests/<module>/` (see [testing.md](testing.md))    |
| `check-device-api-usage`   | No `.cuda` / `"cuda"` / `"nccl"` under `verl_omni/`; per-file whitelist in the script |
| `check-dataproto-usage`    | No `DataProto` under `verl_omni/workers/engine/` — use `TensorDict`         |
| `compileall`               | Every `.py` byte-compiles with `PYTHONWARNINGS=error`                      |
| `check-docs-time-info`     | `Last updated` info in docs                                                |

A green hook covers only its configured scope. Read `[tool.mypy]` and its
overrides in `pyproject.toml` before claiming type coverage; new public contracts
still need explicit types. Formatting is not a correctness check.

Never hand-edit `_generated_*.yaml` — regenerate via
`scripts/generate_trainer_config.sh` with the pinned dependencies installed
(see [config.md](config.md)).

## Spelling (enforced, repo-wide grep)

`check-naming-conventions` greps the **whole worktree** — code, comments, docs,
markdown — and fails the commit on a misspelling of either project name: write
**`verl`** (all lower case) and **`SGLang`** or **`sglang`** (no other casing).

Read the exact patterns off the hook rather than restating them; a file that quotes
a rejected spelling in order to document it fails the hook too:

```bash
grep -A6 'id: check-naming-conventions' .pre-commit-config.yaml
```

Its `--exclude-dir` list covers `.git`, `.github`, `.specstory`, `venv`, `.venv`,
and `__pycache__` — nothing else, so this directory is in scope.

## License Header (mandatory)

Every new `.py` file needs the full Apache 2.0 header, following a neighboring
module. Preserve existing authorship and upstream attribution when adapting code.
The hook's accepted copyright strings are in
`tests/special_sanity/check_license.py`; passing its substring check is not a
substitute for the complete header.

## Docstrings

**What the gate actually checks**: `check-docstrings` only verifies that public
(non-`_`) top-level functions, classes, and methods **have some docstring**, and only
in a hardcoded file list — read it off the hook rather than guessing:

```bash
grep -n 'verl_omni/' tests/special_sanity/check_docstrings.py
```

`check-pr-lint` enforces public API docstrings (`VO015`), type annotations
(`VO018`), and rejects boilerplate docstrings (`CM004`). Style details below are
**review conventions**:

- Document `__all__` exports, shared pipeline contracts, and `compute_score*`
  scorers with concise docstrings (`VO015`). Omit trivial internal docstrings that
  only echo a signature (`CM004`), and never add a `_` prefix just to bypass a gate.
- Use concise Google-style `Args:` / `Returns:` / `Raises:` sections only when
  they convey necessary contracts; omit empty or repetitive template sections.
- Document tensor shapes and dtypes explicitly, e.g. `(C, H, W)` or `(N, C, H, W)`.
- Credit upstream in the module docstring when code is adapted
  (e.g. `It is adapted from https://github.com/kvablack/ddpo-pytorch.`).
- Recent review history favors **terse** docstrings — verbose explanatory blocks
  have been trimmed on request (see #311).

## Comments

Default to no implementation comments. Put necessary caller-facing API contracts
in concise docstrings. Keep an inline comment only for information the code
cannot express: an algorithm's source or invariant, tensor layout, concurrency
ordering, or a verified upstream workaround with its removal condition.

Do not generate line-by-line narration, numbered step commentary, edit history
(`Added`, `Updated`, `Changed`), decorative section banners, or commented-out
code. Rename or simplify code that needs a comment merely to restate what it does.
Long explanations belong in `docs/`; do not move noise into a docstring instead.

`check-pr-lint` enforces these rules (`CM001`–`CM005`, Ruff `ERA001`), including
rejecting PR/RFC/issue tracker references across code and docs (see
[pr_lint_checks.md](../../docs/contributing/pr_lint_checks.md)). Preserve
license headers, tooling directives, and required config comments ([config.md](config.md)).

## Imports and lint scope

Read the selected and ignored rules off `[tool.ruff.lint]` in `pyproject.toml`.
Two ignores shape style: star imports (F403/F405) and f-strings in logging calls
(G004) are allowed.

- Ruff `isort` orders stdlib, third-party, then first-party `verl_omni`.
- Prefer explicit imports in new code; star imports remain in some `__init__` and
  config re-export modules.
- Keep heavy optional imports behind their feature boundary so unrelated CPU
  paths remain importable. A selected scorer may import its own model dependencies
  at module scope. Do not hide a broken installation of a required dependency
  behind an unrelated fallback.

## Naming Conventions

Not automated (`check-naming-conventions` only checks spelling — see above), but
consistently followed:

| Type                 | Pattern                | Example                              |
| -------------------- | ---------------------- | ------------------------------------ |
| Config dataclass     | `XxxConfig`            | `DiffusionModelConfig`, `OmniAlgoConfig` |
| Loss class           | `XxxLoss`              | `OmniDPOLoss`                        |
| Training adapter     | `<Model><Algo>`        | `QwenImageDPO`, `StableDiffusion3FlowGRPO` |
| Rollout adapter      | `XxxWithLogProb` for policy-gradient algos, else `<Model><Algo>Pipeline` | `QwenImagePipelineWithLogProb`, `QwenImageDPOPipeline` |
| Reward scorer fn     | `compute_score` or `compute_score_<name>` | `compute_score_hpsv3` |
| CPU test file        | `test_*_on_cpu.py`     | `test_qwen_image_dpo_adapter_on_cpu.py` |

CPU-test naming is load-bearing, not cosmetic — CI selects tests by that suffix
([testing.md](testing.md)). The handful of non-`_on_cpu` test files are GPU/NPU tests,
`tests/special_sanity/` checks, and a few older `tests/workers/` files.

Name local values for what they hold — `noisy_latents`, `prompt_embeds`,
`sample_level_rewards` — not `x`, `h` or `out`. Short symbols such as `sigma` or
`dt` are fine where the code transcribes a cited equation.

## Device handling

**Avoid hardcoded `.cuda` / `"cuda"` / `"nccl"`; use the device API from
`verl.utils.device`** — `get_device_name`, `is_cuda_available`, `is_npu_available`
are the ones already imported across this tree (upstream verl, not a local module).

`check-device-api-usage` gates this over all of `verl_omni/`, but it is a plain
substring grep on file contents, so it also flags docstrings, comments, and
`EngineRegistry.register(device=["cuda", "npu"])` declarations. Those are exempted
per file in `tests/special_sanity/check_device_api_usage.py`; a handful of genuinely
hardcoded sites are exempted there too, with a `# TODO`. Fix the call; adding an
entry changes the check (see `AGENTS.md`, *Protected surfaces*).

## Reuse over duplication

Sharing between sibling implementations is the established pattern here, not an
aspiration. Four mechanisms are in active use — reach for one before copying a
block:

1. **A sibling's `common.py`.** A few pipeline packages
   (`qwen_image_flow_grpo`, `bagel_flow_grpo`, `wan22_dance_grpo`) export shared
   helpers, and other packages import across directories rather than re-implement:
   ```python
   from verl_omni.pipelines.qwen_image_flow_grpo.common import apply_true_cfg, build_img_shapes
   ```
2. **Subclass the closest existing class.** `QwenImageEditPlusFlowGRPO` and the NFT
   adapter both subclass `QwenImage` from `qwen_image_flow_grpo`; every loss class
   subclasses `DiffusionLossFn`; `MultiVisualRewardManager` subclasses
   `MultiRewardManager`; `PolicyGradient`/`DirectPreference` trainers share
   `BaseRayDiffusionTrainer`.
3. **A mixin, when the shared part cuts across the hierarchy.**
   `QwenImageTokenIdPromptMixin` (defined in `qwen_image_flow_grpo/common.py`) is
   mixed into several unrelated rollout pipelines the same way.
4. **A scoped refactor.** Consolidate shared behavior needed by the change; do not
   bundle neighboring cleanups. Follow `AGENTS.md`'s contribution policy.

Extract genuinely shared behavior, not merely similar-looking lines. There is no
fixed duplicate-line, function-length, or file-length threshold for source code.
Split by responsibility when it improves understanding; do not create a generic
framework or boolean-switch API just to meet a size target.

Two caveats, so this is not applied blindly:

- **Don't invent a shared abstraction for a single caller.** Most pipeline packages
  have no `common.py` — they had nothing worth sharing yet. The same holds for a
  private helper with one caller and no test of its own: inline it until a second
  caller appears.
- **Don't merge across a registry boundary.** Each `(architecture, algorithm)` pair
  registers its own adapter ([pipelines.md](pipelines.md)); collapsing two of them
  into one class with an `if algorithm == ...` switch defeats the dispatch.

## Signatures and control flow

- Pass tunables down from the config dataclass
  ([pipelines.md](pipelines.md#tunables-and-imports)); a lower layer takes the value
  as an argument instead of redeclaring the default. A one-use constant lives at
  its use site.
- When you own a signature, remove a parameter the function no longer uses instead
  of keeping it and `del`-ing it. An override keeps the signature its base defines.
- End an `if`/`elif` dispatch over a closed set of values with an `else` that raises,
  so a value added to the set later cannot fall through silently. A guard clause
  that returns or raises early needs no `else`.
- When porting research code, keep only the path this integration runs; drop the
  training, ablation and debug branches the adapter never takes.

## Runtime boundaries and state

- Validate user config, media metadata, and cross-process payloads at their entry
  points. Reject unsupported or conflicting values with actionable exceptions;
  do not silently drop fields or invent defaults for required replay data.
  Internal helpers can rely on the established contract instead of rechecking it.
- Keep exception handlers narrow. Fallbacks must be part of the documented
  contract, not a way to turn an invalid request into apparent success.
- Prefer explicit interfaces. `getattr`/`setattr` are appropriate for a verified
  optional-dependency capability or framework hook, not to conceal missing
  required state. Document the compatibility boundary, not every attribute access.
- Avoid mutating caller-owned configs during normalization. If shared state must
  change temporarily, define its owner and concurrency scope and restore it in
  `finally`. A lock must cover the protected state's whole use, without unrelated
  work inside it; neither locks nor in-place tensor operations are blanket bans.
- Give long-lived caches a bound or eviction policy and resources a clear owner.
  Do not scatter `empty_cache()` calls as speculative memory fixes.

## Example shell recipes

`check-pr-lint` enforces core recipe hygiene (`EX001`–`EX009`, `VO014`; see
[pr_lint_checks.md](../../docs/contributing/pr_lint_checks.md)). Remaining
items are review conventions. Keep recipes in the existing
algorithm/model directory and match the neighboring `run_*.sh` entrypoints.

- Group environment defaults near the top; document required paths and what each
  path points to. Do not embed local model, dataset, venv, or credential paths.
  Use the existing entrypoint and plain Hydra overrides for declared fields;
  reserve `+` for new keys (see [config.md](config.md)).
- Preserve caller overrides with `"$@"` last and quote each path-bearing argument,
  e.g. `"actor_rollout_ref.model.path=$MODEL_PATH"`. Keep output directories
  attributable to the recipe and overridable rather than sharing a fixed run path.
- For new scripts use a Bash shebang and `set -euo pipefail`, with explicit handling
  of intentionally recoverable commands. Debug tracing is optional; do not expose
  credentials. Preserve the trainer's failure status, including through `tee`.
- Validate positive parallel degrees and GPU divisibility before shell arithmetic.
  Derived replica/worker counts must agree with runtime topology; an engine flag
  alone does not allocate GPUs. State prerequisite patches or unsupported modes.
- Reuse the base recipe when a variant only selects a few overrides. Resolve that
  recipe relative to the script; document whether data/reward paths require a
  repo-root working directory. Do not build a new launcher framework.
- Let the scheduler or parent launcher own log redirection when it already does;
  avoid nesting process-substitution `tee` redirection in that case. Cleanup must
  target only the launched job, not every Ray or GPU process on the host.
- Check `bash -n` and capture argv with a stub trainer to verify defaults, quoting,
  caller precedence and exit propagation without launching GPUs. That does not
  establish runtime correctness; use the [testing guide](../../docs/contributing/testing_guide.md)
  for the required acceptance layer.

## Performance Patterns

Conventions, not gated:

- Avoid needless GPU→CPU syncs (`.item()`, `.cpu()`, `.tolist()`, `print(tensor)`)
  in hot paths; distinguish required boundary work from per-token/per-frame work.
- Prefer batched tensor ops over Python loops over elements. Preserve sample
  boundaries, ordering and loss normalization when batching.
- Be explicit about `dtype`/`device`; do not rely on implicit promotion.
- Measure suspected bottlenecks before adding caches, memory cleanup or alternate
  kernels. Use the [profile skill](../skills/profile/SKILL.md); compare matched
  workloads and distinguish actor-only timing from whole-training throughput.

## Modules

Changes are labeled by module in commit and PR titles. The authoritative list and
the path→module mapping live in the
[commit-and-pr](../skills/commit-and-pr/SKILL.md) skill.
