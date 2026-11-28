# PR Lint Checks

Last updated: 11/28/2026.

Apply the `ready-for-ci` label to run the `Lint` workflow
(`.github/workflows/pr-lint.yml`) and `Protect CI checks`
(`.github/workflows/protect-ci-checks.yml`) as part of full PR CI (single-use;
`drop-ci-labels.yml` removes labels containing `ci` on push), or apply the
persistent `auto-lint` label to run them and re-run them automatically on every
subsequent push (`auto-lint` stays attached across pushes). Labeled GPU and NPU
workflows also call `pr-lint.yml` directly (bypassing the label gate) before
allocating accelerator runners. The workflow runs repository Ruff lint/format,
fast sanity hooks (`check-license`, `validate-structure`,
`check-device-api-usage`, `check-dataproto-usage`, `check-naming-conventions`,
`compileall`), CI-policy enforcement (`check_ci_policy.py`), and incremental
`check-pr-lint`.

## Running Locally

```bash
# Check staged changes (reads the Git index without importing or running code)
uvx pre-commit run check-pr-lint

# Run checker regression suites
uv run --no-project --python 3.12 --with pytest --with pyyaml --with ruff==0.12.2 pytest -q \
  tests/special_sanity/test_ci_policy.py tests/special_sanity/test_pr_lint.py \
  tests/special_sanity/test_ast_rules.py tests/special_sanity/test_comments.py \
  tests/special_sanity/test_example_recipes.py
```

## Rule Catalog

The checkers live under `tests/special_sanity/` (`check_pr_lint.py`,
`check_ast_rules.py`, `check_comments.py`, `check_example_recipes.py`,
`check_ci_policy.py`).

### Python Architecture & Contract Rules (`VO001`–`VO019`)

| Codes | Scope | Enforced Contract |
| --- | --- | --- |
| `VO001`, `VO008` | Shared trainer, worker, agent loop, and pipeline contract modules | Extend via registries and shared interfaces rather than importing concrete pipelines or dispatching on model-name literals. |
| `VO002` | Rollout workers and adapters | Depend on shared rollout contracts instead of training engine internals. |
| `VO003` | `verl_omni/` | Reuse public APIs across packages and sibling reward scorers; keep private `_` helpers within their owning package (`reward_utils` stays shared). |
| `VO004`, `VO005` | Pipeline helpers and Qwen-Image rollout adapters | Use explicit sampling seeds rather than hashing request IDs; compute padded Qwen-Image RoPE lengths with `txt_seq_lens_from_embeds()`. |
| `VO006`, `VO007`, `VO013`, `VO017` | `tests/` | Check or propagate `pytest.main()` exit status; name CPU tests `test_*_on_cpu.py`; use GPU topology helpers instead of bare `torch.cuda.is_available()` skips; assert behavior directly rather than grepping `verl_omni/` source text. |
| `VO009`, `VO010` | `verl_omni/pipelines/` | Keep training and omni rollout adapters stateless (`@classmethod`/`@staticmethod`); import pipeline packages and registered adapter modules from their `__init__.py` entrypoints. |
| `VO011` | Config dataclasses | Inherit `BaseConfig` (or a subclass), avoid `frozen=True` and `Literal[...]`, provide defaults (or `omegaconf.MISSING`, never `dataclasses.MISSING`), unconditionally call `super().__post_init__()` in subclasses, and export in `__all__`. |
| `VO012`, `VO016` | Reward managers and scorers | Keep HTTP client imports in scorers rather than reward managers; propagate `ArtifactContractError` before broad fallback handlers. |
| `VO014` | `verl_omni/`, `tests/`, `examples/` | Pass local model and data paths via configuration or test fixtures instead of hardcoded `/home/`, `/Users/`, `/user/`, `/nfs/`, `/mnt/`, or `/root/` literals. |
| `VO015`, `VO018` | Public APIs (`__all__`, registered classes, shared pipeline contracts, `compute_score*`) | Require concise contract docstrings (`VO015`) and explicit parameter/return type annotations (`VO018`). |
| `VO019` | Changed `verl_omni/` modules | Remove unused private or unexported top-level functions, wire a caller, or export them in `__all__`. |

### Isolated Ruff Rules

`check_pr_lint.py` also runs Ruff with isolated settings and `--ignore-noqa` so
PR-local config cannot silence checks:

- **Production (`verl_omni/`)**: catches unused/redefined names, complexity > 15
  (`C901`), silent exception suppression (`S110`, `S112`, `E722`, `B012`),
  mutable defaults (`B006`), loop-variable capture (`B023`), invalid `__all__`
  exports, unchecked `subprocess.run` (`PLW1510`), unawaited/discarded async
  tasks (`RUF006`), `open()` without context managers (`SIM115`), `eval`/`shell=True`
  (`S307`, `S602`), `breakpoint()` (`T100`), commented-out code (`ERA001`), and
  new `global` statements outside `verl_omni/utils/reward_score/` (`PLW0603`).
- **Tests (`tests/`)**: enforces `T100` and `PT015` (use `pytest.fail()` for
  intentional failures) alongside repository Ruff checks.

### Comment, Docstring & Documentation Rules (`CM001`–`CM005`)

- `CM001`–`CM003`: reject edit/step narration, decorative banners, and echo
  comments that merely restate the next line of code.
- `CM004`: reject boilerplate headers (`Necessity:`/`Motivation:`), filler
  phrases, and trivial private docstrings that only repeat the signature.
- `CM005`: reject PR, RFC, and issue tracker references in comments, docstrings,
  shell recipes, and documentation; make code and docs self-explanatory.

### Example Recipe Rules (`EX001`–`EX009`)

- `EX001`, `EX005`, `EX002`: require valid `bash -n` syntax, LF endings, no
  trailing spaces after `\` continuations, `#!/usr/bin/env bash` with `set -e`
  and `set -o pipefail`, and trailing double-quoted `"$@"` override forwarding
  in `run_*.sh`.
- `EX003`, `EX004`, `EX006`: disallow committed example media, `README_*.md`
  variants, `dataprocess/`, duplicate nested directories, and `.venv/bin/python`.
- `EX007`, `EX008`, `EX009`: catch orphaned/duplicate Hydra overrides, hardcoded
  secrets/device lists/Ascend `source` paths/private IPs/`cd /...`/host-wide
  `pkill`/`pip install`, and require accelerator variant scripts (`_npu`, `_mlu`,
  `_amd`, `_rocm`) to delegate to an existing sibling GPU recipe.

## Baseline & Testing Scope

`check_pr_lint.py` compares the PR head against the merge base so legacy issues
remain quiet while newly introduced findings fail. Static rules complement, and
do not replace, behavioral tests: validate boundary contracts in L1 CPU tests
and numerical/rollout parity in L2 smoke tests (see
[Testing Guide](testing_guide.md)).
