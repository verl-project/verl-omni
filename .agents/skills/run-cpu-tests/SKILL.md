---
name: run-cpu-tests
description: How to write and run verl-omni CPU tests (test_*_on_cpu.py) that exercise adapters, rewards, and configs without a GPU or model weights. Use when adding tests, reproducing a failure locally, or producing the test evidence a PR body requires.
---

# Run & Write CPU Tests

`docs/contributing/testing_guide.md` is authoritative for the layer hierarchy, the
`*_on_cpu.py` naming rule, placement, coverage, the local `pytest` invocations, and
the steps for adding a test. Follow it. This skill adds what it does not cover.

## What CI does that the guide's local commands don't

- The CPU job exports `TORCH_COMPILE_DISABLE=1` and `TORCHINDUCTOR_DISABLE=1`
  (`.github/workflows/cpu_unit_tests.yml`). Set both locally when reproducing a
  failure that only CI sees.
- On pull requests the job triggers on `types: [labeled]` **and only when the label
  is `ci`** — a green checks page on an unlabelled PR means the tests never ran. The
  label is single-use: `drop-ci-labels.yml` removes it on every `synchronize`, so a
  new push does **not** re-run the job until you re-add `ci`.
- `tests/special_sanity/` runs as its own job; those files are `test_*.py`, so the
  CPU job's `python_files` override deliberately skips them.

## Idioms the guide leaves to the reader

**Configs** — construct normally:

```python
cfg = DiffusionLossConfig(loss_mode="flow_dppo")
```

Bypass `__init__` only when `__post_init__` does I/O (loading tokenizers, resolving
paths), as `DiffusionModelConfig` does:

```python
cfg = object.__new__(DiffusionModelConfig)
object.__setattr__(cfg, "architecture", "QwenImagePipeline")
object.__setattr__(cfg, "algorithm", "dpo")
```

`object.__setattr__` is needed because `BaseConfig` gates assignment through
`_mutable_fields`, not because the dataclass is frozen
([config.md](../../rules/config.md)). Reaching for this where plain construction
works is a review comment.

**Mocks and assertions** — `MagicMock` the transformer and assert on call args and
output shapes rather than on real model output. `TensorDict` batches usually carry
metadata, not pixels (`TensorDict({}, batch_size=2)` plus the fields under test).
Use `torch.testing.assert_close(a, b, rtol=..., atol=...)` for tensors and
`pytest.approx` for scalars — never `.equal()` on floats.

## Before opening a PR

Paste the command **and its output** into the PR body (mandatory —
[commit-and-pr](../commit-and-pr/SKILL.md)), then run `pre-commit run --all-files`.

<!--
MAINTAINER GUIDE — keep this to the delta over docs/contributing/testing_guide.md.
Update when cpu_unit_tests.yml changes its env or its label gate, or when the
config-construction idiom changes.
-->
