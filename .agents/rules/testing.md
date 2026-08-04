---
paths:
  - tests/**
  - '**/*_test.py'
  - '**/test_*.py'
---

# Testing Rules

`docs/contributing/testing_guide.md` covers scope, naming, placement, coverage, and
the commands. Step-by-step: [run-cpu-tests](../skills/run-cpu-tests/SKILL.md). Two
things here are enforcement facts rather than guidance.

## Placement is a commit gate, and it is the only one

`validate-structure` fails the commit unless every `test*.py` sits in
`tests/<module>/`, where `<module>` is a first-level directory of `verl_omni/`
(`agent_loop`, `models`, `pipelines`, `reward_loop`, `trainer`, `utils`, `workers`) or
one of `special_{e2e,sanity,standalone,distributed}`. A test at the `tests/` root
fails — three legacy root files are explicitly allowlisted.

Nested directories are fine (`tests/utils/reward_score/`). The hook checks nothing
else: not naming, not docstrings, not that the path mirrors the source. So the
guide's `_on_cpu.py` rule is *not* gated — a misnamed test commits cleanly and then
never runs.

## GPU tests are gated by scripts, not by `skipif`

`tests/gpu_smoke/` and `tests/npu_smoke/` hold **no pytest files** — only shell
scripts that invoke GPU tests living in the ordinary module directories
(`tests/workers/test_diffusers_fsdp_engine.py`, …), because `validate-structure`
forbids `test_*.py` under those directories. Exactly one file in the tree uses
`skipif(not torch.cuda.is_available())`. Keep GPU tests beside the CPU tests and let
the filename keep them out of the CPU job.
