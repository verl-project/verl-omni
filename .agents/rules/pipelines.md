---
paths:
  - verl_omni/pipelines/**
---

# Pipeline Rules

A pipeline is one **(architecture, algorithm)** integration under
`verl_omni/pipelines/<model>_<algorithm>/`. How to add one is documented per case in
`docs/contributing/`; [add-pipeline](../skills/add-pipeline/SKILL.md) routes you to
the right guide. This file holds only the invariants that hold across all of them.

## Dispatch is by registry key, never by import

`model_base.py` keys `DiffusionModelBase`, `DiffusionI2IModelBase` and
`VllmOmniPipelineBase` on `(architecture, algorithm)`, `OmniModelBase` on
`(architecture, stage)`, and `OmniRolloutPipelineBase` on `model_type`. Nothing
imports an adapter by name, so:

- **Registration is an import side effect.** The decorator runs only if the
  subpackage is imported, and the only importer is `verl_omni/pipelines/__init__.py`.
  A directory missing from there silently does not exist — no import-time error.
- `architecture` is auto-detected from `model_index.json`'s `_class_name`;
  `algorithm` comes from `DiffusionModelConfig.algorithm`. Neither is passed
  explicitly at the call site.
- A miss raises `NotImplementedError` listing every registered key. Read that list
  before assuming the adapter is broken — usually it is just not registered.
- Out-of-tree adapters register through `DiffusionModelConfig.external_lib`; adding
  one needs no fork.

## Training adapters are stateless, rollout adapters are not

`DiffusionModelBase` / `OmniModelBase` subclasses are **never instantiated** — every
method is a `@classmethod` or `@staticmethod` and per-run state lives in the config
objects passed in. Do not add `__init__` or instance attributes; thread state
through arguments.

Rollout adapters are the exception: they subclass a concrete vllm_omni pipeline and
**are** instantiated (`__init__(self, *, od_config, prefix="")`).

`register` is per-subclass, so subclassing a registered adapter to override one hook
leaves the parent's key intact — which is why inheriting beats copying here
([code-style](code-style.md#reuse-over-duplication)).

## Tunables and imports

- Tunables belong in the config dataclasses under
  `verl_omni/workers/config/diffusion/`, not module-level constants
  ([config.md](config.md)).
- Import diffusers / vllm_omni lazily inside methods so the adapter stays importable
  on CPU; every adapter's `*_on_cpu.py` test depends on this
  ([testing.md](testing.md)).
