---
paths:
  - verl_omni/trainer/**
  - verl_omni/workers/**
  - verl_omni/agent_loop/**
---

# Trainer / Worker Rules

Invariants for the shared training core — `verl_omni/trainer/`, `verl_omni/workers/`,
`verl_omni/agent_loop/`. How to integrate a specific model or algorithm is documented
per case in `docs/contributing/` — [add-pipeline](../skills/add-pipeline/SKILL.md)
routes you to the right guide. This file holds only the invariants that hold across
all of them.

## Mental model

verl-omni **extends pinned upstream dependencies** (verl, vllm-omni); it is not a
fork of either, and neither is the code you are about to edit. Everything under the
three directories above is the **model- and algorithm-agnostic core**; model and
algorithm specifics live in `verl_omni/pipelines/<model>_<algo>/` packages
([pipelines.md](pipelines.md)). Data flows trainer → verl's `AgentLoopManager` /
`RewardLoopManager` → rollout adapter → pipeline adapter, as `DataProto` /
`TensorDict`. The v1 paths (`main_diffusion_v1.py`; online-RL mode in `main_omni.py`)
are the default; the v0 diffusion path (`main_diffusion.py`) is deprecated.

| You are adding                 | Register at                                                              | Dispatch key                     |
| ------------------------------ | ------------------------------------------------------------------------ | -------------------------------- |
| Loss / advantage estimator     | `register_diffusion_loss` / `register_diffusion_adv_est` in `trainer/diffusion/diffusion_algos.py`; `register_omni_loss` in `trainer/omni/omni_algos.py` | name, e.g. `flow_grpo` / `dpo` |
| Model or rollout adapter       | `DiffusionModelBase.register` / `VllmOmniPipelineBase.register` / `OmniModelBase.register` in `pipelines/model_base.py` | (architecture, algorithm) / (architecture, stage) |
| Agent-loop behavior            | the agent loop stays model-agnostic — prefer fixing or extending the existing worker(s) in `agent_loop/` over registering a new one | — |
| Rollout engine                 | **vllm-omni only** — ask the maintainer before supporting another engine  | —                                |

## Invariants

- **The shared core stays model- and algorithm-agnostic.** No model names, no
  model-specific branches or config in `trainer/`, `workers/`, or `agent_loop/`;
  that specificity belongs in the model's pipeline package or behind a hook. Do not
  touch `workers/engine_workers.py` unless the change is a concern common to all
  algorithms.
- **Extend by registering, not by forking.** A new algorithm registers a loss /
  advantage estimator plus its pipeline adapter; it does not copy or subclass a
  trainer to specialize it for one model. Trainer classes in-tree exist for the modes
  the core supports — the v1 "mode hooks" in `trainer/diffusion/v1/trainer_base.py`
  are the internal sync / separate-async mechanism, not a public extension API. A
  change to shared code must say which models **and** algorithms exercise it.
- **A genuine cross-cutting need is met by refactoring, not specializing — and the
  hook system stays minimal.** Reach first for an existing registry or adapter;
  only if none fits, add a narrow hook (hooks live in `verl_omni/pipelines/`)
  sized to the concrete need, and dispatch to it. Never branch on the model or
  overwrite a method from a pipeline, and never add hooks speculatively — no
  interface, dataclass, or registry ahead of its first real caller.
- **Follow verl where the core mirrors it** — naming, Hydra/OmegaConf config,
  forward loops, checkpoint utils. Check verl's equivalent before inventing
  structure.
