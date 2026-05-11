# Welcome to VeRL-Omni's documentation!

Last updated: 04/30/2026

[VeRL-Omni](https://github.com/verl-project/verl-omni) is a general RL training framework focused on multimodal generative models, built on top of [verl](https://github.com/verl-project/verl). It originated from the multi-modal generation RL effort in `verl`, and now has a dedicated home so it can evolve in a more focused way.

## Scope

VeRL-Omni targets RL post-training for three families of generative models:

1. **Diffusion generative models** for image, video, and audio — e.g., Qwen-Image, Wan2.2.
2. **Unified multimodal understanding + generation models** — e.g., BAGEL, HunyuanImage-3.0.
3. **Omni-modality models** that jointly handle text, image, audio, and video — e.g., Qwen3-Omni.

## Key capabilities

- **Specialized rollout** via [vLLM-Omni](https://github.com/vllm-project/vllm-omni) for high-throughput diffusion and multimodal generation.
- **Flexible reward pipelines** spanning rule-based rewards, model-based rewards, and multimodal reward computation.
- **Modular training backends** that plug into existing parallelism (FSDP, USP) and other optimizations rather than rebuilding the stack from scratch.
- **End-to-end examples and benchmarks** validating co-located sync and fully-async RL on the model families above.
- **High training throughput** — on our reference Qwen-Image FlowGRPO setup, VeRL-Omni achieves **up to ~25% higher end-to-end throughput** than the diffusers-based [`flow_grpo`](https://github.com/yifan123/flow_grpo) reference implementation, driven by vLLM-Omni rollout, FSDP/USP training, and asynchronous reward computation on a dedicated GPU pool.

```{toctree}
:maxdepth: 2
:caption: Getting Started

start/install.md
start/flowgrpo_quickstart.md
start/metrics.md
```

```{toctree}
:maxdepth: 1
:caption: Algorithms

algo/flowgrpo.md
algo/mixgrpo.md
algo/performance.md
```

```{toctree}
:maxdepth: 1
:caption: Performance

perf/profiler.md
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/trainer.rst
api/workers.rst
api/rollout.rst
api/reward.rst
api/pipelines.rst
api/utils.rst
```

```{toctree}
:maxdepth: 1
:caption: Developer Guide

contributing/editing-agent-instructions.md
contributing/integrating_a_diffusion_model.md
contributing/integrating_a_new_algorithm_for_diffusion_model.md
```

## Contribution

VeRL-Omni is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on [GitHub](https://github.com/verl-project/verl-omni) for discussions.

See the [2026 Q2 roadmap](https://github.com/verl-project/verl/issues/5755) for planned work.

### Code Linting and Formatting

We use pre-commit to help improve code quality. To initialize pre-commit, run:

```bash
pip install pre-commit
pre-commit install
```

To resolve CI errors locally, you can also manually run pre-commit by:

```bash
pre-commit run
```

### Adding CI tests

If possible, please add CI test(s) for your new feature. Pick the most relevant workflow from [`.github/workflows/`](https://github.com/verl-project/verl-omni/tree/main/.github/workflows):

| Workflow | When to use |
|---|---|
| `cpu_unit_tests.yml` | New tests that run without a GPU (file name must end with `_on_cpu.py`) |
| `gpu_smoke.yml` | GPU-requiring tests for trainer, worker, rollout, or agent-loop changes |
| `gpu_smoke_verl_latest.yml` | Same as above, but pinned against the latest `verl` main (for upstream compatibility) |
| `sanity.yml` | Static / import-level checks under `tests/special_sanity/` |

Steps:

1. Place your test file in the appropriate directory under `tests/` (e.g. `tests/trainer/`, `tests/workers/`, `tests/agent_loop/`).
2. Open the chosen workflow yml and add any missing path patterns to its `paths` section so the workflow triggers on your changes.
3. Keep the test as lightweight as possible — use small models, reduced steps, and CPU where feasible (see existing `*_on_cpu.py` scripts for examples).
