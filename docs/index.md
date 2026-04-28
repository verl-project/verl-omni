# Welcome to VeRL-Omni's documentation!

Last updated: 04/23/2026

[VeRL-Omni](https://github.com/verl-project/verl-omni) is a general RL training framework focused on diffusion and omni-modality generative models. It starts from the multimodal generation RL work incubated in [verl](https://github.com/verl-project/verl) and provides a dedicated home for building and evolving this stack in a more focused way.

Key capabilities:

- **Specialized rollout support** via [vLLM-Omni](https://github.com/vllm-project/vllm-omni) for concurrent diffusion and multimodal generation.
- **Efficient diffusion RL training** for image and other non-autoregressive models.
- **Flexible reward pipelines** spanning rule-based rewards, model-based rewards, and multimodal reward computation.
- **Modular training backends** that integrate various parallelism strategies (FSDP, USP) without rebuilding the full stack.

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
algo/performance.md
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
:caption: Contributing

contributing/editing-agent-instructions.md
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

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file, which usually corresponds to a `hydra` default config (e.g. `ppo_trainer`, `ppo_megatron_trainer`, `sft_trainer`, etc).
2. Add related path patterns to the `paths` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).
