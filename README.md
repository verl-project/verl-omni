<div align="center">

# VeRL-Omni

### Easy, fast, and stable RL training for diffusion and omni-modality models

</div>

`VeRL-Omni` is a general RL training framework focused on diffusion and omni-modality generative models, built on top of [`verl`](https://github.com/verl-project/verl).

It starts from the multi-modal generation RL work incubated in `verl`, and provides a dedicated home for building and evolving this stack in a more focused way.

## Why `VeRL-Omni` 

Diffusion and omni-modality RL training differs from text-only LLM RL not only in model structure, but also in I/O patterns, compute characteristics, and runtime bottlenecks.

As diffusion and omni-modality model training expands, it is useful to have a dedicated training repository that can evolve faster around:

- **Specialized rollout support** including `vLLM-Omni` for concurrent diffusion and multimodal generation.
- **Efficient diffusion RL training** for image and other non-autoregressive models.
- **Omni-modality support** for text, image, video, audio, and unified multimodal generation workflows.
- **Flexible reward pipelines** spanning rule-based rewards, model-based rewards, and multimodal reward computation.
- **Modular training backends** that can easily integrate various parallism (FSDP, USP) and other optimizations instead of rebuilding the full stack from scratch.
- **E2E examples and benchmarks** for validating high-efficiency e2e RL training on model families such as QwenImage, Qwen-Omni, and BAGEL, in co-located sync or fully-async mode. 




## Roadmap

Future work is tracked here:

- [RFC: Multi-modal Generation RL 2026Q2 Roadmap](https://github.com/verl-project/verl/issues/5755)

## Getting Started

Visit our documentation to learn more.

- [Installation](docs/start/install.md)
- [Quickstart](docs/start/flowgrpo_quickstart.md)

## Supported Models & Algorithms

| Model | Algorithm | State |
|---|---|---|
| Qwen-Image | FlowGRPO | ✅ |
| Wan2.2 | DanceGRPO | WIP |
| BAGEL | FlowGRPO | WIP |
| Qwen3-Omni-Thinker | GRPO | WIP |

## Contributing

Contributions are welcome.

See the [contribution guide](CONTRIBUTING.md).

## Acknowledgement

`verl-omni` builds on the engineering foundations developed in [`verl`](https://github.com/verl-project/verl) and is closely aligned with multimodal inference systems such as [`vLLM-Omni`](https://github.com/vllm-project/vllm-omni).

## Citation

TBD
