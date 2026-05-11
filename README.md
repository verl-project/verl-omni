<div align="center">

# VeRL-Omni

### Easy, fast, and stable RL training for diffusion and omni-modality models

[![Docs](https://img.shields.io/badge/docs-Read%20the%20Docs-8A2BE2)](https://verl-omni.readthedocs.io/en/latest/index.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE) <a href="docs/assets/WeChat.jpg"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

`VeRL-Omni` is a general RL training framework focused on multimodal generative models, built on top of [`verl`](https://github.com/verl-project/verl).

It originated from the multi-modal generation RL effort in `verl`, and now has a dedicated home so it can evolve in a more focused way.

## Why `VeRL-Omni`

Multimodal generative RL training differs from text-only LLM RL not only in model structure, but also in I/O patterns, compute characteristics, and runtime bottlenecks. As this space grows, it deserves a dedicated training repository that can evolve quickly around its own constraints.

### Scope

`VeRL-Omni` targets RL post-training for three families of generative models:

1. **Diffusion generative models** for image, video, and audio — e.g., Qwen-Image, Wan2.2.
2. **Unified multimodal understanding + generation models** — e.g., BAGEL, HunyuanImage-3.0.
3. **Omni-modality models** that jointly handle text, image, audio, and video — e.g., Qwen3-Omni.

### What we focus on

- **Specialized rollout** via [`vLLM-Omni`](https://github.com/vllm-project/vllm-omni) for high-throughput diffusion and multimodal generation.
- **Flexible reward pipelines** spanning rule-based rewards, model-based rewards, and multimodal reward computation.
- **Modular training backends** that plug into existing parallelism (FSDP, USP) and other optimizations rather than rebuilding the stack from scratch.
- **End-to-end examples and benchmarks** validating co-located sync and fully-async RL on the model families above.
- **High training throughput** — on our reference Qwen-Image FlowGRPO setup, `VeRL-Omni` achieves **~25% higher end-to-end throughput** than the diffusers-based [`flow_grpo`](https://github.com/yifan123/flow_grpo) implementation, driven by `vLLM-Omni` rollout, FSDP training, and overlapped reward computation (asynchronous).


<div align="center">
  <img src="docs/assets/arch.png" alt="verl-omni architecture diagram" width="70%">
</div>



## Getting Started  🚀

Visit our documentation to learn more.

- [Installation](https://verl-omni.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl-omni.readthedocs.io/en/latest/start/flowgrpo_quickstart.html)

## Model and Algorithm Support 🎨

| Model              | Category                 | Modality           | Algorithm | Status |
|--------------------|--------------------------|--------------------|-----------|--------|
| Qwen-Image         | Diffusion generator      | Text → Image       | FlowGRPO  | ✅     |
| Wan2.2             | Diffusion generator      | Text → Video       | DanceGRPO | WIP    |
| BAGEL              | Unified understand + gen | Text + Image       | FlowGRPO  | WIP    |
| HunyuanImage-3.0   | Unified understand + gen | Text + Image       | MixGRPO/SRPO       | Planned |
| Qwen3-Omni-Thinker | Omni-modality            | Text / Image / Video / Audio | GSPO    | WIP    |

## Roadmap 🗺

Future work is tracked here:

- [RFC: Multi-modal Generation RL 2026Q2 Roadmap](https://github.com/verl-project/verl/issues/5755)

## Contributing 🤝

Contributions are welcome.

See the [contribution guide](CONTRIBUTING.md).

## Acknowledgement 🌟

`verl-omni` builds on the engineering foundations developed in [`verl`](https://github.com/verl-project/verl) and is closely aligned with multimodal inference systems such as [`vLLM-Omni`](https://github.com/vllm-project/vllm-omni).

## Citation 📚

If you find the project helpful, please cite:

```bibtex
@misc{verlomni_github,
  title        = {{VeRL-Omni: Easy, Fast, and Stable RL Training for Diffusion and Omni-Modality Models}},
  author       = {Yongxiang Huang and Cheung Kawai and Jingan Zhou and Yingshu Chen and {openYuanrong Team} and Xibin Wu},
  year         = {2026},
  howpublished = {\url{https://github.com/verl-project/verl-omni}},
  urldate      = {2026-04-28}
}
```
