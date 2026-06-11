<div align="center">

# VeRL-Omni

### Easy, fast, and stable RL training for diffusion and omni-modality models

[![Docs](https://img.shields.io/badge/docs-Read%20the%20Docs-8A2BE2)](https://verl-omni.readthedocs.io/en/latest/index.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE) <a href="docs/assets/WeChat.jpg"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

`VeRL-Omni` is a general RL training framework focused on multimodal generative models, built on top of [`verl`](https://github.com/verl-project/verl).

It originated from the multi-modal generation RL effort in `verl`, and now has a dedicated home so it can evolve in a more focused way.

## News 🔥

- **[2026-06]** [DiffusionNFT](https://verl-omni.readthedocs.io/en/latest/algo/diffusionnft.html) and [Diffusion DPO](https://verl-omni.readthedocs.io/en/latest/algo/diffusion_dpo.html) are integrated with verified recipes on Qwen-Image/SD3.5. [Wan2.2](examples/dancegrpo_trainer/README.md) is now supported for video generation tasks.  

## Why `VeRL-Omni`

Multimodal generative RL training differs from text-only LLM RL not only in model structure, but also in I/O patterns, compute characteristics, and runtime bottlenecks. As this space grows, it deserves a dedicated training repository that can evolve quickly around its own constraints.

### Scope

`VeRL-Omni` targets RL post-training for three families of generative models:

1. **Diffusion generative models** for image, video, and audio — e.g., Qwen-Image, Wan2.2.
2. **Unified multimodal understanding + generation models** — e.g., BAGEL, HunyuanImage-3.0.
3. **Omni-modality models** that jointly handle text, image, audio, and video — e.g., Qwen3-Omni.

### What we focus on

- **Optimized rollout:** [`vLLM-Omni`](https://github.com/vllm-project/vllm-omni) as a rollout backend for high-throughput multimodal generation.
- **Flexible and async multi-reward serving:** Support for multi-reward serving (HPSv3, GenRM-OCR, UnifiedReward, etc.), [HTTP scorer](https://verl-omni.readthedocs.io/en/latest/start/http_scorer.html), and [asynchronous reward computation](https://verl-omni.readthedocs.io/en/latest/algo/async_reward.html) to overlap the rollout phase.
- **Modular training backends:** Selectable [VeOmni](https://github.com/ByteDance-Seed/VeOmni) and FSDP2 backends with combinable parallelism (USP/TP/DP) for distributed training.
- **Stability tools:** Improved diffusion RL stability with [rollout correction](https://verl-omni.readthedocs.io/en/latest/algo/rollout_correction.html) and deterministic rollout/reward/trainer.
- **End-to-end examples and benchmarks:** Validated recipes for co-located sync and fully-async RL on the model families above.
- **High training throughput:** On our reference Qwen-Image FlowGRPO setup, `VeRL-Omni` achieves **~25% higher end-to-end throughput** than the diffusers-based [`flow_grpo`](https://github.com/yifan123/flow_grpo) implementation, driven by `vLLM-Omni` rollout, FSDP2 trainer, overlapped reward computation (asynchronous), etc.


<div align="center">
  <img src="docs/assets/arch.png" alt="verl-omni architecture diagram" width="70%">
</div>


## Getting Started  🚀

Visit our documentation to learn more.

- [Installation](https://verl-omni.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl-omni.readthedocs.io/en/latest/start/flowgrpo_quickstart.html)

## Model and Algorithm Support 🎨

<table>
  <tr>
    <th>Model</th>
    <th>Category</th>
    <th>Modality</th>
    <th>Algorithm</th>
    <th>Status</th>
  </tr>
  <tr>
    <td rowspan="5">Qwen-Image</td>
    <td rowspan="5">Diffusion generator</td>
    <td rowspan="5">Text → Image</td>
    <td>FlowGRPO (+ CPS/SDE)</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>MixGRPO</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>GRPO-Guard</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>DiffusionNFT</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>DPO</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Wan2.2</td>
    <td>Diffusion generator</td>
    <td>Text → Video</td>
    <td>DanceGRPO</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>LTX2.3</td>
    <td>Diffusion generator</td>
    <td>Text → Video + Audio</td>
    <td>FlowGRPO</td>
    <td>WIP</td>
  </tr>
  <tr>
    <td>BAGEL</td>
    <td>Unified understand + gen</td>
    <td>Text + Image</td>
    <td>FlowGRPO</td>
    <td>WIP</td>
  </tr>
  <tr>
    <td rowspan="2">HunyuanImage-3.0</td>
    <td rowspan="2">Unified understand + gen</td>
    <td rowspan="2">Text + Image</td>
    <td>MixGRPO</td>
    <td>Planned</td>
  </tr>
  <tr>
    <td>SRPO</td>
    <td>Planned</td>
  </tr>
  <tr>
    <td>Qwen3-Omni-Thinker</td>
    <td>Omni-modality</td>
    <td>Text / Image / Video / Audio</td>
    <td>GSPO</td>
    <td>WIP</td>
  </tr>
  <tr>
    <td>SD3.5</td>
    <td>Diffusion generator</td>
    <td>Text → Image</td>
    <td>DPO</td>
    <td>✅</td>
  </tr>
</table>


## Ascend NPU Support 💠

`VeRL-Omni` now supports Ascend NPU. For instructions on how to install and get started with FlowGRPO training on Ascend NPU, please refer to our [Ascend NPU Quickstart Guide](https://verl-omni.readthedocs.io/en/latest/start/flowgrpo_quickstart_npu.html).


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
