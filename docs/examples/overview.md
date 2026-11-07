(examples_overview)=
# Which example should I run?

Last updated: 10/06/2026

{doc}`../start/models` is the full catalogue: every supported model with its
architecture, trainers, example scripts, and hardware requirements. This page
is the task-oriented shortcut — start from what you want to do and follow the
first link that matches. Example pages linked as `examples/**` point into the
repository's `examples/` directory; the tables in {doc}`../start/models` are
kept up to date with every recipe.

## Text-to-image RL

| Goal | Start with |
|---|---|
| Validate the stack end-to-end on the smallest setup | [SD3.5 FlowGRPO quickstart](../start/flowgrpo_quickstart.md) — 3 GPUs, LoRA, OCR reward |
| Production image RL on a strong open model | [Qwen-Image FlowGRPO (LoRA, V1)](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/README.md) — see the [Qwen-Image notes](../start/models.md#qwen-image) |
| Compare policy-gradient variants on the same task | [FlowDPPO](flowdppo_trainer.md), [GRPO-Guard](grpoguard_trainer.md), [MixGRPO](mixgrpo_trainer.md), [DanceGRPO](dancegrpo_trainer.md), [FlowGRPO](flowgrpo_trainer.md) |
| Offline or non-policy-gradient objectives | [Diffusion-DPO](dpo_trainer.md), [DiffusionNFT](diffusionnft_trainer.md) |
| Latent-reward / reward-model research | [SD3.5 with DiNa latent reward model](flowgrpo_trainer_sd35_drm.md) |

## Image editing (i2i)

| Goal | Start with |
|---|---|
| Post-train an instruction-following image editor | [Qwen-Image-Edit FlowGRPO](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image_edit/README.md) — see the [Qwen-Image-Edit notes](../start/models.md#qwen-image-edit) |

## Video and audio generation

| Goal | Start with |
|---|---|
| Text-to-video RL | [Wan2.2-TI2V-5B DanceGRPO](https://github.com/verl-project/verl-omni/blob/main/examples/dancegrpo_trainer/README.md) — see the [Wan2.2 notes](../start/models.md#wan22-ti2v-5b) |
| Text-to-(video+audio) RL | [LTX-2.3 FlowGRPO](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/ltx2/README.md) or [MiniMax-H3 FlowGRPO](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/minimax_h3/README.md) — see the [LTX-2.3](../start/models.md#ltx-23) and [MiniMax-H3](../start/models.md#minimax-h3) notes |

## Unified understanding + generation

| Goal | Start with |
|---|---|
| RL on a model that both understands and generates images | [BAGEL FlowGRPO](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/bagel/README.md) — see the [BAGEL notes](../start/models.md#bagel) |

## Omni-modality (AR) and speech

| Goal | Start with |
|---|---|
| Post-train an omni-modality AR model (text/image/audio/video in) | [Qwen3-Omni thinker GSPO](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/README.md), also [DAPO](dapo_trainer.md) and [OPD distillation](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/README.md#mmk12-on-policy-distillation-opd) — see the [Qwen3-Omni notes](../start/models.md#qwen3-omni-30b-a3b-thinker) |
| Post-train a TTS model | [Qwen3-TTS GRPO](https://github.com/verl-project/verl-omni/blob/main/examples/grpo_trainer/qwen3_tts/README.md) — see the [Qwen3-TTS notes](../start/models.md#qwen3-tts-12hz-06b-base) |

## Before you launch

- Scripts named `*_v1.sh` run the V1 trainer (default since v0.3.0); see
  {doc}`../start/diffusion_v1` for the trainer guide and the v0 migration
  notes.
- Perf tuning after your first successful run: start at the
  {doc}`../perf/tuning_guide` — section 5 there covers omni-modality runs.
- Hardware requirements per recipe live in {doc}`../start/models`; nothing on
  this page changes them.
