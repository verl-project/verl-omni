---
name: add-pipeline
description: Router for adding a diffusion or omni pipeline to verl-omni. Classifies the task (new architecture vs new algorithm; policy-gradient vs direct-preference) and points at the authoritative guide under docs/contributing/. Use when integrating a new model, or a new model+algorithm pair such as flow-GRPO / DPO / NFT / DanceGRPO.
---

# Add a Pipeline

`docs/contributing/` owns the steps and the final checklist for every case below.
This skill routes you to the right guide and adds only what those guides leave
out. Open the guide — do not work from a remembered procedure.

## Step 0 — Classify, on two axes

Architecture and algorithm integration are orthogonal: one algorithm reaches any
number of architectures through the adapter pair, and vice versa.

**Adding an algorithm?**

| Trains from | Family | Guide |
| ----------- | ------ | ----- |
| reverse-trajectory logprob ratios + advantages (FlowGRPO, MixGRPO, DanceGRPO) | policy gradient | `integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md` |
| final samples, rewards, or chosen/rejected pairs (offline DPO, DiffusionNFT) | direct preference | `integrating_a_new_direct_preference_algorithm_for_diffusion_model.md` |

Read `## Classify the Algorithm First` at the top of the direct-preference guide
even if you land on the policy-gradient one: it also splits offline vs online and
maps each family to its trainer (`PolicyGradientRayTrainer` /
`DirectPreferenceRayTrainer`) and FSDP engine. That decides
`algorithm.trainer_type` and `algorithm.sample_source`, which nothing else in the
repo infers for you.

**Adding an architecture?**

| Model | Guide |
| ----- | ----- |
| diffusers text-to-image | `integrating_a_diffusion_model.md` — read first, the rest extend it |
| diffusers image-edit / I2I | `integrating_an_i2i_diffusion_model.md` |
| a `nn.Module` that diffusers cannot load | `integrating_a_non_diffusers_model.md` |
| multimodal autoregressive (omni) | `integrating_an_omni_model.md` |
| step execution on an adapter that already works | `integrating_a_stepwise_continuous_batching_model.md` |

Both at once? Architecture first — an algorithm can only be paired with an adapter
that already exists.

## Step 1 — Find the closest pair to copy

List the live registry rather than guessing at a template:

```bash
grep -rhoE '@[A-Za-z]+\.register\([^)]*\)' verl_omni/pipelines/*/[dv]*.py | sort -u
```

`DiffusionModelBase` rows are training adapters, `VllmOmniPipelineBase` rows are
rollout adapters; a pair with no rollout row reuses another package's pipeline.
Copy the closest package, then share rather than fork
([code-style](../../rules/code-style.md#reuse-over-duplication)).

## Step 2 — Work the guide's checklist literally

Every guide but the omni one ends in a `- [ ]` checklist (the omni guide ends in its
own `## Common pitfalls` instead). Work it item by item; between them the checklists
cover what is most often missed: the star-import in `verl_omni/pipelines/__init__.py`
(without it the adapter is invisible and nothing errors at import time), mirroring a
new pipeline config field in **both** `diffusion_rollout.yaml` and
`diffusion_model.yaml`, and wiring a smoke test into `tests/gpu_smoke/`.

## Step 3 — When the run is wrong rather than broken

`docs/contributing/common_pitfalls.md` is symptom-first: fp32 latent/scheduler
precision loss, RoPE sequence-length mismatch, per-request vs per-GPU SDE seeding.
Check it before debugging a reward curve that trains but diverges from diffusers.

## Step 4 — Test, then open the PR

Adapter-boundary CPU tests: [run-cpu-tests](../run-cpu-tests/SKILL.md).
Title, trailers, duplicate-work checks: [commit-and-pr](../commit-and-pr/SKILL.md).

<!--
MAINTAINER GUIDE — this file must stay a router. If you catch yourself restating a
guide's steps here, edit the guide instead. Update the tables when a guide is
added, renamed, or changes scope.
-->
