---
name: debug-rl-convergence
description: Router for diagnosing a verl-omni diffusion RL run (DiffusionNFT, FlowGRPO, DPO) that trains without errors but whose reward stays flat or degrades. Use when asked why a run is not converging, to tune ref_kl_coef / learning rate / guidance, or to design ablations and validation that can detect a real gain.
---

# Debug RL Convergence

`docs/contributing/tuning_diffusion_rl_convergence.md` owns the procedure. Open it
and work its sections in order — do not tune from a remembered recipe.

## Step 0 — Rule out numerical bugs

If rollout and training disagree numerically, or the run diverges from diffusers,
this is not a tuning problem: go to `docs/contributing/common_pitfalls.md` or the
[train-infer-consistency](../train-infer-consistency/SKILL.md) skill first.

## Step 1 — Gate, measure, classify

| Question | Guide section |
| -------- | ------------- |
| Can this reward improve on this data at all? | 1. Gate the task |
| Would validation detect a gain of a few hundredths? | 2. Make validation able to detect a change |
| Not moving, moving wrong, or moving only on training data? | 3. Classify the failure |

Report each answer with evidence (baseline mean, paired CI, metric trajectories)
before proposing a change.

## Step 2 — Ablate one variable, then look at images

Sections 4–5. Before launching, state the step time, the step budget, and the
gate that will stop the arm.

## Step 3 — Algorithm-specific defaults

Section 6 for DiffusionNFT (CFG-free training, anchor strength, fair baselines).

<!--
MAINTAINER GUIDE — this file must stay a router. Put new lessons in the guide,
with the evidence and its confidence, and update the table above if its sections
change.
-->
