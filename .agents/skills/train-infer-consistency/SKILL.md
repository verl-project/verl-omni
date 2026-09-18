---
name: train-infer-consistency
description: Route verl-omni training/inference consistency checks through MindStudio's MSProbe collection and root-cause analysis skills. Use when collecting paired rollout/actor dumps or investigating numerical differences in diffusion or omni models.
---

# Training/Inference Consistency

The MindStudio skills below own collection and analysis. Read them in order;
this skill connects their inputs and outputs without repeating their procedures.

| Stage | Skill |
| --- | --- |
| Collect dumps | [verl-omni-msprobe-dump](https://gitcode.com/Ascend/msagent/tree/master/skills/accuracy/verl-omni-msprobe-dump) |
| Analyze differences | [rl-consistency-analysis](https://gitcode.com/Ascend/msagent/tree/master/skills/accuracy/rl-consistency-analysis) |

## Prerequisite — MSProbe

MSProbe (`mindstudio-probe`) is required for data collection. If missing, identify
the Python environment used for the user's verl-omni task and install it using
that environment's package manager and workflow (e.g., uv or pip).

## Step 0 — Screen a previous run, if provided

If the user provides an experiment directory or logs, check the saved config for
`calculate_log_probs=true` and inspect that run's `rollout_corr/*` metrics for
initial evidence of differences. Record the source and affected steps/timesteps.
Missing metrics or bypassed actor log-prob recomputation is inconclusive.
Without previous artifacts, proceed directly to collection.

## Step 1 — Load the skills

Prefer installed skills or an existing local msagent checkout. Read each
`SKILL.md`; if missing, fetch the complete skill directory and referenced resources
from the linked repository, preserving `scripts/`, `references/`, and relative
paths. Record the source/version used. If unavailable, report what is missing
rather than reconstructing the procedure from memory.

## Step 2 — Collect paired dumps

Follow the collection skill using the user's launch script and current verl-omni
source. Collect full tensor data for both sides within the diagnostic window,
using Step 0's findings, if available, to guide reproduction and fine-grained
analysis. Establish sample pairing through correlation logs, not aggregate
metrics. Produce the diagnostic wrapper, both dumps, and correlation logs.
Existing artifacts may be reused after passing that skill's checks; if either
side is missing, fix collection and rerun.

## Step 3 — Pair and analyze

Confirm the dumps represent the same sample and computation, with comparable
inputs and weights. Pass the paired dumps, correlation evidence, and layout
differences to the analysis skill. Follow its module-mapping and script workflow,
then interpret the results against current source. Statistics support screening;
elementwise conclusions require tensor evidence.

## Step 4 — Deliver the report

Provide actual paths to the diagnostic script, dumps, module mapping, and
`output_5_root_cause_report.md`. Include the screening metrics and diagnostic
config changes. Explain the pairing, evidence, limitations, and
next steps. If execution is unavailable, complete the feasible integration work
and identify unverified steps without claiming collection or analysis succeeded.

<!--
MAINTAINER GUIDE — Keep this skill a router. Collection and analysis procedures
belong to the linked MindStudio skills. Recheck links and artifact handoff when
their directory layout or output contract changes.
-->
