---
name: rl-insight-tune
description: Diagnose and tune verl or verl-omni performance using RL-Insight training, inference, timeline, and hardware metrics. Use for slow steps, rollout waits, pipeline imbalance, or memory pressure.
---

# RL-Insight Performance Tuning

Find what limits training progress, test a targeted change, and verify the benefit.

## Principles

- **Optimize training progress.** Engine throughput and hardware utilization explain performance; success means improving the user's training objective.
- **Keep comparisons fair.** Preserve workload and training semantics, use comparable monitoring, and test one hypothesis at a time.
- **Follow evidence.** Align measurements by run, role, and time. Treat missing data as unknown and verify that proposed settings affect the installed implementation.

## Workflow

1. **Establish the baseline.** Read the launcher and resolved configuration. Identify trainer/rollout roles, device allocation, objective, constraints, and budget.
2. **Verify observation.** Enable the required RL-Insight and hardware collection. Check advancing steps, generation counters, and coverage of all expected devices. See [metrics and tuning](references/tuning.md).
3. **Diagnose.** Use the timeline to locate the limiting stage, then use engine, queue, and hardware data to explain it. Check timer meanings and overlapping work. Profile the affected role when metrics cannot distinguish causes.
4. **Test a hypothesis.** State the evidence, parameter change, predicted intermediate effect, and rollback conditions. Run a comparable experiment when execution is requested; otherwise propose it. Use the existing experiment controller when available.
5. **Decide.** Verify that the setting took effect, the predicted behavior changed, and training improved beyond observed variation. Check memory, output lengths, numerical health, and policy staleness. Include startup and buffer-fill costs; repeat uncertain comparisons. Keep, revert, or report inconclusive, then continue within the original budget.

Report baseline versus candidate results, configuration differences, decisions, limitations, and links to raw evidence. Follow the controller's record format when one is required; no additional schema is necessary for standalone analysis.

<!--
MAINTAINER GUIDE — Recheck metric names and tuning keys when RL-Insight or backend
integrations change. Keep this workflow short and details in the reference.
-->
