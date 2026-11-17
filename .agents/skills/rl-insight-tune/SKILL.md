---
name: rl-insight-tune
description: Use RL-Insight metrics, timelines, and hardware telemetry to quickly diagnose and tune verl and verl-omni experiments for performance, training quality, and memory.
---

# Tune Experiments with RL-Insight

## 1. Establish a baseline

- Read the launcher and resolved configuration. Identify trainer and rollout roles and their device allocations.
- Define the objective and acceptable trade-offs. Match the workload, evaluation settings, and run phases across comparisons.

## 2. Query telemetry

- Query Prometheus for metrics and Tempo for timeline spans. Use `experiment_name` and the same time window for both. Use the known run window when the experiment tag is unavailable.
- Summarize training, evaluation, rollout, queue, and hardware data locally. Check coverage for the expected roles and devices. See [querying telemetry](references/tuning.md).

## 3. Diagnose the bottleneck

- Use the [metric interpretation table](references/tuning.md#interpret-and-test) to align timeline phases with throughput, quality, and memory signals. Check what each timer measures and which phases overlap.
- When telemetry leaves multiple plausible causes, use the [verl-omni profiling skill](../profile/SKILL.md) to choose a profiler and capture the affected role. For Ascend NPU, use the [NPU profiling recipe](../../../docs/perf/profiler.md#7-ascend-npu-profiling-npu).

## 4. Test one hypothesis

- Use the verl and verl-omni tuning guides to choose a setting that the installed trainer or backend actually uses.
- State the expected effect and which metrics will confirm it. Record any settings that must change together.

## 5. Compare and decide

- Verify the setting took effect and check whether the predicted change exceeds run-to-run variation. Include startup and buffer fill when relevant.
- Compare throughput, evaluation quality or numerical consistency, and peak memory. Keep, revert, or mark the change inconclusive.

Record the config delta, measurement window, baseline and candidate results, decision, and evidence location.

<!--
MAINTAINER GUIDE — Recheck this workflow when RL-Insight storage, telemetry, or
the linked tuning guides change.
-->
