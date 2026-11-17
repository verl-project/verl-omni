# Querying Metrics and Choosing Experiments

## Query stored telemetry

Use `experiment_name` to scope both Prometheus metrics and Tempo spans. Apply the same start and end times to both queries, especially when a name is reused. If the label or attribute is missing, use the known run window and verify the source by endpoint, rank, or device where available. Treat overlapping runs that cannot be distinguished as ambiguous.

Use the Prometheus TSDB on the RL-Insight monitoring host. Managed deployments default to `~/.rl-insight/data/prometheus`. Confirm the running Prometheus process's `--storage.tsdb.path` or the deployment configuration (`prometheus.data_dir`, then `server.data_dir` with a `prometheus` subdirectory). For external Prometheus, use its configured database.

Use PromQL through a database client for live queries. Include `experiment_name="my-run"` in the series selector. For direct offline reads, use the matching Prometheus release's `promtool` with a consistent snapshot or a stopped copy. Use the metric catalog or dashboard query to discover series for the subsystem under investigation.

```bash
mkdir -p evidence
promtool tsdb dump --sandbox-dir-root=evidence \
  --match="$SERIES_SELECTOR" --min-time="$START_MS" --max-time="$END_MS" \
  "$TSDB_SNAPSHOT" > evidence/samples.txt
```

`SERIES_SELECTOR` is a Prometheus series selector and the bounds are Unix milliseconds. Check the snapshot covers the interval, including recent head data when needed.

Query Tempo with TraceQL, for example `{ span.experiment_name = "my-run" }`, over the same window. Use the matching spans to summarize phase durations, overlap, and waits by role or lane. Align those summaries with Prometheus trends before drawing conclusions. Preserve queries, windows, and raw evidence on disk and pass only compact results into the agent context.

## Interpret and test

Start with the signals that answer your question. Trainer names below are RL-Insight catalog examples without the `rl_insight_monitor_` prefix. Confirm availability and units in the deployed backend. For non-token workloads, use trajectories or samples as the unit of work.

| Question | Signals to check | How to interpret them |
| --- | --- | --- |
| Is training making progress faster? | `timing_s_step`, `perf_total_num_tokens`, completed steps, consumed trajectories, whole-run time | Divide total consumed work by elapsed time over a matched window. Compare step time and work per step together. Report the device count and whether startup and drain are included. The catalog's `perf_throughput` is already normalized per device. |
| Which phase limits step time? | Tempo spans and trainer timers for rollout, reward, log-probability, actor update, and weight sync | Look for delays on the critical path and late ranks or replicas. Compare overlap before adding phase times. A long generation window can include reward or wait time. |
| Is rollout supply or engine capacity limiting progress? | Per-replica `vllm:num_requests_running`, `vllm:num_requests_waiting`, generated tokens, and prefill, decode, or queue latency | Trainer waits with lightly loaded replicas suggest tracing request supply, group completion, and reward. A growing request backlog with busy replicas suggests engine pressure. Use latency tails and sequence lengths to narrow the cause. |
| Where does the async pipeline stall? | Trainer wait spans, TransferQueue occupancy and flow, rollout completions, and weight sync spans | Check whether the trainer lacks complete groups or work accumulates upstream. Align production and consumption with queue trends and verify whether a gain persists after buffer fill. |
| Is actor compute, communication, or the host limiting progress? | Actor phase time, per-device compute and bandwidth, link activity, host CPU and I/O, and `perf_mfu_actor` where supported | Align samples with the active phase and compare ranks. Low device utilization alone leaves several possible causes. Compare MFU only with compatible FLOP estimates and device peak assumptions. |
| Where is memory pressure coming from? | Per-device HBM or VRAM, allocator peaks, `vllm:kv_cache_usage_perc` for AR rollout, preemptions, and OOM logs | Compare peaks by role and phase, including role transitions. Rising KV pressure with preemption suggests a rollout capacity issue. Cross-check hardware samples with allocator peaks because sampling can miss brief spikes. |
| Did the change affect training quality? | Matched evaluation scores, reward, loss, KL or clip fraction where applicable, response length and truncation, and async staleness | Check whether the workload or policy age shifted along with throughput. Short comparisons can expose regressions. Assess convergence over a representative training window. |

Use reset-aware counter increments, direct gauge values, and histogram bucket rates. Align series and spans by run, role, and time, and record coverage gaps.

Use the limiting stage to narrow the search in the tuning guides. Inspect the resolved configuration and implementation for settings that can change that stage or its inputs. Prioritize candidates by expected impact, evidence, and trade-offs. Define the expected mechanism and the user-facing outcome before running a comparison.

## References

- [RL-Insight setup and exporter coverage](../../../../docs/start/rl_insight.md)
- [RL-Insight metric catalog](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/metric-name-catalog.md)
- [RL-Insight hardware monitoring](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/hardware/index.md)
- [verl-omni performance tuning](../../../../docs/perf/tuning_guide.md)
- [verl performance tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
