# Metrics and Tuning

Discover actual metric names, types, labels, and units before querying. Examples below are version-dependent. Filter to this run's endpoints and devices; compare the same role and interval.

## Collect

Reuse healthy RL-Insight services. For supported verl-omni versions, include `rl_insight` in `trainer.logger`, set `actor_rollout_ref.rollout.disable_log_stats=false`, and enable TransferQueue metrics when used. Configure hardware exporters separately; the training logger does not start them.

Inspect Prometheus directly and preserve the responses:

```bash
mkdir -p evidence
curl -fsS "$PROMETHEUS_URL/api/v1/targets" > evidence/targets.json
curl -fsS "$PROMETHEUS_URL/api/v1/label/__name__/values" > evidence/names.json
curl -fsS "$PROMETHEUS_URL/api/v1/metadata" > evidence/metadata.json
curl -fsSG "$PROMETHEUS_URL/api/v1/query_range" \
  --data-urlencode "query=$RUN_SCOPED_PROMQL" \
  --data-urlencode "start=$START" --data-urlencode "end=$END" \
  --data-urlencode "step=10s" > evidence/window.json
```

Check target coverage, metric definitions, and API response status. HTTP success alone does not prove useful observation.

## Inspect

| Area | Measurements | Interpretation |
|---|---|---|
| Training | Unique steps, consumed tokens/trajectories, step and wall time, phase/wait timers | Optimize consumed work. Async timers may represent waiting; overlapping times are not additive. |
| Engine | Running/waiting requests, generation counters, KV usage, preemptions, prefill/decode/TTFT histograms | Inspect each replica during actual generation. Production can include unconsumed work; gauges can remain stale after sleep. |
| Pipeline | Queue trends, data/reward latency, sleep/wake/sync timeline, policy age | Distinguish missing supply, group tails, transfers, and coordination. Progress counters are not automatically queue lengths. |
| Hardware | Per-device memory, compute/bandwidth activity, frequency, power/temperature, links; host CPU/I/O | Align with role activity. Missing data is not zero; low utilization does not identify a kernel bottleneck. |

Use reset-aware increments/rates for Counters, direct values for Gauges, and bucket rates for histogram quantiles. Deduplicate step observations and duplicate scrapes. Report sampled memory peaks as sampled, not allocator peaks. Report MFU as N/A unless the model FLOPs and device-peak assumptions are supported.

## Select an experiment

Rollout keys below are relative to `actor_rollout_ref.rollout`. Confirm backend support and the actual code path before testing.

| Evidence | Candidate | Verify |
|---|---|---|
| Trainer waits; engine concurrency is below its limit and waiting is low | Investigate submission, preprocessing, reward, group tails, or supported lookahead | Better request supply and less trainer wait. Increasing an unused concurrency limit will not help. |
| Waiting grows, KV has room, and a configured limit is reached | Test `max_num_seqs` or `max_num_batched_tokens` | Better consumed throughput without unacceptable latency, memory cost, or preemption. |
| KV pressure and preemption grow | Reduce concurrency or reassess memory allocation | Less recomputation and better overall progress; include hybrid-role coexistence. |
| Repeated prefixes and prefill materially limit progress | Test `enable_prefix_caching` | Correct cache identity/invalidation, then improved training throughput. Hits or TTFT alone are insufficient. |
| Actor/log-probability work dominates with memory headroom | Test the active token-budget or microbatch setting | Lower phase time while preserving effective batch and loss semantics. |
| Synchronization or rank imbalance dominates | Inspect adapter synchronization, workload balance, and parallel placement | Lower critical-path wait; check coupled settings and policy-age changes. |

For lookahead changes, verify whether the setting affects initial buffer fill or sustained supply. Track extra production, queue trends, memory, and staleness before claiming a sustainable gain. Use focused profiling to distinguish communication, host dispatch, transfers, and operator work.

## References

- [verl performance tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
- [verl-omni performance tuning](https://verl-omni.readthedocs.io/en/latest/perf/tuning_guide.html)
- [RL-Insight metric catalog](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/metric-name-catalog.md)
- [Hardware monitoring](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/hardware/index.md)
