(tuning_guide)=
# Performance Tuning Guide

Last updated: 08/31/2026

This page is the starting point for tuning a VeRL-Omni diffusion RL run. It
does not repeat the detail already covered by the more specific pages —
instead it gives you a decision order and links to the right page for each
decision, plus a troubleshooting checklist for the OOM/throughput problems
that come up across all of them.

## Where time actually goes

A FlowGRPO-style step has three stages, and each has its own tuning surface:

| Stage | What it does | Tune with |
|---|---|---|
| Rollout | Generate images/video/audio for each prompt | {ref}`rollout_batching` |
| Reward | Score generated samples | [Async Reward](../algo/async_reward.md) |
| Actor | Compute advantages and update the policy | [Tuning and Improving MFU](diffusion_mfu.md#tuning-and-improving-mfu) |

Before changing any config, profile the step to see which stage actually
dominates wall time — see [Profiling FlowGRPO / diffusion training](profiler.md).
Guessing which stage is slow from symptoms alone is unreliable: a rollout
that looks slow is often actually reward-bound once you profile it (see
[Async Reward](../algo/async_reward.md#motivation)).

## 1. Decide your GPU layout first

Layout changes are the highest-leverage tuning decision and should come
before any per-stage knob, because they change what "optimal" means for the
other stages.

- **Colocated** (actor, rollout, and reward share the same GPU pool):
  simplest setup, no idle GPUs, but reward/rollout and actor training
  time-share the same devices — a slow reward model stalls the whole step.
  This is the default in most `examples/` scripts.
- **Disaggregated reward pool** (`reward.reward_model.enable_resource_pool=True`):
  puts reward-model inference on its own GPUs so it overlaps with rollout
  generation instead of blocking it. Worth it once reward scoring is a
  significant fraction of step time — see [Async Reward](../algo/async_reward.md)
  for the config and the GPU-count tradeoff.

Scaling to more nodes is a separate, orthogonal decision — colocated vs.
disaggregated reward still applies once you add nodes. The default multi-node
recipe in [Multi-Node Training](../start/multi_node_training.md) keeps reward
colocated (`REWARD_TP=4` on every node); that guide also shows how actor,
rollout, and reward pools map onto nodes for both layouts.

Re-profile after any layout change — the stage that was the bottleneck
before a layout change is often not the bottleneck after it.

## 2. Tune rollout throughput

Once the GPU layout is fixed, the rollout engine has its own batching
tradeoff that is independent of everything else: step-wise continuous
batching vs. request-level batching. See {ref}`rollout_batching` for how to
choose between them, the config knobs (`step_execution`,
`++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs`,
`++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms`),
and measured before/after numbers for the example recipes. Set these under
`engine_kwargs.vllm_omni` — the bare `rollout.max_num_seqs` dataclass field is
a different knob and overriding it silently has no effect on batching.

## 3. Tune actor throughput and memory

[Tuning and Improving MFU](diffusion_mfu.md#tuning-and-improving-mfu) is the
actor-side playbook: `param_offload` / `optimizer_offload`, Ulysses
sequence-parallel size, micro-batch size, `layered_summon`, and the
gradient-checkpointing MFU caveat. Read that section before changing actor
config — it also explains *why* each knob helps, which matters when your OOM
point differs from the reference 20B-on-H200 setup it was written against.

## 4. Troubleshooting checklist

Symptoms that show up regardless of which stage causes them:

**OOM during rollout generation**
- Lower `max_num_seqs` first if you are on the request-level batching path —
  it packs one full activation buffer per concurrent request rather than
  reusing a step-wise buffer, so it uses more memory per concurrent request
  for the same value (see the warning in {ref}`rollout_batching`).
- Check `actor_rollout_ref.rollout.gpu_memory_utilization` isn't already
  near 1.0 — this is the vLLM-Omni engine's memory reservation, not an LLM KV
  cache, and it leaves less headroom for the packed activation memory a high
  `max_num_seqs` needs.

**OOM during `update_actor` / `update_weights`**
- This is the actor-side path — go through the offload ordering in
  [Tuning and Improving MFU](diffusion_mfu.md#tuning-and-improving-mfu)
  (`optimizer_offload=True` first, then
  `param_offload=True` as a last resort) rather than reducing batch size
  first, since offloading costs less throughput than a smaller micro-batch.
- If both offload flags are already `True`, confirm `layered_summon=True` —
  disabling it under offload tends to OOM during weight sync.

**Step time dominated by reward scoring**
- Confirm with a profiler trace (recipe 6 in [profiler.md](profiler.md)) before
  changing anything — reward cost is easy to misattribute to rollout because
  both run inside the same wall-clock "generation" window when reward is
  colocated.
- Move to a disaggregated reward pool
  (`reward.reward_model.enable_resource_pool=True`) so
  scoring overlaps generation instead of gating it; see
  [Async Reward](../algo/async_reward.md).

**MFU looks implausibly low or above 1.0**
- See [How FLOPs are computed](diffusion_mfu.md#how-flops-are-computed) first
  — if `perf/mfu/actor > 1.0`, the two documented causes are a mis-identified
  device peak on relabeled SKUs (pin the real peak with
  `VERL_OMNI_DEVICE_FLOPS_TFLOPS`) and a missing DP gather of sequence
  lengths, not the LoRA vs. full-FT FLOPs caveat — that one only matters when
  comparing LoRA against full FT, not as a path to MFU above 1.0.

## See also

- [Profiling FlowGRPO / diffusion training](profiler.md)
- {ref}`diffusion_mfu`
- {ref}`rollout_batching`
- [Async Reward](../algo/async_reward.md)
- [Multi-Node Training](../start/multi_node_training.md)
- [Performance Reference](../algo/performance.md) — measured throughput for
  colocated vs. async-reward recipes
