(rollout_server_routing)=
# Multi-Replica Rollout Server Routing

Last updated: 06/14/2026

Multi-replica rollout server routing lets VeRL-Omni choose **which rollout HTTP
replica** handles each diffusion generation request when `actor_rollout_ref.rollout.nnodes > 0`
or multiple `OmniLLMServer` actors are running. It is most useful for online
diffusion RL (FlowGRPO, MixGRPO, DiffusionNFT) where each prompt is expanded into
`rollout.n` parallel samples that should ideally land on the same replica.

## Motivation

Upstream `verl` uses `GlobalRequestLoadBalancer` with a single behavior:
**least-inflight routing**, sticky only on the per-request `request_id`. That is a
good default for LLM serving fairness and multi-turn prefix caching, but it is a
poor fit for diffusion RL batching patterns.

In a typical FlowGRPO step with `train_batch_size=32` and `rollout.n=16`, the
trainer issues **512 independent HTTP rollout requests**. Each request has a
unique `request_id`, but only **32 prompt groups** exist because 16 requests
share the same per-prompt `uid`.

With least-inflight routing keyed on `request_id`, those 16 copies scatter across
replicas. Each GPU sees a fragmented subset of the workload instead of a dense
group of homogeneous prompts.

`prompt_uid_affinity` routes by the prompt-group key (`uid` by default) so all
`rollout.n` copies for one prompt co-locate on one replica. That improves local
batch formation when combined with vLLM-Omni request-level batching or wide
`max_num_seqs` scheduling.

| Goal | `least_inflight` | `prompt_uid_affinity` |
| --- | --- | --- |
| Optimize for | Fair spread / low tail latency | Co-locate related rollout copies |
| Sticky key | `request_id` (unique per HTTP call) | `routing_key` (default: batch `uid`) |
| Typical effect with `rollout.n=16` | ~1–4 requests per replica at a time | Up to 16 requests per replica per prompt |

## What rollout server routing means

Rollout server routing in VeRL-Omni is **verl-side replica selection** in
`OmniLLMServerManager` / `OmniLLMServerClient`. It is distinct from:

- **`scheduling_policy`** (`fcfs`, etc.) — internal vLLM-Omni diffusion scheduler
  policy on a single replica.
- **Request-level batching** — fusing concurrent requests into one GPU forward
  inside a replica (vLLM-Omni feature; orthogonal to routing).

The routing layer sits between the diffusion agent loop and the rollout HTTP
servers:

1. The trainer assigns one `uid` per prompt, then repeats the batch
   `rollout.n` times (interleaved).
2. Each agent-loop worker calls `server_manager.generate(..., routing_key=uid)`.
3. `OmniRequestLoadBalancer` picks a replica according to `policy`.
4. The chosen `OmniLLMServer` runs generation as usual.

Training semantics stay on-policy: routing only affects **where** a request
executes, not **what** policy generates it.

## Quickstart

Diffusion trainers default to `prompt_uid_affinity` via
`verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml`. For
multi-replica FlowGRPO, set standalone rollout nodes and keep the default policy:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=16
```

To restore verl-style fair spreading across replicas:

```bash
actor_rollout_ref.rollout.server_routing.policy=least_inflight
```

Explicit affinity override (equivalent to the diffusion default):

```bash
actor_rollout_ref.rollout.server_routing.policy=prompt_uid_affinity
```

## Config reference

Settings live under `actor_rollout_ref.rollout.server_routing`:

| Config | Meaning |
| --- | --- |
| `server_routing.policy` | Replica routing policy (see table below). Diffusion defaults to `prompt_uid_affinity`; shared `server_routing.yaml` keeps `least_inflight` for omni LLM rollouts. |
| `server_routing.routing_key_field` | Per-sample kwargs field used as the routing key. Defaults to `uid` in code; usually omitted from yaml. |

Supported policies:

| Policy | Behavior |
| --- | --- |
| `prompt_uid_affinity` | Sticky route by `routing_key`. New keys pick the least-loaded replica; subsequent requests with the same key follow that replica. Recommended for diffusion RL with `rollout.n > 1`. |
| `least_inflight` | Sticky route by `request_id` with least-loaded assignment for new keys. Spreads concurrent requests for fairness. |
| `prompt_hash_sharding` | Deterministic shard `hash(routing_key) % num_replicas`. |
| `round_robin` | Rotate replicas regardless of load. |

Base schema: `verl_omni/trainer/config/rollout/server_routing.yaml`.
Diffusion override: `verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml`.

## What is `uid`?

`uid` is a **prompt-group identifier** assigned by the diffusion trainer each
step (one UUID per prompt before `rollout.n` expansion). After interleaved
repeat, all `rollout.n` copies of the same prompt share the same `uid`. The same
field is used for GRPO advantage grouping.

You rarely need to set `routing_key_field`; keep the default `uid` unless you
introduce a custom grouping field in the agent-loop kwargs.

## How it plugs in

The trainer creates per-prompt `uid` values, then expands for rollout:

```python
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
)
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
)
```

The single-turn diffusion agent loop extracts the routing key and forwards it to
the server manager:

```python
routing_key_field = OmegaConf.select(
    self.rollout_config, "server_routing.routing_key_field", default="uid"
)
value = kwargs.get(routing_key_field)
routing_key = str(value) if value is not None else None

output = await self.server_manager.generate(
    request_id=uuid4().hex,
    prompt_ids=prompt_ids,
    sampling_params=sampling_params,
    routing_key=routing_key,
)
```

`OmniLLMServerManager` installs `OmniRequestLoadBalancer` and reads the
policy from config:

```python
policy = OmegaConf.select(
    self.rollout_config, "server_routing.policy", default="least_inflight"
)
self.global_load_balancer = OmniRequestLoadBalancer.remote(
    servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
    policy=policy,
)
```

For `prompt_uid_affinity`, the load balancer sticks on `routing_key` (falling
back to `request_id` when the key is missing). For `least_inflight`, it sticks on
`request_id` only.

Use `OmniLLMServerManager` (not upstream `LLMServerManager`) in diffusion agent-loop
tests and trainers so `routing_key` is accepted end-to-end.

## Interaction with request-level batching

Replica routing and vLLM-Omni request-level batching are **orthogonal**:

- Routing decides **which replica** receives a request.
- Batching decides **how requests on one replica** are fused into GPU forwards.

They compose well: `prompt_uid_affinity` sends all `rollout.n` copies of a prompt
to one replica, giving local admission windows more homogeneous concurrent traffic.
In multi-replica FlowGRPO experiments, using both reduced step time compared with
`least_inflight` alone when replicas could form wider fused batches.

Routing does not require request-level batching to be enabled, and request-level
batching does not require affinity routing. Enable both when training throughput
is limited by fragmented per-replica load.

## Tests

| Test | Purpose |
| --- | --- |
| `tests/workers/rollout/test_rollout_server_routing.py` | CPU tests for policies, Hydra defaults, and `uid` clustering. |
| `tests/workers/rollout/rollout_vllm/test_rollout_server_routing_perf.py` | Multi-replica GPU benchmark comparing policies under FlowGRPO-like bursts. |

Run CPU coverage:

```bash
pytest tests/workers/rollout/test_rollout_server_routing.py -q
```

## References

- [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
- [FlowGRPO quickstart](../start/flowgrpo_quickstart.md)
- [Async reward](async_reward.md) — another advanced feature for hiding reward latency behind rollout
