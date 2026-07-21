(request_level_batching)=
# Request-Level Rollout Batching

Last updated: 07/20/2026.

Request-level batching packs multiple in-flight diffusion requests into one
transformer forward on each vLLM-Omni worker. It applies to any diffusion RL
algorithm whose rollout adapter sets `supports_request_batch = True` (not only
FlowGRPO).

Requirements:

- `actor_rollout_ref.rollout.step_execution=false` (yaml default)
- Rollout adapter with `supports_request_batch = True`

This is **not** step-wise continuous batching (`step_execution=true`). For that
path, see
[`integrating_a_stepwise_continuous_batching_model.md`](../contributing/integrating_a_stepwise_continuous_batching_model.md).

## How to enable

```bash
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=32
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=10
```

| Knob | Meaning |
|---|---|
| `max_num_seqs` | Max requests packed per forward. vLLM-Omni engine default is `1` (serial). |
| `request_batch_max_wait_ms` | Optional admission wait (ms) before scheduling. Engine default is `0`; `10` is enough for typical training bursts. |

Example FlowGRPO scripts already set these (`max_num_seqs=32` for Qwen-Image,
`256` for SD3.5; `wait_ms=10`). Override with `MAX_NUM_SEQS` /
`REQUEST_BATCH_MAX_WAIT_MS`, or Hydra as above.

Pick `max_num_seqs` from memory headroom. For Qwen-Image e2e (32×16, 512²,
LoRA + true CFG), keep `max_num_seqs≤32` — larger values can OOM.

## Expected performance (our experiments)

Gen time excludes step 1 (warmup).

| Workload | serial (`max_num_seqs=1`) | batched | Δ gen |
|---|---:|---:|---:|
| Qwen-Image LoRA, 32×16, 512², `max_num_seqs=32` | 226.4s | 107.9s | **−52%** |
| SD3.5 LoRA, 8×8, 384² FA3, `max_num_seqs=256` | 25.4s | 22.3s | **−12%** |

`wait_ms` has little effect on these recipes (`0` vs `50` differs by ~2% on
Qwen-Image). The gain comes from `max_num_seqs > 1`.

## Supporting a new model

Implement request-batch handling in the rollout adapter. See
[`integrating_a_diffusion_model.md`](../contributing/integrating_a_diffusion_model.md#request-level-batching).
