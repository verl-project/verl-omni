(separate_async_omni)=
# Separate-Async RL Training for Qwen3-Omni

Last updated: 09/08/2026

`trainer.v1.trainer_mode=omni_separate_async` runs training and rollout on
separate GPU pools for omni AR models (Qwen3-Omni thinker). Standalone rollout
replicas generate one batch ahead of training; the trainer pushes weights to
them every `trainer.v1.separate_async.parameter_sync_step` steps over a
non-naive checkpoint engine (nccl/nixl/...). Generations aborted by a weight
sync continue from the tokens already produced and finish under the new
weights, with `min/max_global_steps` recording the weight-version span of
every sample.

## When to use

- Long-tail completions (large `rollout.n`, high length variance) leave the
  trainer or rollout GPUs idle in `omni_sync`.
- Multimodal prefill (image/video/audio encoders) makes generation the
  dominant phase of the step.

Off-policyness is bounded by the one-batch-ahead pipeline: a sample is trained
at most one weight version after it was generated (at
`parameter_sync_step=1`). Raising `parameter_sync_step` trades sync overhead
for staleness; the replay buffer's
`trainer.v1.sampler.max_off_policy_threshold` remains the hard backstop
(`drop` or `wait`).

## GPU layout

`trainer.n_gpus_per_node × trainer.nnodes` GPUs run the FSDP actor;
`actor_rollout_ref.rollout.n_gpus_per_node × actor_rollout_ref.rollout.nnodes`
additional GPUs run standalone rollout replicas
(`n_gpus_per_node / tensor_model_parallel_size` replicas per node). Single-node
replicas only for AR omni today (`run_headless` is not implemented upstream).

## Run

```bash
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_separate_async_v1.sh
```

The example splits 4 GPUs into 2 trainer + 2 rollout (one TP=2 replica) and
uses GSPO + GRPO advantages with LoRA. Key overrides:

| knob | default | meaning |
|---|---|---|
| `trainer.v1.separate_async.num_warmup_batches` | 4 | prompt batches submitted before the first step |
| `trainer.v1.separate_async.parameter_sync_step` | 4 | push weights to standalone replicas every N steps |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | — | must be non-naive (`nccl`, `nixl`, `mooncake`) |

Constraints enforced at startup: rollout GPUs > 0, a non-naive checkpoint
backend, and `data.train_batch_size == parameter_sync_step * ppo_mini_batch_size`.
The trainer reads `parameter_sync_step` from `v1.separate_async` (the key the
parent validates and syncs on), not from a mode-specific stub. The MMK12
example sets `parameter_sync_step=8` so `128 == 8 * 16`.

LoRA recipes should set `actor_rollout_ref.model.lora.merge=False` so weight
sync ships only adapter tensors (applied on the replicas via the LoRA-aware
checkpoint engine manager).

## Monitor

Watch `training/off_policy/*` metrics (staleness mean/max, dropped samples)
and `timing_s/update_weights`. If weight sync stalls dominate, raise
`parameter_sync_step` or shorten `data.max_response_length`.

## Test

CPU (no GPU required; covers registration, the `parameter_sync_step` key fix,
LoRA-aware worker/manager wiring, CPU save/restore wiring, recovery-client
contracts, and the `adapter_name` forwarding fix):

```bash
pytest -s --asyncio-mode=auto \
    tests/trainer/omni/test_ray_omni_trainer_separate_async_on_cpu.py \
    tests/workers/rollout/test_omni_rollout_recovery_on_cpu.py \
    tests/workers/test_omni_fsdp_engine_on_cpu.py
```

GPU smoke (2 GPUs, tiny-random Qwen3-Omni, 3 separate-async steps):

```bash
bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_separate_async_smoke.sh
```

For a parity check against the synchronous baseline, run the same recipe with
`trainer.v1.trainer_mode=omni_sync` and compare reward curves; at
`parameter_sync_step=1` they should overlap within noise.

## Limitations

- **Abort semantics.** Weight sync is
  abort-then-pause: `abort_all_requests` issues a timeout-bounded, ACK'd
  `engine.abort` while generate is live, then `pause_generation(mode="abort")`
  as the idle boundary and admission hold. Success-path terminals come from
  the engine with the tokens generated so far (failure-path synthesis is
  enqueue-then-raise only). `FullyAsyncLLMServerClient` continues from
  `prompt + partial_tokens` under the new weights and shrinks the remaining
  token budget — pinned by `test_omni_rollout_recovery_on_cpu.py` (budgets
  `[8, 6]`). Scope is thinker-only / single-stage AR; multi-stage abort is
  still broken upstream.
- Single-node standalone replicas for AR omni (`vLLMOmniHttpServer.run_headless`
  is not implemented).
- Hybrid (colocated) replicas are not idle: they serve the first sampling
  window and every validation (kept current via the colocated checkpoint
  engine), and sleep during training phases.
- **Prefix cache after weight sync.** Abort-then-pause clears the frontend mm
  cache when `reset_prefix_cache=True`. Prefix-hash reuse after
  sleep/wake is still an upstream vllm-omni residual; the example and
  smoke keep `enable_prefix_caching=False`.
- **Decoupled PPO is the default.** Generated omni config ships
  `algorithm.rollout_correction.bypass_mode: false`, and nothing in this
  trainer forces bypass, so the parent's save/restore path
  (`PPOTrainerSeparateAsync._compute_old_log_prob`) runs on GPU.
  `OmniDetachActorWorker` supplies the CPU snapshot used when
  `parameter_sync_step > 1`. The MMK12 example sets `parameter_sync_step=8`
  so that path is engaged. Cover save/restore plus hard-abort/weight-sync
  before raising the knob further.
- NPU AR sleep/wake relies on vllm-ascend behavior.
