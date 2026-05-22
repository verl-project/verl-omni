# Common Pitfalls

Last updated: 05/22/2026.

---

## Float32 precision loss in stored rollout latents

### Symptom

Training metrics show a systematic negative bias **at step 1** (before any
weight update):

- `actor/ratio_mean` consistently below `1.0` (e.g. `0.99996`)
- `actor/ppo_kl` and `actor/pg_clipfrac` inflated at step 1
- `actor/pg_clipfrac_higher` is **zero** — all clipping on the lower side
- Most visible with rollout correction (`bypass_mode=True`), but also
  degrades stored trajectory precision in standard training.

### Root cause

`FlowMatchSDEDiscreteScheduler.step()` computes `log_prob` in **float32**
using the fp32 `prev_sample`, then **casts `prev_sample` back to
`model_output.dtype` (bfloat16)** before returning.  The stored latents
lose precision, creating a mismatch with the log-prob computation.

Additionally, `sample_previous_step()` did not cast `model_output` to
float32, unlike the [official flow_grpo implementation](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)
which casts all inputs to fp32 upfront.

### Fix

Two changes in the scheduler, one in the rollout adapter.
The training adapter is **unchanged** — it already uses fp32 correctly.

**1. Scheduler** — stop truncating `prev_sample` and assert `model_output` is fp32:

```python
# In FlowMatchSDEDiscreteScheduler.step():
# REMOVE: prev_sample = prev_sample.to(model_output.dtype)

# In FlowMatchSDEDiscreteScheduler.sample_previous_step():
# ADD: assert model_output.dtype == torch.float32
```

**2. Rollout adapter** — cast to model dtype for transformer forward (perf),
cast noise_pred to fp32 for scheduler (precision), store latents in fp32:

```python
x = latents.to(self.transformer.dtype)         # cast to bf16 for transformer
noise_pred = self.transformer(hidden_states=x, ...)
latents, log_prob, _, _ = self.scheduler.step(
    noise_pred.float(), ...)                   # fp32 for scheduler
all_latents.append(latents)                    # fp32 from scheduler
all_latents.append(latents.float())            # initial latent: bf16 → fp32
```

### Verification

After applying the fixes, at step 1 the systematic bias from precision loss
is eliminated.  Any remaining sub-1e-4 KL divergence is from the vLLM vs
PyTorch attention kernel difference, which is unavoidable when using different
inference backends.

| Metric | Before fix | After fix |
|---|---|---|
| `actor/ratio_mean` | ~0.99996 | ~1.00000 |
| `actor/ppo_kl` | ~3.5×10⁻⁵ | ≤1×10⁻⁵ |
| `actor/pg_clipfrac` | ~10% | ≤5% |
| `actor/pg_clipfrac_higher` | ~0% | roughly symmetric |
