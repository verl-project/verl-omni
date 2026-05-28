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

### Fix

Two changes in the scheduler, one in the rollout adapter.
The training adapter is **unchanged** — it already uses fp32 correctly.

**1. Scheduler** — `step()` no longer truncates `prev_sample` to bfloat16,
and `sample_previous_step()` asserts `model_output` is float32 so callers
cannot accidentally pass lower precision.

**2. Rollout adapter** — latents are cast to the transformer's native dtype
before the forward pass (performance), noise_pred is cast to float32 before
the scheduler (precision), and all stored latents are in float32.

### Verification

The fix eliminates the systematic precision-loss bias from the scheduler.
In non-bypass mode (no rollout correction) `ratio_mean ≈ 1.0` at step 1.
In bypass mode a ~3×10⁻⁵ KL divergence remains due to the vLLM vs PyTorch
attention kernel difference, which is unavoidable when using different
inference backends.

| Metric | Before fix (bypass) | After fix (bypass) | No bypass |
|---|---|---|---|
| `actor/ppo_kl` | ~3.6×10⁻⁵ | ~3.3×10⁻⁵ | ~1×10⁻⁶ |
| `actor/pg_clipfrac` | ~12% | ~9% | ~1% |
