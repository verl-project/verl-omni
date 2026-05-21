# Common Pitfalls

Last updated: 05/21/2026.

This document collects subtle issues that have caused non-obvious training
problems.  Each entry describes the symptom, the root cause, and the fix.

---

## Float32 precision loss in stored rollout latents

### Symptom

Training metrics show a systematic negative bias **at step 1** (before any
weight update):

- `actor/ratio_mean` consistently below `1.0` (e.g. `0.99996`)
- `actor/ppo_kl` and `actor/pg_clipfrac` inflated at step 1
- The effect is most visible with
  [rollout correction](../algo/rollout_correction.md) (`bypass_mode=True`),
  but it also affects standard training to a lesser degree.

With bypass mode the bias creates a consistent offset between the rollout
log-probs (computed by the vLLM-Omni stack) and the training log-probs
(recomputed by the FSDP stack reading the stored trajectory).  Without
bypass mode both `old_log_prob` and `current_log_prob` are computed by the
training stack, so the bias largely cancels out — but the stored trajectory
is still less accurate than it should be.

### Root cause

`FlowMatchSDEDiscreteScheduler.step()` computes `log_prob` at **float32**
precision, then casts the returned `prev_sample` (which becomes the next
`latents`) back to `model_output.dtype` — typically `bfloat16`:

```python
# In FlowMatchSDEDiscreteScheduler.step():
sample = sample.to(torch.float32)          # upcast
prev_sample, log_prob, ... = self.sample_previous_step(...)  # log_prob in fp32
# ...
prev_sample = prev_sample.to(model_output.dtype)  # ← bf16 cast!
return (prev_sample, log_prob, ...)
```

The `log_prob` was evaluated with the pre-cast float32 value; the training
stack later reads the post-cast lower-precision value.  For each latent
element $i$ with bf16 rounding error $\varepsilon_i$:

$$\Delta_i = -\frac{(\text{noise}_i + \varepsilon_i)^2}{2\sigma^2} + \frac{\text{noise}_i^2}{2\sigma^2}
= -\frac{2\varepsilon_i \cdot \text{noise}_i + \varepsilon_i^2}{2\sigma^2}$$

Averaged over the $C \times H \times W$ spatial dimensions,
$\mathbb{E}[\varepsilon_i \cdot \text{noise}_i] = 0$ (no correlation) but
$\mathbb{E}[\varepsilon_i^2] > 0$, giving a consistent negative bias:

$$\mathbb{E}[\Delta] \approx -\frac{\mathbb{E}[\varepsilon^2]}{2\sigma^2} < 0$$

This bias is **not** random noise — it is systematic and pushes
`ratio_mean` below `1.0` for every sample.

### Fix

In your rollout adapter's `diffuse()` method, store latents in float32
so the training stack reads the same values that were used for the
rollout-side log-prob computation:

```python
# Inside the SDE loop, after scheduler.step():
if i >= sde_window[0] and i < sde_window[1]:
    all_latents.append(latents.float())   # ← NOT latents
    all_log_probs.append(log_prob)
    all_timesteps.append(timestep_value)
```

The same applies to the **initial latent** captured at
`i == sde_window[0]` before the first SDE step:

```python
elif i == sde_window[0]:
    cur_noise_level = noise_level
    all_latents.append(latents.float())   # ← NOT latents
```

The runtime denoising loop still uses the casted `latents` (returned by
`scheduler.step()`) for the next transformer forward pass — this is fine
and preserves performance.  Only the **stored** latents (those that the
training stack reads back) must be float32.

**Reference implementation:**
[`verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py)

**Affected models.**  Any diffusion model whose rollout adapter stores
`all_latents` from the output of `FlowMatchSDEDiscreteScheduler.step()`
without upcasting to float32.

### Verification

After applying the fix, at step 1 you should see:

| Metric | Before fix | After fix |
|---|---|---|
| `actor/ratio_mean` | ~0.99996 | ~1.00000 |
| `actor/ppo_kl` | ~3.5×10⁻⁵ | ~2×10⁻⁶ |
| `actor/pg_clipfrac` | ~10% | ~1% |
