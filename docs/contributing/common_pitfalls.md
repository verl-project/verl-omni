# Common Pitfalls

Last updated: 06/15/2026.

---

## Float32 Precision Loss in Stored Rollout Latents

(symptom-float32)=
### Symptom

Training metrics show a systematic negative bias **at step 1** (before any
weight update):

- `actor/ratio_mean` consistently below `1.0` (e.g. `0.99996`)
- `actor/ppo_kl` and `actor/pg_clipfrac` inflated at step 1
- `actor/pg_clipfrac_higher` is **zero** — all clipping on the lower side
- Most visible with rollout correction (`bypass_mode=True`), but also
  degrades stored trajectory precision in standard training.

(root-cause-float32)=
### Root cause

`FlowMatchSDEDiscreteScheduler.step()` computes `log_prob` in **float32**
using the fp32 `prev_sample`, then **casts `prev_sample` back to
`model_output.dtype` (bfloat16)** before returning.  The stored latents
lose precision, creating a mismatch with the log-prob computation.

(fix-float32)=
### Fix

Two changes in the scheduler, one in the rollout adapter.
The training adapter is **unchanged** — it already uses fp32 correctly.

**1. Scheduler** — `step()` no longer truncates `prev_sample` to bfloat16,
and `sample_previous_step()` asserts `model_output` is float32 so callers
cannot accidentally pass lower precision.

**2. Rollout adapter** — latents are cast to the transformer's native dtype
before the forward pass (performance), noise_pred is cast to float32 before
the scheduler (precision), and all stored latents are in float32.

(verification-float32)=
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

---

## RoPE Sequence Length Mismatch

(symptom-rope)=
### Symptom

When `step_execution=True`, `actor/ppo_kl` is elevated even at step 1
compared to the full-forward (`step_execution=False`) path. The effect
persists across training steps and cannot be eliminated by the fp32
latent-storage fix alone.

This also affects the **stock vllm-omni (non-stepwise) path** in some
configurations — the root cause is upstream, not specific to stepwise
mode.

(root-cause-rope)=
### Root cause

vllm-omni sets Rotary Position Embedding (RoPE) sequence lengths from
`mask.sum()` (valid token count), while diffusers sets them from the
padded encoder-tensor width (`text_seq_len`). Under continuous batching,
vllm-omni pads all requests to a shared `target_seq_len`, so valid
tokens at positions beyond ~50 receive incorrect RoPE — they get the
positional encoding of a much shorter sequence.

Concretely, if a request has 200 valid tokens and is padded to width
1058, `mask.sum()` = 200 but the embedding width is 1058. The RoPE
position for token 100 is computed as position 100 of a 200-length
sequence rather than position 100 of a 1058-length sequence.

(fix-rope)=
### Fix

In `prepare_encode`, set `txt_seq_lens` from the padded embed width
instead of from `mask.sum()`:

```python
# Wrong (vllm-omni default):
txt_seq_lens = [int(mask.sum()) for mask in prompt_embeds_mask]

# Correct (matches diffusers):
txt_seq_lens = [int(prompt_embeds.shape[1])] * int(prompt_embeds.shape[0])
```

The stepwise adapters in `verl_omni/experimental/` already do this.
The stock vllm-omni path is still affected and tracked as an upstream
issue.

(verification-rope)=
### Verification

Compare `actor/ppo_kl` at step 1 between `step_execution=True` and
`step_execution=False` runs with all other knobs identical. After the
fix the difference should be within numerical tolerance (~3×10⁻⁵ KL
divergence due to unavoidable vLLM vs PyTorch attention kernel
difference).

---

## Float32 Precision Loss in Stepwise Scheduler

(symptom-fp32-stepwise)=
### Symptom

Training metrics show a systematic negative bias **at step 1** when
`step_execution=True`:

- `actor/ratio_mean` consistently below `1.0`
- `actor/ppo_kl` and `actor/pg_clipfrac` inflated at step 1
- `actor/pg_clipfrac_higher` is **zero** — all clipping on the lower side

The same model/config produces correct `ratio_mean ≈ 1.0` when
`step_execution=False`.

(root-cause-fp32-stepwise)=
### Root cause

`step_scheduler` stores `new_latents` in the model's compute dtype (bf16)
instead of fp32. The trainer later recomputes log-probs on these stored
latents via `FlowMatchSDEDiscreteScheduler.sample_previous_step()` in
fp32, creating a precision mismatch. Additionally, under continuous
batching the engine gathers latents across in-flight requests:
a freshly-added request has fp32 latents while stepped requests have
bf16 latents, producing a "Mixed dtypes in latents batch" error.

(fix-fp32-stepwise)=
### Fix

Two changes in `step_scheduler`:

1. Store `new_latents.float()` in the trajectory lists.
2. Keep `state.latents` in fp32 throughout — do NOT cast to model dtype
   after the scheduler step. `denoise_step` already casts to the
   transformer dtype before the forward pass.

```python
# Wrong:
state.latents = new_latents  # bf16

# Correct:
state.latents = new_latents.to(torch.float32)
```

The non-CB `diffuse()` path already does this correctly — the stepwise
override must match.

(verification-fp32-stepwise)=
### Verification

`ratio_mean ≈ 1.0` at step 1 with `step_execution=True`, matching the
`step_execution=False` baseline within tolerance.
