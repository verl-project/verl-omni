# Common Pitfalls

Last updated: 06/15/2026.

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

---

## RoPE text-length mismatch under continuous batching

### Symptom

When `step_execution=True`, `actor/ppo_kl` is elevated even at step 1
compared to the full-forward (`step_execution=False`) path. The effect
persists across training steps and cannot be eliminated by the fp32
latent-storage fix alone.

This also affects the **stock vllm-omni (non-stepwise) path** in some
configurations — the root cause is upstream, not specific to stepwise
mode.

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

### Verification

Compare `actor/ppo_kl` at step 1 between `step_execution=True` and
`step_execution=False` runs with all other knobs identical. After the
fix the difference should be within numerical tolerance (~3×10⁻⁵ KL
divergence due to unavoidable vLLM vs PyTorch attention kernel
difference).

---

## fp32 latent storage regression in stepwise mode

### Symptom

Training metrics show a systematic negative bias **at step 1** when
`step_execution=True`:

- `actor/ratio_mean` consistently below `1.0`
- `actor/ppo_kl` and `actor/pg_clipfrac` inflated at step 1
- `actor/pg_clipfrac_higher` is **zero** — all clipping on the lower side

The same model/config produces correct `ratio_mean ≈ 1.0` when
`step_execution=False`.

### Root cause

`step_scheduler` stores `new_latents` in the model's compute dtype (bf16)
instead of fp32. The trainer later recomputes log-probs on these stored
latents via `FlowMatchSDEDiscreteScheduler.sample_previous_step()` in
fp32, creating a precision mismatch. Additionally, under continuous
batching the engine gathers latents across in-flight requests:
a freshly-added request has fp32 latents while stepped requests have
bf16 latents, producing a "Mixed dtypes in latents batch" error.

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

### Verification

`ratio_mean ≈ 1.0` at step 1 with `step_execution=True`, matching the
`step_execution=False` baseline within tolerance.

---

## MixGRPO window-positioning bypass in stepwise mode

### Symptom

With `step_execution=True` and MixGRPO, different rollouts in the same
batch receive different SDE windows rather than sharing a single window.
Advantage estimation is corrupted — training may converge to a wrong
optimum or diverge.

### Root cause

MixGRPO requires all rollouts in a batch to share one SDE window for
correct advantage estimation. In full-forward mode this is set by
`_maybe_make_progressive_window()` inside `forward()`. In stepwise mode
`forward()` is never called, so the window-positioning logic is silently
bypassed.

### Fix

Call `_maybe_make_progressive_window()` in `prepare_encode` **before**
delegating via `super()`. Use multiple inheritance so the MixGRPO
window-setting logic runs first, then the stepwise `prepare_encode`
draws the already-fixed window:

```python
class QwenImageMixGRPOPipelineWithLogProbStepwise(
    QwenImageMixGRPOPipelineWithLogProb,        # window logic
    QwenImagePipelineWithLogProbStepwise,        # stepwise overrides
):
    def prepare_encode(self, state, **kwargs):
        if state.sampling is not None:
            if state.sampling.extra_args is None:
                state.sampling.extra_args = {}
            self._maybe_make_progressive_window(
                state.sampling.extra_args, kwargs
            )
        return super().prepare_encode(state, **kwargs)
```

### Verification

Add an assertion in `prepare_encode` that all requests in a batch have
the same `sde_window` after the window is set. Verify that MixGRPO
training metrics with `step_execution=True` match the full-forward
baseline.

---

## Tokenizer fallback mismatch between diffusers and vllm-omni

### Symptom

The engine's dummy warm-up run (which submits a text prompt since no
pre-tokenized `prompt_ids` are available) produces different tokenization
than the diffusers training pipeline. This causes a prompt-embedding
mismatch between rollout and training, silently degrading reward quality.

### Root cause

The diffusers pipeline uses `max_length=tokenizer_max_length + drop_idx`
(e.g. 1058) with `truncation=True`, while vllm-omni may use different
defaults for `max_length` and truncation behavior. The warm-up path in the
stepwise adapter must match the diffusers behavior exactly.

### Fix

In the stepwise adapter's `_tokenize_text_prompt` helper, replicate the
diffusers tokenization parameters:

```python
def _tokenize_text_prompt(self, text):
    prompt = [text] if isinstance(text, str) else text
    txt = [self.prompt_template_encode.format(e) for e in prompt]
    tokens = self.tokenizer(
        txt,
        max_length=self.tokenizer_max_length
            + self.prompt_template_encode_start_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(self.device)
    return tokens.input_ids, tokens.attention_mask
```

### Verification

The warm-up pass should produce embeddings with the same sequence length
as the diffusers pipeline for the same prompt template.

---

## Device placement of attention_mask during text encoding

### Symptom

A device-mismatch error when `attention_mask` is passed to
`_extract_masked_hidden` — the mask stays on CPU while the hidden states
are on GPU.

### Root cause

In `_get_qwen_prompt_embeds`, the encoder call moves `prompt_ids` and
`attention_mask` to `self.device` inside the `.to(self.device)` call, but
the *original* `attention_mask` tensor (captured in the outer scope) still
points to CPU memory. When it is later passed to `_extract_masked_hidden`,
the device mismatch causes a runtime error.

### Fix

Move both `prompt_ids` and `attention_mask` to device **in-place** before
the encoder call, so the same tensor objects are on the correct device
downstream:

```python
def _get_qwen_prompt_embeds(self, prompt_ids, attention_mask=None, dtype=None):
    ...
    prompt_ids = prompt_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    encoder_hidden_states = self.text_encoder(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    split_hidden_states = self._extract_masked_hidden(
        hidden_states, attention_mask
    )
    ...
```

### Verification

The warm-up path and first real request should both complete without
device-mismatch errors. This is typically caught by the engine's dummy
run on startup.
