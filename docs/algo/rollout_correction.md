# Rollout Correction for Diffusion Training (Experimental)

Last updated: 05/19/2026

> **Status:** Experimental. The API, default thresholds and recommended preset may change.

## Why

A FlowGRPO training step has three log-probability sources:

1. **Rollout policy** — vllm / vllm_omni sample with low-precision kernels (e.g. fp8 / bf16,
   tensor parallelism).
2. **Old policy recompute** — the actor re-runs the same trajectories under its full-precision
   training graph to produce `old_log_probs`.
3. **Current policy** — recomputed every actor mini-step to drive PPO ratios.

The recompute pass (step 2) typically costs ~20 % of the per-step time. Setting
`algorithm.rollout_correction.bypass_mode=True` skips it and reuses the rollout backend's
log-probs directly as `old_log_probs`, which yields the largest single training-time saving
but introduces an off-policy bias because the rollout and training stacks evaluate the same
trajectory slightly differently.

**Rollout Correction** addresses the off-policy bias with two orthogonal mechanisms:

- **Importance Sampling (IS)** — multiply per-sample loss by a clipped ratio
  `clamp(exp(old_logp - rollout_logp), ...)`.
- **Rejection Sampling (RS)** — zero out loss for samples whose log-ratio falls
  outside a configurable band, so the optimizer never sees extreme outliers.

The two are orthogonal and can be combined.

## Quickstart

Enable on top of any FlowGRPO run by adding two blocks of overrides:

```bash
algorithm.rollout_correction.bypass_mode=True \
algorithm.rollout_correction.rollout_is=sequence \
algorithm.rollout_correction.rollout_rs=seq_mean_k1 \
algorithm.rollout_correction.rollout_rs_threshold="0.5_2.0"
```

> **Note on `rollout_is` in bypass mode:** When `bypass_mode=True`, the PPO ratio
> ``exp(current − rollout)`` already serves as the IS correction.  The ``rollout_is``
> setting is used for IS diagnostics only; weights are **not** applied to the loss.
> Only ``rollout_rs`` rejection sampling affects the gradient.

A runnable end-to-end example lives at
[`examples/flowgrpo_trainer/run_qwen_image_ocr_lora_rollout_corr.sh`](../../examples/flowgrpo_trainer/run_qwen_image_ocr_lora_rollout_corr.sh).

## Config reference

All knobs sit under `algorithm.rollout_correction`.  Defaults are off so existing
recipes are unchanged.

| Key | Default | Description |
| --- | --- | --- |
| `bypass_mode` | `false` | Skip old-log-prob recompute; reuse `rollout_log_probs`. |
| `loss_type` | `"ppo_clip"` | Bypass loss type: ``"ppo_clip"`` (IS via PPO ratio). |
| `rollout_is` | `null` | IS aggregation: `null`, `"token"`, `"sequence"`. In bypass ppo_clip mode controls metrics only. |
| `rollout_is_threshold` | `2.0` | IS truncation: float → TIS upper clamp, `"lower_upper"` → IcePop. |
| `rollout_is_batch_normalize` | `false` | Normalize IS weights to mean 1 across the batch. |
| `rollout_rs` | `null` | RS mode(s): `seq_mean_k1`, `seq_mean_k2`, `token_k1`, etc. |
| `rollout_rs_threshold` | `null` | RS threshold. K1 modes: `"lower_upper"`; K2/K3: single upper. |

Math is delegated to ``verl.trainer.ppo.rollout_corr_helper``.
See the upstream `verl` documentation for the full mathematical formulation:
`Rollout Correction Math <https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html>`_.

## Logged metrics

| Metric | Meaning |
| --- | --- |
| `rollout_corr/rollout_is_mean` / `rollout_is_max` / `rollout_is_min` | Post-clip IS weight stats. |
| `rollout_corr/rollout_is_eff_sample_size` | Effective sample size of IS weights. |
| `rollout_corr/rollout_rs_masked_fraction` | Token-level fraction of steps rejected by RS. |
| `rollout_corr/rollout_rs_seq_masked_fraction` | Sequence-level fraction rejected by RS. |
| `rollout_corr/kl` | KL(π_rollout ‖ π_old) — direct off-policy drift estimator. |
| `rollout_corr/k3_kl` | K3 KL estimator (more stable for small KL). |
| `rollout_corr/log_ppl_diff` | Mean per-sequence log-PPL difference (rollout − old). |
| `rollout_corr/chi2_token` / `chi2_seq` | χ² divergence at token- and sequence-level. |

In **bypass mode** metrics are computed per SDE step inside ``diffusion_loss``
and appear under ``actor/rollout_corr/*``.  In **decoupled mode** they are
emitted once per global batch under ``rollout_corr/*``.

If `rollout_corr/rollout_rs_seq_masked_fraction` is consistently above ~5 %, the
rollout backend is drifting too far — tighten the RS band or fall back to
`bypass_mode=False`.

> **Gradient dilution note:** RS rejection zeroes the per-element loss for
> rejected samples but does **not** remove them from the `mean()` denominator.
> At high sustained rejection rates the effective gradient magnitude decreases by
> the factor `kept / total`.  Monitor `rollout_corr/rollout_rs_seq_masked_fraction`
> and widen the RS band if it exceeds ~10 % over several steps.

## Hyperparameter notes

Defaults (`rollout_is_threshold=2.0`, `loss_type=ppo_clip`) transfer well because:

- The helper operates on the log-ratio directly (unit-less).
- Diffusion log-probs are mean-pooled across latent dimensions, so per-step
  variance is lower than per-token LLM log-probs.

### Diffusion-specific tuning guide

The SDE window is short (`sde_window_size` is usually 2), which changes the
statistical behaviour of several RS modes:

| Concern | Recommendation |
| --- | --- |
| **`seq_mean_k1` with window=2** (decoupled mode only) | The LLM default ``"0.5_2.0"`` means the *mean* log-ratio over only 2 steps must lie in ``[−0.69, 0.69]``.  A single outlier step can reject the entire sample.  If `rollout_corr/rejected_ratio` is high, widen to e.g. ``"0.3_3.0"`` or ``"0.2_5.0"``. |
| **Bypass mode RS** | In bypass mode, IS/RS is computed per SDE step with shape ``(B, 1)``, so all RS modes (``token_k1``, ``seq_mean_k1``, etc.) are effectively equivalent — each step is evaluated independently.  ``seq_mean_*`` is recommended for consistency if you plan to switch between modes. |
| **Token-level RS** (`token_k1`, etc.) | With only 2 tokens, token-level statistics have very low power — a single token cannot be rejected in isolation because the per-token stat is averaged from thousands of latent dims.  Prefer `seq_mean_*` or `seq_max_*` modes (decoupled) or any mode (bypass — all are per-step). |
| **`rollout_is=sequence`** | The product of 2 per-step ratios.  With diffusion's low per-step variance this is usually well-behaved; the default threshold of 2.0 is generous. |
| **First-run diagnostics** | Always inspect `rollout_corr/log_ppl_abs_diff` and `rollout_corr/rollout_is_max` for the first 50 steps of a new recipe.  If `log_ppl_abs_diff > 1.0` or `rollout_is_max` is pinned at the clamp threshold, the rollout-training gap is larger than expected — consider lowering the rollout precision gap or falling back to `bypass_mode=False`. |

### `loss_type` and when to use each

| `loss_type` | Mode | IS applied? | When to use |
| --- | --- | --- | --- |
| `ppo_clip` (default) | bypass | No (PPO ratio handles IS) | Standard bypass — fastest, PPO clipping provides IS. |
| `ppo_clip` | decoupled | Yes (``old/rollout``) | Highest fidelity — 3 policies, IS corrects rollout→old drift. |


## How it plugs in

1. **Bypass entrypoint.** ``apply_bypass_mode_to_diffusion_batch`` sets
   ``old_log_probs := rollout_log_probs`` (zero-cost).  The trainer-side
   decoupled correction is skipped because ``old == rollout`` would be a no-op.
2. **Per-step correction.** ``diffusion_loss`` reads ``config.rollout_correction``
   and computes IS/RS per SDE step via ``compute_rollout_correction_and_rejection_mask``.
   For ``ppo_clip`` only the RS mask is applied; the PPO ratio ``exp(current − rollout)``
   handles IS.
3. **Decoupled correction.** ``apply_rollout_correction_to_diffusion_batch``
   runs once per global batch using ``old_log_probs`` vs ``rollout_log_probs``
   and stashes a combined ``rollout_is_weights`` tensor.
4. **Loss application.** ``flow_grpo`` / ``grpo_guard`` multiply per-element
   loss by (detached) ``rollout_is_weights``.  Diffusion has no padding so
   rejection is weight=0 — no separate mask needed.

The config lives on ``DiffusionActorConfig.rollout_correction`` (imported from
verl).  No dedicated loss registration or engine modifications are required.
