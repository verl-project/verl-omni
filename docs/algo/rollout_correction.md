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

**Rollout Correction** is the same fix shipped in verl
([`docs/algo/rollout_corr.md`](https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md))
applied to the diffusion stack:

- **Importance Sampling (IS)** multiplies the per-sample loss by a clipped ratio
  `clamp(exp(old_logp - rollout_logp), ...)` (TIS / Token-level IS or sequence-level IS).
- **Rejection Sampling (RS)** zeroes out the loss contribution of samples whose log-ratio
  falls outside a configurable band, so the optimizer is never fed extreme outliers.

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
> setting is still used to compute IS diagnostics (logged under
> ``actor/rollout_corr/*``), but IS weights are **not** applied to the loss — only
> ``rollout_rs`` rejection sampling affects the gradient.  This matches verl's
> ``loss_type: ppo_clip`` contract (see ``compute_policy_loss_bypass_mode``).

A runnable end-to-end example lives at
[`examples/flowgrpo_trainer/run_qwen_image_ocr_lora_rollout_corr.sh`](../../examples/flowgrpo_trainer/run_qwen_image_ocr_lora_rollout_corr.sh).

## Config reference

All knobs sit under `algorithm.rollout_correction`.  Field names and defaults
mirror ``verl.trainer.config.algorithm.RolloutCorrectionConfig`` so that configs
are portable between the two stacks.  Defaults are off-by-default so existing
recipes are unchanged:

| Key | Default | Description |
| --- | --- | --- |
| `bypass_mode` | `false` | Reuse `rollout_log_probs` as `old_log_probs` (skip the actor recompute pass). |
| `loss_type` | `"ppo_clip"` | Loss type in bypass mode: ``"ppo_clip"`` (IS via PPO ratio, default) or ``"reinforce"`` (reserved — IS weights applied explicitly, no PPO clipping).  Matches verl's ``loss_type``. |
| `rollout_is` | `null` | One of `null`, `"token"`, `"sequence"`. In **decoupled mode** IS weights multiply the loss. In **bypass ppo_clip mode** the PPO ratio handles IS; ``rollout_is`` controls metrics reporting only. `"sequence"` is recommended for diffusion because SDE log-probs are already pooled across many latent dimensions. |
| `rollout_is_threshold` | `2.0` | Float for TIS upper clamp, or a string `"lower_upper"` to enable IcePop-style asymmetric clipping. |
| `rollout_is_batch_normalize` | `false` | Normalize the IS weights to mean 1 across the batch (variance reduction). |
| `rollout_rs` | `null` | RS mode. Common picks: `seq_mean_k1`, `seq_mean_k2`, `token_k1`. See verl's helper for the full list. |
| `rollout_rs_threshold` | `null` | Threshold required by most RS modes. For `seq_mean_k1` use `"0.5_2.0"` (drop samples whose mean ratio leaves the band). |

The implementation delegates the math to
`verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_and_rejection_mask`, so the
algorithm semantics are byte-for-byte identical to the verl LLM stack.

## Logged metrics

The standard `rollout_corr/*` metrics are emitted in both operating modes:

| Metric | Meaning |
| --- | --- |
| `rollout_corr/timing` (via `timing_s/rollout_corr`, decoupled mode only) | Wall time of the trainer-side correction step. |
| `rollout_corr/is_weights/{mean,max,min}` | Stats of the (post-clip) IS weights. |
| `rollout_corr/rejected_ratio` | Fraction of samples rejected by RS. |
| `rollout_corr/log_ratio/{mean,std,abs_max}` | Health check on rollout vs training drift. |

In **bypass mode** the metrics are computed per micro-batch / SDE step inside the engine
using `(current_log_prob, rollout_log_prob)` and aggregated with the standard mean reducer,
so the keys appear under `actor/rollout_corr/*` in the trainer logger.  IS weights are
computed for diagnostics but **not applied to the loss** (the PPO ratio handles IS in this
mode).  In **decoupled mode** they are emitted once per global batch from the trainer-side
correction (`old_log_prob` vs `rollout_log_prob`) and appear directly under `rollout_corr/*`.

If `rollout_corr/rejected_ratio` is consistently above ~5 %, your rollout backend is
drifting too far — tighten the RS band, lower the rollout precision gap, or fall back to
`bypass_mode=False`.

## Hyperparameter notes vs verl

We deliberately keep the same defaults as verl (`rollout_is_threshold=2.0`,
`loss_type=ppo_clip`).  They transfer reasonably because:

- The helper operates on the log-ratio directly, which is unit-less.
- Diffusion `old_log_probs` from
  [`verl_omni.utils.flow_match_sde`](../../verl_omni/utils/flow_match_sde.py) are mean-pooled
  across latent dimensions, so the per-step variance is lower than per-token LLM log-probs.

### Diffusion-specific tuning guide

The SDE window is short (`sde_window_size` is usually 2), which changes the
statistical behaviour of several RS modes:

| Concern | Recommendation |
| --- | --- |
| **`seq_mean_k1` with window=2** | The LLM default ``"0.5_2.0"`` means the *mean* log-ratio over only 2 steps must lie in ``[−0.69, 0.69]``.  A single outlier step can reject the entire sample.  If `rollout_corr/rejected_ratio` is high, widen to e.g. ``"0.3_3.0"`` or ``"0.2_5.0"``. |
| **Token-level RS** (`token_k1`, etc.) | With only 2 tokens, token-level statistics have very low power — a single token cannot be rejected in isolation because the per-token stat is averaged from thousands of latent dims.  Prefer `seq_mean_*` or `seq_max_*` modes. |
| **`rollout_is=sequence`** | The product of 2 per-step ratios.  With diffusion's low per-step variance this is usually well-behaved; the default threshold of 2.0 is generous. |
| **First-run diagnostics** | Always inspect `rollout_corr/log_ratio/abs_max` and `rollout_corr/is_weights/max` for the first 50 steps of a new recipe.  If `abs_max > 1.0` or `is_weights/max` is pinned at the threshold, the rollout-training gap is larger than expected — consider lowering the rollout precision gap or falling back to `bypass_mode=False`. |

### `loss_type` and when to use each

| `loss_type` | Mode | IS applied? | When to use |
| --- | --- | --- | --- |
| `ppo_clip` (default) | bypass | No (PPO ratio handles IS) | Standard bypass — fastest, PPO clipping provides IS. |
| `ppo_clip` | decoupled | Yes (``old/rollout``) | Highest fidelity — 3 policies, IS corrects rollout→old drift. |
| `reinforce` (reserved) | bypass | Yes (``current/rollout``) | Future: pure policy gradient without PPO clipping. |

## How it plugs in

The wiring follows verl's pattern (see `verl.trainer.ppo.rollout_corr_helper`):

1. **Bypass entrypoint.** When `rollout_correction.bypass_mode=True`,
   `RayDiffusionTrainer` calls
   `verl_omni.trainer.diffusion.rollout_correction.apply_bypass_mode_to_diffusion_batch`,
   which sets `batch["old_log_probs"] = batch["rollout_log_probs"]` (zero-cost). The
   trainer-side correction is then **skipped** because `old == rollout` would make it a
   no-op; instead the trainer forwards the `rollout_correction` config to the engine via
   non-tensor metadata on the batch.
2. **Per-step in-engine correction (bypass).** Inside the FSDP engine's `forward_step`
   (see [`diffusers_impl.py`](../../verl_omni/workers/engine/fsdp/diffusers_impl.py)),
   `compute_per_step_rollout_correction` runs once per micro-batch / SDE step using
   `(current_log_prob, rollout_log_prob)`.  IS metrics are computed for diagnostics,
   but only the **RS rejection mask** is forwarded as ``rollout_is_weights`` (rejected
   samples get weight 0, kept samples get weight 1).  IS weights are intentionally
   **not** applied — the PPO ratio ``exp(current − rollout)`` already serves as the IS
   correction in bypass mode (see verl's ``compute_policy_loss_bypass_mode`` which
   passes ``rollout_is_weights=None`` for ``ppo_clip``).
3. **Decoupled (non-bypass) correction.** When `bypass_mode=False`,
   `apply_rollout_correction_to_diffusion_batch` runs once per global batch using the
   recomputed `old_log_probs` vs `rollout_log_probs` and stashes a single batch-level
   `rollout_is_weights` tensor combining **both** IS weights and RS mask. This path
   also emits the `rollout_corr/*` diagnostics.
4. **Loss application.** The registered losses (`flow_grpo`, `grpo_guard`) multiply the
   per-element PG loss by the (detached) ``rollout_is_weights`` before taking the mean.
   - **Decoupled mode**: weights = IS multiplier × RS keep-mask (``old/rollout`` ratio).
   - **Bypass mode**: weights = RS keep-mask only (0 or 1); the PPO ratio handles IS.
   Because diffusion has no padding, RS rejection is folded into the same tensor
   (rejected samples get weight 0) — no separate mask is required.

### Intentional deviation from verl

verl's `apply_bypass_mode` additionally swaps the policy loss to a dedicated
`bypass_mode` loss via `open_dict` on the policy-loss config.  verl-omni achieves
the same semantics without a dedicated loss function:

- **Bypass PPO-clip** (``loss_type=ppo_clip``, default): the PPO ratio
  ``exp(current − rollout)`` provides IS correction; ``compute_per_step_rollout_correction``
  only emits the RS mask, matching verl's ``rollout_is_weights=None``.
- **Bypass REINFORCE** (``loss_type=reinforce``, reserved): IS weights would be
  forwarded alongside the RS mask — no PPO clipping is applied.
- **Decoupled mode**: the trainer-side correction stashes genuine IS weights computed
  from ``old_log_probs`` vs ``rollout_log_probs``, and the loss function applies them
  uniformly — no loss-config mutation is needed.

The config field names, defaults, and ``loss_type`` semantics are kept identical to
verl's ``RolloutCorrectionConfig`` to simplify future rebasing.
