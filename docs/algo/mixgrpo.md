# Mix-GRPO

Last updated: 05/12/2026.

Mix-GRPO ([paper](https://arxiv.org/abs/2507.21802),
[code](https://github.com/Tencent-Hunyuan/MixGRPO)) extends Flow-GRPO with a
**Mixed ODE-SDE rollout** and a **sliding-window training schedule** that
together greatly cut the cost of online RL fine-tuning of flow-matching
diffusion models.

* The rollout uses **deterministic ODE sampling outside a contiguous window**
  of denoising steps and **stochastic SDE sampling inside the window** -- only
  the in-window steps yield meaningful log-probabilities and contribute to the
  policy gradient.
* A trainer-side scheduler **slides the window across training iterations**,
  so over the course of training every part of the trajectory is exercised
  while each individual rollout still pays SDE cost on a small fraction of
  the trajectory.

In practice this lets you keep a short inference horizon (e.g. 10 steps) for
fast iteration while MixGRPO's ODE/SDE split reduces gradient variance, or
scale to a longer horizon (e.g. 50 steps) and still train at the cost of a
small SDE window per rollout.

## How verl-omni implements MixGRPO

MixGRPO shares the SDE step formula, advantage estimator and loss with
FlowGRPO, so only the `(architecture, algorithm)` adapter pair changes.

| Layer | What it does | Code |
|---|---|---|
| Algo config | Two extra knobs (`sample_strategy`, `iters_per_group`) on the existing `DiffusionRolloutAlgoConfig`. | `verl_omni/workers/config/diffusion/rollout.py` |
| Adapter pair | Subclasses of the FlowGRPO adapters re-registered under `algorithm="mix_grpo"`; the rollout adapter materialises the deterministic window for `progressive`. | `verl_omni/pipelines/qwen_image_mix_grpo/` |
| Rollout | Already supports a contiguous SDE window (ODE outside / SDE inside) -- no changes needed. | `verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py` |

## Configuration

Algorithm dispatch lives on `actor_rollout_ref.model.algorithm`; everything
else is rollout configuration under `actor_rollout_ref.rollout.algo`.

Because MixGRPO reuses FlowGRPO's advantage estimator and PPO loss verbatim,
you must also pin the two cascaded fields back to `flow_grpo` so they don't
propagate the unknown `mix_grpo` token into the validator at
[`DiffusionLossConfig.valid_modes`](../../verl_omni/workers/config/diffusion/actor.py)
and the [`DiffusionAdvantageEstimator`](../../verl_omni/trainer/diffusion/diffusion_algos.py)
enum (see [Caveat: cascade vs. validators](#caveat-cascade-vs-validators) below):

```yaml
algorithm:
  adv_estimator: flow_grpo            # MixGRPO reuses FlowGRPO's estimator
actor_rollout_ref:
  model:
    algorithm: mix_grpo               # selects the (arch, algo) adapter pair
  actor:
    diffusion_loss:
      loss_mode: flow_grpo            # MixGRPO reuses FlowGRPO's loss
  rollout:
    algo:
      # ----- Common SDE configs ---------------------------------------------
      noise_level: 1.0                # SDE noise magnitude
      sde_type: sde                   # sde | cps
      sde_window_size: null           # window length / "group size"
      sde_window_range: null          # [start, end] envelope; null = full trajectory

      # ----- MixGRPO sliding-window scheduler (mix_grpo only) -------------
      sample_strategy: random         # random | progressive
      iters_per_group: 1              # progressive only
      sde_window_seed: 0              # random only
```

### Caveat: cascade vs. validators

`actor_rollout_ref.model.algorithm` is wired to *four* dispatch points via
OmegaConf templates of the form
`${oc.select:actor_rollout_ref.model.algorithm,flow_grpo}`:

1. The `(architecture, algorithm)` adapter pair lookups
   (`DiffusionModelBase` and `VllmOmniPipelineBase`).
2. `algorithm.adv_estimator`.
3. `actor_rollout_ref.actor.diffusion_loss.loss_mode`.

Setting `actor_rollout_ref.model.algorithm=mix_grpo` therefore propagates
`mix_grpo` to all four sites. Adapter dispatch (1) is happy — both adapters
are registered under `algorithm="mix_grpo"`. The other two points fail at
runtime because:

* `DiffusionAdvantageEstimator` only enumerates `flow_grpo`, so
  `compute_advantage` raises when it tries to look up `mix_grpo`.
* `DiffusionLossConfig.__post_init__` checks `loss_mode in ["flow_grpo"]`
  and raises `ValueError: Invalid diffusion loss_mode: mix_grpo`.

Pinning `algorithm.adv_estimator=flow_grpo` and
`actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo` keeps the
adapter dispatch on `mix_grpo` while the estimator/loss stay on the
existing FlowGRPO implementations.

### Field semantics

* **`actor_rollout_ref.model.algorithm`** -- selects the registered
  `(architecture, algorithm)` adapter pair. `mix_grpo` routes to
  [`verl_omni/pipelines/qwen_image_mix_grpo/`](../../verl_omni/pipelines/qwen_image_mix_grpo/__init__.py).
* **`noise_level`** -- magnitude of injected SDE noise inside the window.
  Outside the window `noise_level` is forced to `0` so the step degenerates
  to a deterministic Euler ODE step.
* **`sde_type`** -- `sde` (FlowGRPO formulation) or `cps`
  (Coefficients-Preserving Sampling).
* **`sde_window_size`** -- length of the active SDE window, called
  "group size" in MixGRPO. `null` means "use the entire trajectory" (the
  legacy FlowGRPO setting).
* **`sde_window_range`** -- a `[start, end]` envelope of valid window-start
  positions:
  * Under `random`, the rollout backend draws the start uniformly from
    `[start, end - sde_window_size + 1)`.
  * Under `progressive`, the same envelope clamps the deterministic
    sliding window.
  * `null` defaults to the full trajectory `[0, num_inference_steps]`
    (minus the last ODE step where `sigma_prev = 0`).
* **`sample_strategy`** -- *MixGRPO only*. `random` draws a fresh window per
  step (seeded so all ranks agree); `progressive` advances the window by
  `sde_window_size` every `iters_per_group` iterations.
* **`iters_per_group`** -- *MixGRPO progressive only*. Number of training
  iterations spent at each window position.
* **`sde_window_seed`** -- *MixGRPO random only*. Base seed for the per-step random
  window draws. Distinct from the rollout generator seed
  (`val_kwargs.seed`) so the two random streams stay decoupled.

### Validation

Validation always uses the deterministic ODE path with `noise_level=0`, so
the sliding-window settings are irrelevant there.

## Reference recipe

A ready-to-run script is provided at
`examples/mixgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_mixgrpo.sh`. The default
config uses a **10-step trajectory with a 2-step window** (`random` strategy),
matching the FlowGRPO baseline's inference budget:

```bash
algorithm.adv_estimator=flow_grpo
actor_rollout_ref.model.algorithm=mix_grpo
actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo
actor_rollout_ref.rollout.algo.sample_strategy=random
actor_rollout_ref.rollout.algo.sde_window_seed=42
actor_rollout_ref.rollout.algo.sde_window_size=2
actor_rollout_ref.rollout.algo.sde_window_range=[0,5]
actor_rollout_ref.rollout.algo.noise_level=1.2
actor_rollout_ref.rollout.algo.sde_type=sde
```

The first and third lines pin the cascaded estimator/loss back to
`flow_grpo`; see [Caveat: cascade vs. validators](#caveat-cascade-vs-validators).


## Tuning guide

The two most impactful parameters are **`num_inference_steps`** (rollout
trajectory length) and **`sde_window_size`** (how many steps use SDE).

| Setting | `num_inference_steps` | `sde_window_size` | `sample_strategy` | Speed | Quality |
|---|---|---|---|---|---|
| Fast (default) | 10 | 2 | `random` | ~7 min/step | Good — matches FlowGRPO budget |
| Long trajectory | 50 | 4 | `progressive` | ~23 min/step | Higher reward baseline, but gradients are diluted (only 8% of trajectory is SDE) |

**Guidelines:**

* **Start with the default** (10 steps, window 2). This gives the fastest
  iteration and strongest learning signal per step because a larger fraction
  of the trajectory contributes to gradients.
* **Increase `num_inference_steps`** (e.g. 50) when image quality at rollout
  time is important and you can afford the wall-clock cost. Pair with a
  proportionally larger `sde_window_size` (e.g. 4) to keep the gradient
  signal strong.
* **`sde_window_size / num_inference_steps` ratio** controls the trade-off:
  a higher ratio means more gradient signal per step but higher SDE cost;
  a lower ratio is cheaper but gradients are noisier.
* **`sample_strategy`**: use `random` for short trajectories (window
  positions are already well-covered); use `progressive` with
  `iters_per_group` for long trajectories to ensure systematic coverage.
* **Validation** always uses the deterministic ODE path (`noise_level=0`)
  regardless of training settings.

## References

* MixGRPO: J. Li *et al.*, *MixGRPO: Unlocking Flow-based GRPO Efficiency
  with Mixed ODE-SDE*, arXiv:2507.21802.
* MixGRPO repo: <https://github.com/Tencent-Hunyuan/MixGRPO>.
* FlowGRPO: Y. Liu *et al.*, *Flow-GRPO: Training Flow Matching Models via
  Online RL*, arXiv:2505.05470.
* Coefficients-Preserving Sampling: arXiv:2509.05952.
