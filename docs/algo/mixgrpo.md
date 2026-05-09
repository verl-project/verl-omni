# Mix-GRPO

Last updated: 05/06/2026.

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

The integration is **fully decoupled from the model and rollout pipeline**.
Switching algorithms only requires flipping a YAML field; the underlying
`FlowMatchSDEDiscreteScheduler` and `vllm_omni_rollout_adapter` already
support a contiguous SDE window and just consume per-step overrides.

| Layer | What it does | Code |
|---|---|---|
| Algo config | `algo_type` selector (`flow_grpo` / `mix_grpo`) plus a small set of MixGRPO configs. | `verl_omni/workers/config/diffusion/rollout.py` |
| Trainer | Builds an `SDEWindowScheduler` from the algo config and queries it every step to inject `sde_window_size` / `sde_window_range` overrides. | `verl_omni/trainer/diffusion/sde_window_scheduler.py`, `RayFlowGRPOTrainer.fit()` |
| Agent loop | Merges the per-step overrides from `meta_info["algo_overrides"]` into the rollout sampling params. | `verl_omni/agent_loop/diffusion_agent_loop.py` |
| Rollout | Already supports a contiguous SDE window (ODE outside / SDE inside) -- no changes needed. | `verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py` |

## Configuration

All configs live under `actor_rollout_ref.rollout.algo`. Trainer-only fields
(`algo_type`, `sample_strategy`, `iters_per_group`, `seed`) are stripped by
the agent loop before reaching the rollout backend, so the rollout API stays
unchanged.

The full surface is **8 fields**:

```yaml
actor_rollout_ref:
  rollout:
    algo:
      # ----- Selector (both algorithms) ------------------------------------
      algo_type: flow_grpo            # flow_grpo | mix_grpo

      # ----- Common SDE configs ---------------------------------------------
      noise_level: 1.0                # SDE noise magnitude
      sde_type: sde                   # sde | cps
      sde_window_size: null           # window length / "group size"
      sde_window_range: null          # [lo, hi] envelope; null = full trajectory

      # ----- MixGRPO sliding-window scheduler (mix_grpo only) -------------
      sample_strategy: random         # random | progressive
      iters_per_group: 1              # progressive only
      sde_window_seed: 0              # random only
```

### Field semantics

* **`algo_type`** -- `flow_grpo` (default) keeps the legacy rollout-side
  random-window behaviour; `mix_grpo` enables the trainer-side sliding
  scheduler.
* **`noise_level`** -- magnitude of injected SDE noise inside the window.
  Outside the window `noise_level` is forced to `0` so the step degenerates
  to a deterministic Euler ODE step.
* **`sde_type`** -- `sde` (FlowGRPO formulation) or `cps`
  (Coefficients-Preserving Sampling).
* **`sde_window_size`** -- length of the active SDE window, called
  "group size" in MixGRPO. `null` means "use the entire trajectory" (the
  legacy FlowGRPO setting).
* **`sde_window_range`** -- a `[lo, hi]` envelope of valid window-start
  positions:
  * For `flow_grpo`, the rollout backend draws the start uniformly from
    `[lo, hi - sde_window_size + 1)`.
  * For `mix_grpo`, this is the eligible range over which the trainer-side
    scheduler slides the window.
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
the MixGRPO-specific fields are irrelevant there. The default config keeps
`actor_rollout_ref.rollout.val_kwargs.algo.algo_type=flow_grpo`.

## Reference recipe

A ready-to-run script is provided at
`examples/flowgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo.sh`. The default
config uses a **10-step trajectory with a 2-step window** (`random` strategy),
matching the FlowGRPO baseline's inference budget:

```bash
actor_rollout_ref.rollout.algo.algo_type=mix_grpo
actor_rollout_ref.rollout.algo.sample_strategy=random
actor_rollout_ref.rollout.algo.sde_window_seed=42
actor_rollout_ref.rollout.algo.sde_window_size=2
actor_rollout_ref.rollout.algo.sde_window_range=[0,5]
actor_rollout_ref.rollout.algo.noise_level=1.2
actor_rollout_ref.rollout.algo.sde_type=sde
```

To switch back to the FlowGRPO baseline keep everything else the same and
override `actor_rollout_ref.rollout.algo.algo_type=flow_grpo`.

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

## Switching from FlowGRPO

Existing FlowGRPO configs continue to work unchanged because
`algo_type=flow_grpo` is the default and the FlowGRPO scheduler simply
forwards the static `sde_window_size` / `sde_window_range` already used by
the rollout backend.

## References

* MixGRPO: J. Li *et al.*, *MixGRPO: Unlocking Flow-based GRPO Efficiency
  with Mixed ODE-SDE*, arXiv:2507.21802.
* MixGRPO repo: <https://github.com/Tencent-Hunyuan/MixGRPO>.
* FlowGRPO: Y. Liu *et al.*, *Flow-GRPO: Training Flow Matching Models via
  Online RL*, arXiv:2505.05470.
* Coefficients-Preserving Sampling: arXiv:2509.05952.
