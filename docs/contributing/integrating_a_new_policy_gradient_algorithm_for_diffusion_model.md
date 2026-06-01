# How to Integrate a New Policy-Gradient Algorithm for Diffusion Model

Last updated: 06/02/2026.

This guide explains how to add a new PPO-like policy-gradient algorithm to
VeRL-Omni's diffusion trainer. Policy-gradient diffusion algorithms operate on
the reverse denoising trajectory and train from log-probability ratios plus
advantages. Examples include FlowGRPO, MixGRPO, and GRPO-Guard.

If your algorithm trains from final samples, rewards, or chosen/rejected
preferences instead of reverse-step logprobs, use
[`integrating_a_new_direct_preference_algorithm_for_diffusion_model.md`](integrating_a_new_direct_preference_algorithm_for_diffusion_model.md)
instead. That guide covers direct-preference algorithms such as offline DPO
and online DiffusionNFT.

The contracts described here are orthogonal to model integration: a single
policy-gradient algorithm can be extended to any number of model architectures
by pairing it with the `DiffusionModelBase` / `VllmOmniPipelineBase` adapters
described in
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md).

We use **FlowGRPO**
([Liu et al., 2025](https://arxiv.org/abs/2505.05470),
 [`verl_omni/pipelines/qwen_image_flow_grpo/`](../../verl_omni/pipelines/qwen_image_flow_grpo/__init__.py))
as the worked example throughout. It is the reference policy-gradient
algorithm in this repository and exercises every extension point.

---

## TL;DR

A new policy-gradient algorithm needs **five pieces**:

1. **An SDE step formula** for the rollout, usually a new `sde_type` in
   [`FlowMatchSDEDiscreteScheduler`](../../verl_omni/pipelines/schedulers/flow_match_sde.py),
   or a brand-new scheduler if the family changes.
2. **An advantage estimator** registered with `@register_diffusion_adv_est(...)`.
3. **A loss function** registered with `@register_diffusion_loss(...)`.
4. **One adapter pair per (architecture, algorithm) combination**: a
   `DiffusionModelBase` subclass and a `VllmOmniPipelineBase` subclass,
   both decorated with `@register(architecture, algorithm="<name>")`.
5. **PPO FSDP engine dispatch** through `actor_rollout_ref.model.model_type`.
   Policy-gradient algorithms normally reuse `PPODiffusersFSDPEngine`,
   registered as `model_type="diffusion_model"`.

Set `algorithm.trainer_type=policy_gradient` in launch scripts for
policy-gradient algorithms. The trainer entrypoint
([`main_diffusion.py`](../../verl_omni/trainer/main_diffusion.py)) then
selects `PolicyGradientRayTrainer`.

---

## Mental Model

VeRL-Omni layers algorithm dispatch on top of model dispatch. At runtime:

```text
   actor_rollout_ref.model.algorithm = "flow_grpo"    <- primary CLI flag
                | (OmegaConf template)               | (OmegaConf template)
   algorithm.adv_estimator = "flow_grpo"    actor_rollout_ref.actor.diffusion_loss.loss_mode = "flow_grpo"
                |                              |                              |
   DiffusionModelBase.get_class(arch, algo)    VllmOmniPipelineBase.get_class(arch, algo)
                |                              |
   QwenImage (training adapter)            QwenImagePipelineWithLogProb (rollout adapter)

   actor_rollout_ref.model.model_type = "diffusion_model"
                |
   PPODiffusersFSDPEngine

   loss_mode
                |
   FlowGRPOLoss
```

All four registries (`DiffusionModelBase`, `VllmOmniPipelineBase`,
`register_diffusion_adv_est`, `register_diffusion_loss`) are wired to
`actor_rollout_ref.model.algorithm` via OmegaConf templates, so a single CLI
flag selects everything **provided every site recognises the new name**. The
FSDP engine is selected separately via `actor_rollout_ref.model.model_type`.

If your algorithm reuses an existing estimator or loss without registering an
alias, explicitly pin those sites back to the existing name on the CLI; see
[Reusing an existing estimator or loss](#reusing-an-existing-estimator-or-loss).

---

## Step 1 — Pick or Add an SDE Step Formula

The training and rollout sides must agree on the formula used to sample the
previous denoising step under the policy. FlowGRPO uses
[`FlowMatchSDEDiscreteScheduler`](../../verl_omni/pipelines/schedulers/flow_match_sde.py)
with `sde_type="sde"`, which implements the standard flow-matching SDE from
the paper:

$$
x_{t-1} = x_t + \mathrm{d}t \cdot v_\theta(x_t, t) - \tfrac{1}{2}\,\sigma_t^2 \nabla_x \log p_t(x_t) \cdot \mathrm{d}t + \sigma_t \sqrt{|\mathrm{d}t|}\,\epsilon
$$

where `sigma_t = sqrt(σ_t/(1-σ_t)) · noise_level`.

If your algorithm reuses this family, call
`scheduler.sample_previous_step(..., sde_type="sde", noise_level=..., ...)`
from your training adapter and pass `sde_type=...` through to the rollout
loop. If your algorithm needs a different formula:

1. **Preferred**: add a new branch to
   `FlowMatchSDEDiscreteScheduler.sample_previous_step` keyed on a new
   `sde_type` literal. Keep all branches numerically consistent: compute
   `pred_original_sample`, then `prev_sample_mean`, then optionally a Gaussian
   log-prob.
2. **Fallback**: write a brand-new scheduler under
   `verl_omni/pipelines/schedulers/`. This is rarely necessary; the
   flow-matching family covers most published policy-gradient diffusion
   algorithms.

The scheduler must always return
`(prev_sample, log_prob, prev_sample_mean, std_dev_t)` in that order so the
trainer can compute the importance ratio without algorithm-specific glue.

---

## Step 2 — Register the Advantage Estimator

Open
[`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py)
and add a member to the `DiffusionAdvantageEstimator` enum, then register your
function with `@register_diffusion_adv_est(...)`:

```python
class DiffusionAdvantageEstimator(str, Enum):
    FLOW_GRPO = "flow_grpo"
    # ... add new entries here

@register_diffusion_adv_est(DiffusionAdvantageEstimator.FLOW_GRPO)
def compute_flow_grpo_outcome_advantage(
    sample_level_rewards: torch.Tensor,
    index: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: DiffusionAlgoConfig | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-normalised outcome advantage used by FlowGRPO."""
    ...
    return advantages, returns
```

The estimator receives `sample_level_rewards` and the group `index` (the
prompt UID). Return the `(advantages, returns)` pair as full-batch tensors.

If your estimator needs additional kwargs that are not already wired by
[`compute_advantage`](../../verl_omni/trainer/diffusion/ray_diffusion_trainer.py),
extend the matching branch in `ray_diffusion_trainer.compute_advantage` to
forward them.

---

## Step 3 — Register the Loss

Open
[`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py)
and add the pure loss function plus the registered worker-side adapter:

```python
@register_diffusion_loss("flow_grpo")
class FlowGRPOLoss(DiffusionLossFn):
    """FlowGRPO clipped policy objective."""

    required_model_output_keys = ("log_probs",)
    required_data_keys = ("old_log_probs", "advantages")

    @classmethod
    def compute_loss(
        cls,
        *,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        config: DiffusionActorConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Clipped-PPO objective averaged across denoising steps."""
        ...
        return pg_loss, pg_metrics

    def __call__(self, *, config, model_output, data) -> DiffusionLossResult:
        pg_loss, pg_metrics = self.compute_loss(
            old_log_prob=data["old_log_probs"],
            log_prob=model_output["log_probs"],
            advantages=data["advantages"],
            config=config,
        )
        return DiffusionLossResult(loss=pg_loss, metrics=pg_metrics)
```

Finally, add the loss name to the validation list in
[`DiffusionLossConfig.__post_init__`](../../verl_omni/workers/config/diffusion/actor.py):

```python
valid_modes = ["flow_grpo", "<your_new_algo>"]
```

---

## Step 4 — Write the (Architecture, Algorithm) Adapter Pair

For each model architecture you want to train under the new algorithm, add a
package under `verl_omni/pipelines/<arch>_<algo>/` and register both adapters:

```python
# verl_omni/pipelines/qwen_image_flow_grpo/diffusers_training_adapter.py
@DiffusionModelBase.register("QwenImagePipeline", algorithm="flow_grpo")
class QwenImage(DiffusionModelBase):
    ...
```

```python
# verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py
@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_grpo")
class QwenImagePipelineWithLogProb(QwenImagePipeline):
    ...
```

The adapter contracts are documented in
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md). For
policy-gradient algorithms, the rollout adapter returns reverse-trajectory
tensors such as `all_latents`, `all_timesteps`, `all_log_probs`, and prompt
embeds; the training adapter recomputes fresh logprobs for the actor update.

**Code reuse.** Algorithms in the same family typically share most adapter
code. Two patterns work well:

- **Promote helpers.** If FlowGRPO and your new algorithm share input
  preparation, move the common code to a shared module inside one of the
  packages, such as
  [`verl_omni/pipelines/qwen_image_flow_grpo/common.py`](../../verl_omni/pipelines/qwen_image_flow_grpo/common.py).
- **Subclass the rollout.** Rollout adapters are deep enough that subclassing
  is usually cleanest:

  ```python
  from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import (
      QwenImagePipelineWithLogProb,
  )

  @VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="my_algo")
  class QwenImageMyAlgoPipelineWithLogProb(QwenImagePipelineWithLogProb):
      def forward(self, req, *, sde_type="my_sde", sde_window_size=None, **kw):
          return super().forward(
              req,
              sde_type=sde_type,
              sde_window_size=sde_window_size,
              **kw,
          )
  ```

Finally, add a star-import to
[`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py) so
the registries learn about your package on import.

---

## Step 5 — Register or Reuse the FSDP Engine

Policy-gradient algorithms that consume reverse-trajectory logprob tensors
normally reuse `PPODiffusersFSDPEngine`, registered as
`model_type="diffusion_model"`. Set this explicitly in launch scripts if your
algorithm also registers other model types:

```bash
actor_rollout_ref.model.model_type=diffusion_model
```

Only register a new engine if the actor forward/backward batch contract
differs from PPO's `all_latents`, `all_timesteps`, `old_log_probs`, and
`advantages` contract. Direct-preference and forward-process algorithms follow
that separate path in
[`integrating_a_new_direct_preference_algorithm_for_diffusion_model.md`](integrating_a_new_direct_preference_algorithm_for_diffusion_model.md).

Do **not** register a new model type just because an algorithm reuses an
existing policy-gradient adapter with a different loss mode. For example,
GRPO-Guard can reuse the FlowGRPO model algorithm and select the guarded
objective through `actor_rollout_ref.actor.diffusion_loss.loss_mode`.

---

## Step 6 — Wire the Config Knobs

If your algorithm exposes new rollout knobs, add them to the
`DiffusionRolloutAlgoConfig` block in
[`diffusion_rollout.yaml`](../../verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml)
and to the matching dataclass in
[`verl_omni/workers/config/diffusion/rollout.py`](../../verl_omni/workers/config/diffusion/rollout.py).
Mirror them to the model-side block in
[`diffusion_model.yaml`](../../verl_omni/trainer/config/diffusion/model/diffusion_model.yaml)
using the `${oc.select:actor_rollout_ref.rollout.algo.<field>,<default>}`
pattern so a single CLI flag toggles both contexts.

The algorithm dispatch is already wired. Setting
`actor_rollout_ref.model.algorithm=<your_algo>` on the CLI:

- selects the `(architecture, algorithm)` adapter pair,
- propagates to `algorithm.adv_estimator` via
  `${oc.select:actor_rollout_ref.model.algorithm,flow_grpo}`, and
- propagates to `actor_rollout_ref.actor.diffusion_loss.loss_mode` via the
  same pattern.

Set the trainer type explicitly:

```bash
algorithm.trainer_type=policy_gradient
```

A single algorithm flag covers all dispatch points **only when every site
recognises the new name**. See the next subsection for reuse patterns.

### Reusing an existing estimator or loss

If your algorithm reuses an existing estimator and/or loss, the cascade above
will propagate your new algorithm name to those sites, and validators will
reject it unless you register or override the reused pieces:

- `DiffusionAdvantageEstimator` is a closed enum. `compute_advantage` fails to
  look up an unknown name.
- `DiffusionLossConfig.__post_init__` checks `loss_mode in valid_modes` and
  raises `ValueError` for anything not in the allowlist.
- `EngineRegistry` dispatches on `model_type`; an unrecognised `model_type`
  raises during worker construction.

You have two ways out:

1. **Pin the cascaded fields back to the existing name.** Add explicit
   overrides to your launch script and any documented YAML examples:

   ```bash
   algorithm.adv_estimator=<existing_estimator>
   actor_rollout_ref.model.algorithm=<your_algo>
   actor_rollout_ref.actor.diffusion_loss.loss_mode=<existing_loss>
   actor_rollout_ref.model.model_type=diffusion_model
   ```

   This is what
   [`examples/mixgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo.sh`](../../examples/mixgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo.sh)
   does for the estimator and loss. MixGRPO still uses the PPO engine contract.

2. **Register your name as an alias** in `diffusion_algos.py` and add
   `<your_algo>` to `DiffusionLossConfig.valid_modes`. Register or reuse the
   matching FSDP model type. The cascade then works without per-launch
   overrides.

---

## Step 7 — Example Launch Script

Add a runnable example under `examples/<algo>_trainer/`. Copy
[`examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh`](../../examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh)
and update the algorithm dispatch flags:

```bash
algorithm.trainer_type=policy_gradient \
actor_rollout_ref.model.algorithm=<your_algo> \
actor_rollout_ref.model.model_type=diffusion_model \
actor_rollout_ref.rollout.algo.sde_type=<your_sde_type> \
actor_rollout_ref.rollout.algo.noise_level=<noise_level> \
```

Document any algorithm-specific knobs in the example's `README.md`.

---

## Step 8 — Smoke Test

Add an end-to-end smoke test under `tests/special_e2e/` modelled on
[`tests/special_e2e/run_flowgrpo_qwen_image.sh`](../../tests/special_e2e/run_flowgrpo_qwen_image.sh).

Name the script `run_<algo>_<model>.sh`, for example
`run_flowgrpo_qwen_image.sh`. Register it in
[`tests/gpu_smoke/run_gpu_smoke_tests.sh`](../../tests/gpu_smoke/run_gpu_smoke_tests.sh)
as a new numbered test entry. The script must exercise the policy-gradient
dispatch chain: advantage estimator, loss, adapter pair, FSDP engine, and SDE
step against a `tiny-random/<ModelName>` checkpoint.

---

## Final Checklist

- [ ] SDE step formula available: either an existing `sde_type` works, or a
      new branch / scheduler is added under `verl_omni/pipelines/schedulers/`.
- [ ] `DiffusionAdvantageEstimator.<NAME>` enum entry added and the estimator
      function is registered with `@register_diffusion_adv_est(...)`.
- [ ] Loss function registered with `@register_diffusion_loss("<name>")` and
      added to `DiffusionLossConfig.valid_modes`.
- [ ] One `(architecture, algorithm)` adapter pair per supported model, both
      decorated with `@register(architecture, algorithm="<name>")`.
- [ ] Policy-gradient algorithm uses `algorithm.trainer_type=policy_gradient`
      and `actor_rollout_ref.model.model_type=diffusion_model`, unless it has
      a truly different actor batch contract.
- [ ] `verl_omni/pipelines/__init__.py` star-imports the new package.
- [ ] Any new rollout algorithm field is mirrored in both
      [`diffusion_rollout.yaml`](../../verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml)
      and
      [`diffusion_model.yaml`](../../verl_omni/trainer/config/diffusion/model/diffusion_model.yaml).
- [ ] Example launch script under `examples/<algo>_trainer/`.
- [ ] Smoke test under `tests/special_e2e/run_<algo>_<model>.sh` wired into
      `tests/gpu_smoke/run_gpu_smoke_tests.sh`.
- [ ] If the registry or adapter contract changed, update
      [`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md) to
      match.
