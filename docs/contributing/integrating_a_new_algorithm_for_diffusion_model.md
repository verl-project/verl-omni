# How to Integrate a New Diffusion RL Algorithm

Last updated: 06/01/2026.

This guide explains how to add a new diffusion RL algorithm to VeRL-Omni's
diffusion trainer. New algorithms must first decide whether they are
policy-gradient reverse-trajectory algorithms or direct-preference
forward-process algorithms. The contracts described here are orthogonal to
model integration: a single algorithm can be extended to any number of model
architectures by pairing it with the
`DiffusionModelBase` / `VllmOmniPipelineBase` adapters described in
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md).

We use **FlowGRPO**
([Liu et al., 2025](https://arxiv.org/abs/2505.05470),
 [`verl_omni/pipelines/qwen_image_flow_grpo/`](../../verl_omni/pipelines/qwen_image_flow_grpo/__init__.py))
as the worked example throughout — it is the reference algorithm in this
repository and exercises every extension point.

---

## TL;DR

A new PPO-like algorithm needs **five pieces**:

1. **An SDE step formula** for the rollout — usually a new `sde_type` in
   [`FlowMatchSDEDiscreteScheduler`](../../verl_omni/pipelines/schedulers/flow_match_sde.py),
   or a brand-new scheduler if the family changes.
2. **An advantage estimator** registered with `@register_diffusion_adv_est(...)`.
3. **A loss function** registered with `@register_diffusion_loss(...)`.
4. **One adapter pair per (architecture, algorithm) combination** — a
   `DiffusionModelBase` subclass and a `VllmOmniPipelineBase` subclass,
   both decorated with `@register(architecture, algorithm="<name>")`.
5. **An FSDP training engine selection** registered directly with
   `@EngineRegistry.register(model_type=…)`. PPO-like algorithms reuse
   `PPODiffusersFSDPEngine` (registered as `model_type="diffusion_model"`);
   direct-preference / forward-process algorithms register their own
   `DiffusersFSDPEngine` subclass under a distinct `model_type`.

The trainer entrypoint
([`main_diffusion.py`](../../verl_omni/trainer/main_diffusion.py))
and the Ray driver
([`ray_diffusion_trainer.py`](../../verl_omni/trainer/diffusion/ray_diffusion_trainer.py))
dispatch on model/loss registry strings above, plus two orthogonal algorithm
config fields:

| Field | Values | Purpose |
|-------|--------|---------|
| `algorithm.trainer_type` | `policy_gradient`, `direct_preference` | Selects `PolicyGradientRayTrainer` (FlowGRPO, MixGRPO, …) vs `DirectPreferenceRayTrainer` (DPO, DiffusionNFT, ...) |
| `algorithm.sample_source` | `online`, `offline` | `BaseRayDiffusionTrainer.init_workers` skips rollout/reward engine init when `offline` |


---

## Mental Model

VeRL-Omni layers algorithm dispatch on top of model dispatch. At
runtime:

```text
   actor_rollout_ref.model.algorithm = "flow_grpo"    ← primary CLI flag
                ↓ (OmegaConf template)               ↓ (OmegaConf template)
   algorithm.adv_estimator = "flow_grpo"    actor_rollout_ref.actor.diffusion_loss.loss_mode = "flow_grpo"
                ↓                              ↓                              ↓
   DiffusionModelBase.get_class(arch, algo)    VllmOmniPipelineBase.get_class(arch, algo)
                ↓                              ↓
   QwenImage (training adapter)            QwenImagePipelineWithLogProb (rollout adapter)

   EngineRegistry.get(model_type, backend, device)
                ↓
   PPODiffusersFSDPEngine or an algorithm-specific DiffusersFSDPEngine subclass

   loss_mode
                ↓
   FlowGRPOLoss
```

All four registries (`DiffusionModelBase`, `VllmOmniPipelineBase`,
`register_diffusion_adv_est`, `register_diffusion_loss`) are wired to
`actor_rollout_ref.model.algorithm` via OmegaConf templates, so a single
CLI flag selects everything **provided every site recognises the new
name**. The FSDP engine is selected via `actor_rollout_ref.model.model_type`
(see Step 5). If your algorithm reuses an existing estimator or loss without
registering an alias, you must explicitly pin those sites back to the
existing name on the CLI; see
[Reusing an existing estimator or loss](#reusing-an-existing-estimator-or-loss)
below.

## PPO-like vs Direct-preference Diffusion Algorithms

Before adding files, classify the algorithm's training contract.

**PPO-like algorithms** operate on the reverse denoising trajectory and use
policy-gradient likelihoods. Examples include FlowGRPO, MixGRPO, and
GRPO-Guard. Their rollout batch contains trajectory tensors such as
`all_latents`, `all_timesteps`, rollout or recomputed `old_log_probs`, and
optional reference reverse logprobs. They use `PolicyGradientRayTrainer`, 
`PPODiffusersFSDPEngine.forward_backward_batch()` / reverse `forward_step()`
and their losses consume logprob-like tensors plus per-timestep advantages.

**Direct-preference algorithms** train from final samples or preferences and define a
separate forward-process objective. Examples include DiffusionNFT,
Diffusion-DPO, DGPO/GPO, and AWM. Their rollout batch should contain the final
clean latent (`latents_clean`), sample-level rewards or preference pairs, and
an explicit forward-training timestep tensor (`train_timesteps`). They use
`DirectPreferenceRayTrainer` and an algorithm-specific FSDP engine such as
`NFTDiffusersFSDPEngine`, and their losses consume prediction-space tensors
rather than reverse-step logprobs.

---

## Step 1 — Pick or Add an SDE Step Formula

The training and rollout sides must agree on the formula used to sample
the previous denoising step under the policy. FlowGRPO uses
[`FlowMatchSDEDiscreteScheduler`](../../verl_omni/pipelines/schedulers/flow_match_sde.py)
with `sde_type="sde"`, which implements the standard flow-matching SDE
from the paper:

$$
x_{t-1} = x_t + \mathrm{d}t \cdot v_\theta(x_t, t) - \tfrac{1}{2}\,\sigma_t^2 \nabla_x \log p_t(x_t) \cdot \mathrm{d}t + \sigma_t \sqrt{|\mathrm{d}t|}\,\epsilon
$$

where `sigma_t = sqrt(σ_t/(1-σ_t)) · noise_level`.

If your algorithm reuses this family, simply call
`scheduler.sample_previous_step(..., sde_type="sde", noise_level=..., ...)`
from your training adapter and pass `sde_type=...` through to the rollout
loop. If your algorithm needs a different formula:

1. **Preferred** — add a new branch to
   `FlowMatchSDEDiscreteScheduler.sample_previous_step` keyed on a new
   `sde_type` literal. Keep all branches numerically consistent (compute
   `pred_original_sample`, then `prev_sample_mean`, then optionally a
   Gaussian log-prob).
2. **Fallback** — write a brand-new scheduler under
   `verl_omni/pipelines/schedulers/`. This is rarely necessary; the
   flow-matching family covers most published PPO-like diffusion
   algorithms.

The scheduler must always return
`(prev_sample, log_prob, prev_sample_mean, std_dev_t)` in that order so
the trainer can compute the importance ratio without algorithm-specific
glue.

---

## Step 2 — Register the Advantage Estimator

Open
[`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py)
and add a member to the `DiffusionAdvantageEstimator` enum, then register
your function with `@register_diffusion_adv_est(...)`:

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

The estimator receives `sample_level_rewards` (shape `(B,)`) and the
group `index` (the prompt UID). Return the `(advantages, returns)` pair
as full-batch tensors.

If your new algorithm reuses an existing estimator verbatim, just set
`algorithm.adv_estimator=<existing_name>` in your launch script.

If your estimator needs additional kwargs that are not already wired by
[`compute_advantage`](../../verl_omni/trainer/diffusion/ray_diffusion_trainer.py),
extend the `if adv_estimator == DiffusionAdvantageEstimator.<NAME>:` branch in
`ray_diffusion_trainer.compute_advantage` to forward them.

---

## Step 3 — Register the Loss

Open
[`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py)
and add the pure loss function plus the registered worker-side adapter:

```python
@register_diffusion_loss("flow_grpo")
class FlowGRPOLoss(DiffusionLossFn):
    """Flow-GRPO clipped policy objective."""

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

For each model architecture you want to train under the new algorithm,
add a package under
`verl_omni/pipelines/<arch>_<algo>/` and register both adapters:

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

The adapter contracts (the four `DiffusionModelBase` classmethods, the
rollout `forward()` shape) are documented in
[`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md);
nothing about them changes when you swap algorithms.

**Code reuse.** Algorithms in the same family typically share most
adapter code. Two patterns work well:

- **Promote helpers.** If FlowGRPO and your new algorithm share input
  preparation, move the common code to a shared module inside one of the
  packages (e.g.
  [`verl_omni/pipelines/qwen_image_flow_grpo/common.py`](../../verl_omni/pipelines/qwen_image_flow_grpo/common.py))
  and import it from both packages.
- **Subclass the rollout.** Rollout adapters are deep enough that
  subclassing is usually cleanest:

  ```python
  from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import (
      QwenImagePipelineWithLogProb,
  )

  @VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="my_algo")
  class QwenImageMyAlgoPipelineWithLogProb(QwenImagePipelineWithLogProb):
      def forward(self, req, *, sde_type="my_sde", sde_window_size=None, **kw):
          return super().forward(req, sde_type=sde_type,
                                 sde_window_size=sde_window_size, **kw)
  ```

Finally, add a star-import to
[`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py)
so the registries learn about your package on import.

---

## Step 5 — Register the FSDP Engine

Open
[`verl_omni/workers/engine/fsdp/diffusers_impl.py`](../../verl_omni/workers/engine/fsdp/diffusers_impl.py)
and register the engine that should execute the actor forward/backward path
for your algorithm. Each engine is registered directly with
`@EngineRegistry.register(model_type=…, backend=…, device=…)`; the caller
selects the correct engine by setting `actor_rollout_ref.model.model_type`
in their launch script.

If your algorithm is PPO-like and consumes reverse-trajectory logprob tensors
(`all_latents`, `all_timesteps`, `old_log_probs`, `advantages`), it reuses
`PPODiffusersFSDPEngine` (already registered as `model_type="diffusion_model"`)
— no new engine class is needed. Just set `model_type=diffusion_model` in your
launch script.

If the forward/backward batch contract differs, create a sibling engine and
register it under a new `model_type`:

```python
@EngineRegistry.register(model_type="<your_algo>_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class MyAlgoDiffusersFSDPEngine(DiffusersFSDPEngine):
    """FSDP engine for <your_algo>."""

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        ...

    def prepare_model_inputs(self, micro_batch, step: int):
        ...

    def prepare_model_outputs(self, output, micro_batch):
        ...

    def forward_step(self, micro_batch, loss_function, forward_only, step):
        ...
```

Then add `actor_rollout_ref.model.model_type=<your_algo>_model` to your launch
script. This is the pattern used by DPO (`diffusion_dpo_model`) and DiffusionNFT
(`diffusion_nft_model`).

Direct-preference / forward-process algorithms follow this second path.
For example, DiffusionNFT registers `NFTDiffusersFSDPEngine` as
`model_type="diffusion_nft_model"` because its batch contains `latents_clean`,
`train_timesteps`, and `reward_prob`, not PPO's reverse-step logprob tensors.

**Direct-preference algorithms** (DPO, DiffusionNFT, and similar) use
`DirectPreferenceRayTrainer`, which is designed to accommodate both offline
(DPO) and online (DiffusionNFT) training via config flags rather than
algorithm-name branches. The relevant trainer-side config knobs are:

| Config key | Purpose |
|---|---|
| `algorithm.sample_source` | `offline` or `online` — selects data flow |
| `algorithm.paired_preference` | `true` for paired chosen/rejected data (DPO); doubles actor batch size and disables shuffle |
| `actor_rollout_ref.model.policy_state_adapters` | Include `"old"` to enable old-adapter management (EMA/copy update after each actor step) |

If your algorithm needs custom actor batch preparation (e.g. computing
group-relative advantages from rollout rewards before passing the batch to
the actor), override the `prepare_actor_batch` static method on your
`DiffusionLossFn` subclass. The base implementation is a no-op (the batch
is passed through unchanged, as for DPO where the dataset already contains
everything the actor needs). The trainer resolves the active loss class from
config at init and dispatches through it — no changes to the trainer are
required. See `DiffusionNFTLoss.prepare_actor_batch` in
`verl_omni/trainer/diffusion/diffusion_algos.py` for a reference
implementation.

If your algorithm needs multiple LoRA policy states (for example `default`
and `old`), declare them with `actor_rollout_ref.model.policy_state_adapters`
and reuse the shared `LoRAAdapterMixin` helpers for adapter selection, copy,
and EMA updates instead of adding engine-specific adapter plumbing.

Do **not** register a name just because it is a loss mode. For example,
GRPO-Guard currently reuses the `flow_grpo` model algorithm and selects the
guarded objective through `actor_rollout_ref.actor.diffusion_loss.loss_mode`;
it does not need its own FSDP engine registration unless it also adds a
distinct model algorithm and adapter pair.

---

## Step 6 — Wire the Config Knobs

If your algorithm exposes new rollout knobs (e.g. an `sde_window_size`),
add them to the `DiffusionRolloutAlgoConfig` block in
[`diffusion_rollout.yaml`](../../verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml)
and to the matching dataclass in
[`verl_omni/workers/config/diffusion/rollout.py`](../../verl_omni/workers/config/diffusion/rollout.py).
Mirror them to the model-side block in
[`diffusion_model.yaml`](../../verl_omni/trainer/config/diffusion/model/diffusion_model.yaml)
using the `${oc.select:actor_rollout_ref.rollout.algo.<field>,<default>}`
pattern so a single CLI flag toggles both contexts.

The algorithm dispatch is already wired. Setting
`actor_rollout_ref.model.algorithm=<your_algo>` on the CLI:

- selects the `(architecture, algorithm)` adapter pair (Step 4),
- selects the FSDP engine registered for `<your_algo>` (Step 5),
- propagates to `algorithm.adv_estimator` via
  `${oc.select:actor_rollout_ref.model.algorithm,flow_grpo}`, and
- propagates to `actor_rollout_ref.actor.diffusion_loss.loss_mode` via
  the same pattern.

A single flag covers all five dispatch points **only when every site
recognises the new name** — see the next subsection for the alternative.

Set the trainer type explicitly for every algorithm:

```bash
# PPO-style reverse-trajectory algorithms
algorithm.trainer_type=policy_gradient

# direct-preference / forward-process algorithms
algorithm.trainer_type=direct_preference
```

Keep rollout data-contract knobs in the rollout config, worker-loss knobs in
the actor loss config, model state knobs in the model config, and
shared algorithm-level knobs under `algorithm`. Algorithm-specific
trainer-side knobs belong under `algorithm`. Loss-specific worker-side knobs
belong under `actor_rollout_ref.actor.diffusion_loss`.

### Reusing an existing estimator or loss

If your algorithm reuses an existing estimator and/or loss (for example,
MixGRPO uses FlowGRPO's verbatim), the cascade above will propagate your
new algorithm name to those sites, and the validators will reject it unless
you register or override the reused pieces:

* `DiffusionAdvantageEstimator` is a closed enum — `compute_advantage`
  fails to look up an unknown name.
* `DiffusionLossConfig.__post_init__` checks `loss_mode in valid_modes`
  and raises `ValueError` for anything not in the allowlist.
* `EngineRegistry` dispatches on `model_type`; an unrecognised `model_type`
  raises during worker construction.

You have two ways out, pick whichever is cleaner for your algorithm:

1. **Pin the cascaded fields back to the existing name.** Add explicit
   overrides to your launch script and any documented YAML examples:

   ```bash
   algorithm.adv_estimator=<existing_estimator>
   actor_rollout_ref.model.algorithm=<your_algo>
   actor_rollout_ref.actor.diffusion_loss.loss_mode=<existing_loss>
   ```

   This is what
   [`examples/mixgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo.sh`](../../examples/mixgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo.sh)
   does for the estimator / loss. MixGRPO still registers its model algorithm
   with `PPODiffusersFSDPEngine` because it uses the PPO engine contract.

2. **Register your name as an alias** in `diffusion_algos.py` (decorate
   the existing function with both names) and add `<your_algo>` to
   `DiffusionLossConfig.valid_modes`. Register the same model algorithm with
   the appropriate FSDP engine. The cascade then "just works" without
   per-launch overrides.

---

## Step 7 — Example Launch Script

Add a runnable example under `examples/<algo>_trainer/`. Copy
[`examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh`](../../examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh)
and update the algorithm dispatch flags:

```bash
actor_rollout_ref.model.algorithm=<your_algo> \
actor_rollout_ref.rollout.algo.sde_type=<your_sde_type> \
actor_rollout_ref.rollout.algo.noise_level=<noise_level> \
```

Document any algorithm-specific knobs in the example's `README.md`.

---

## Step 8 — Smoke Test

Add an end-to-end smoke test under `tests/special_e2e/` modelled on
[`tests/special_e2e/run_flowgrpo_qwen_image.sh`](../../tests/special_e2e/run_flowgrpo_qwen_image.sh)
(PPO-like algorithms) or
[`tests/special_e2e/run_diffusionnft_qwen_image.sh`](../../tests/special_e2e/run_diffusionnft_qwen_image.sh)
(direct-preference algorithms).

Name the script `run_<algo>_<model>.sh` (for example,
`run_flowgrpo_qwen_image.sh` or `run_diffusionnft_qwen_image.sh`).

Register the script in
[`tests/gpu_smoke/run_gpu_smoke_tests.sh`](../../tests/gpu_smoke/run_gpu_smoke_tests.sh)
as a new numbered test entry. The script must exercise the full
algorithm dispatch chain (adv estimator + loss + adapter pair + FSDP engine
and SDE step, or the direct-preference forward-process contract) against a
`tiny-random/<ModelName>` checkpoint.

---

## Final Checklist

- [ ] SDE step formula available — either an existing `sde_type` works, or
      a new branch / scheduler is added under
      `verl_omni/pipelines/schedulers/`.
- [ ] `DiffusionAdvantageEstimator.<NAME>` enum entry added and the
      estimator function is registered with
      `@register_diffusion_adv_est(...)`.
- [ ] Loss function registered with `@register_diffusion_loss("<name>")`
      and added to `DiffusionLossConfig.valid_modes`.
- [ ] One `(architecture, algorithm)` adapter pair per supported model,
      both decorated with `@register(architecture, algorithm="<name>")`.
- [ ] FSDP engine registered via `@EngineRegistry.register(model_type=…)` —
      either reuses `PPODiffusersFSDPEngine` (`model_type="diffusion_model"`)
      or registers an algorithm-specific subclass under a new `model_type`.
      Launch script sets `actor_rollout_ref.model.model_type` accordingly.
- [ ] Direct-preference algorithms: set `algorithm.trainer_type=direct_preference` and
      configure the relevant flags (`sample_source`, `paired_preference`,
      `policy_state_adapters`). Override `_prepare_actor_batch` in a
      `DirectPreferenceRayTrainer` subclass if the rollout batch needs
      algorithm-specific preparation before the actor update.
- [ ] `verl_omni/pipelines/__init__.py` star-imports the new package.
- [ ] Any new rollout algorithm field is mirrored in both
      [`diffusion_rollout.yaml`](../../verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml)
      and
      [`diffusion_model.yaml`](../../verl_omni/trainer/config/diffusion/model/diffusion_model.yaml).
- [ ] Example launch script under `examples/<algo>_trainer/`.
- [ ] Smoke test under `tests/special_e2e/run_<algo>_<model>.sh` wired
      into `tests/gpu_smoke/run_gpu_smoke_tests.sh`.
- [ ] Document the batch contract (`latents_clean`, `train_timesteps`,
      sample-level rewards or pairs/groups) for direct-preference algorithms.
- [ ] If the registry or adapter contract changed, update
      [`integrating_a_diffusion_model.md`](integrating_a_diffusion_model.md)
      to match.
