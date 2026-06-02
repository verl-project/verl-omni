# How to Integrate a New Direct-Preference Algorithm for Diffusion Model

Last updated: 06/02/2026.

This guide explains how to add a direct-preference diffusion algorithm to
VeRL-Omni. Direct-preference algorithms train from final samples, rewards, or
chosen/rejected preferences using a forward-process objective. They do not
optimize reverse denoising trajectories with policy-gradient logprob ratios.

For PPO-like policy-gradient algorithms such as FlowGRPO, MixGRPO, and
GRPO-Guard, use
[`integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md`](integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md)
instead.

---

## Classify the Algorithm First

Two independent questions determine the implementation path.

### Policy-gradient vs direct-preference

**Policy-gradient algorithms** treat diffusion generation as a reverse-process
MDP. Rollout stores trajectory tensors such as `all_latents`, `all_timesteps`,
`old_log_probs`, optional reference logprobs, and per-timestep advantages. The
trainer computes a PPO-like objective over likelihood ratios.

**Direct-preference algorithms** train from final samples or preferences. The
actor batch contains clean latents or preference pairs plus objective-specific
forward-training tensors. For example, DPO uses paired `noise`, `timesteps`, and
`ref_noise_pred`, while DiffusionNFT uses `train_timesteps` and
algorithm-specific reward probabilities. The loss consumes prediction-space
tensors rather than reverse-step logprobs.

### Offline vs online

**Offline direct-preference** algorithms consume data that has already been
generated and scored. Offline DPO is the reference implementation: data preparation
writes win/lose pairs to parquet, training sets `algorithm.sample_source=offline`,
and rollout/reward workers are not started.

**Online direct-preference** algorithms generate samples during training and
score them with a reward function. DiffusionNFT is the reference
implementation: rollout produces final clean latents, reward scoring happens
live, and `DiffusionNFTLoss.prepare_actor_batch(...)` converts rollout outputs
into the forward-process actor batch.

| Algorithm family | Examples | Data source | Trainer | Engine contract |
|---|---|---|---|---|
| PPO-like policy gradient | FlowGRPO, MixGRPO, GRPO-Guard | Online rollout trajectories | `PolicyGradientRayTrainer` | `PPODiffusersFSDPEngine`, reverse logprob tensors |
| Offline direct preference | Offline DPO | Precomputed win/lose pairs | `DirectPreferenceRayTrainer` | `DPODiffusersFSDPEngine`, paired noisy-latent tensors |
| Online direct preference | DiffusionNFT | Live rollout + reward | `DirectPreferenceRayTrainer` | `NFTDiffusersFSDPEngine`, clean latents + forward timesteps |

---

## TL;DR

A new direct-preference algorithm usually needs **five pieces**:

1. **Trainer routing** via `algorithm.trainer_type=direct_preference`.
2. **A data-source contract** via `algorithm.sample_source=offline` or
   `algorithm.sample_source=online`.
3. **A loss** registered with `@register_diffusion_loss(...)`.
4. **An algorithm-specific FSDP engine** registered with
   `@EngineRegistry.register(model_type=...)` when the actor batch differs
   from PPO's reverse-trajectory contract.
5. **Model and rollout adapters** only when the algorithm changes the
   architecture-specific input/output contract.

The shared trainer is
[`DirectPreferenceRayTrainer`](../../verl_omni/trainer/diffusion/ray_diffusion_trainer.py).
It supports both offline and online rollout through config flags.

---

## Step 1 — Choose the Data Source

Set the trainer type for every direct-preference algorithm:

```bash
algorithm.trainer_type=direct_preference
```

Then choose the sample source.

For offline preference datasets:

```bash
algorithm.sample_source=offline
```

The trainer initializes actor workers only, skips rollout and reward workers,
reads `sample_level_scores` from the batch, and skips validation generation by
default. Use this path for DPO-style datasets where preference labels or scores
are prepared before training.

For online preference or reward-split training:

```bash
algorithm.sample_source=online
```

The trainer starts the normal rollout and reward stack, repeats prompts by
`actor_rollout_ref.rollout.n`, scores generated samples, and then delegates
algorithm-specific batch preparation to the active loss class.

---

## Step 2 — Define the Batch Contract

Document the actor batch keys before writing the engine or loss. The trainer
will pass a `TensorDict` to the worker; the engine and loss must agree on every
key and shape.

For paired offline algorithms such as Offline DPO, set:

```bash
algorithm.paired_preference=true
```

This tells `DirectPreferenceRayTrainer._update_actor(...)` to double the mini
batch size and disable shuffling when needed, so adjacent chosen/rejected
samples remain together. The reference DPO path uses:

- `OfflineDPODataset` to read one win/lose row per prompt.
- `offline_dpo_collate_fn` to expand rows into adjacent `[win, lose]` samples
  with a shared `uid`.
- `DPODiffusersFSDPEngine` to create shared `noise` and `timesteps` for each
  pair.
- `DPOLoss` to compare model and reference prediction errors pairwise.

For online algorithms such as DiffusionNFT, keep:

```bash
algorithm.paired_preference=false
```

The rollout batch should contain final clean samples rather than reverse
trajectories. The reference DiffusionNFT path uses:

- `latents_clean` from the rollout adapter.
- live `sample_level_scores` from the reward function.
- `train_timesteps` sampled for forward-process training.
- `reward_prob` computed from group-relative rewards in
  `DiffusionNFTLoss.prepare_actor_batch(...)`.

---

## Step 3 — Register the Loss

Add a registered loss class in
[`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py):

```python
@register_diffusion_loss("<your_algo>")
class MyDirectPreferenceLoss(DiffusionLossFn):
    """Forward-process direct-preference objective."""

    required_model_output_keys = ("<model_output>",)
    required_data_keys = ("<batch_key>",)

    @classmethod
    def compute_loss(cls, **kwargs):
        ...

    def __call__(self, *, config, model_output, data) -> DiffusionLossResult:
        self.validate_inputs(
            loss_name="<your_algo>",
            model_output=model_output,
            data=data,
        )
        ...
        return DiffusionLossResult(loss=loss, metrics=metrics)
```

Then add the loss name to
[`DiffusionLossConfig.__post_init__`](../../verl_omni/workers/config/diffusion/actor.py).

Override `DiffusionLossFn.prepare_actor_batch(...)` only when the trainer must
transform rollout outputs before actor update. Offline DPO does not need this because
the offline dataset and reference forward pass already supply the loss inputs.
DiffusionNFT does need it because online rewards must be converted into
forward-process tensors such as `reward_prob` and `train_timesteps`.

---

## Step 4 — Register the FSDP Engine

Direct-preference algorithms usually need their own engine because their actor
batch does not match PPO's reverse-trajectory contract. Register the engine in
[`verl_omni/workers/engine/fsdp/diffusers_impl.py`](../../verl_omni/workers/engine/fsdp/diffusers_impl.py):

```python
@EngineRegistry.register(
    model_type="<your_algo>_model",
    backend=["fsdp", "fsdp2"],
    device=["cuda", "npu"],
)
class MyDirectPreferenceDiffusersFSDPEngine(DiffusersFSDPEngine):
    """FSDP engine for <your_algo>."""

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        ...

    def prepare_model_inputs(self, micro_batch, step: int):
        ...

    def prepare_model_outputs(self, output, micro_batch):
        ...
```

Then set:

```bash
actor_rollout_ref.model.model_type=<your_algo>_model
```

DPO uses `model_type=diffusion_dpo_model` and
`DPODiffusersFSDPEngine`. DiffusionNFT uses
`model_type=diffusion_nft_model` and `NFTDiffusersFSDPEngine`.

---

## Step 5 — Add Model and Rollout Adapters

Add adapters only for the contexts your algorithm actually uses.

Offline algorithms generally need a training adapter but may not need a rollout
adapter. Offline DPO registers the SD3 training adapter under
[`verl_omni/pipelines/sd3_dpo/`](../../verl_omni/pipelines/sd3_dpo/__init__.py)
and consumes precomputed latents plus prompt embeddings from parquet.

Online algorithms need a rollout adapter when generated samples must carry
algorithm-specific fields. DiffusionNFT registers
[`verl_omni/pipelines/qwen_image_diffusion_nft/`](../../verl_omni/pipelines/qwen_image_diffusion_nft/__init__.py):

- The rollout adapter emits final clean latents for forward-process training.
- The training adapter implements the shared model hooks:
  `prepare_model_inputs` to build architecture-specific transformer kwargs and
  `forward` to run a single prediction-space model pass.

Register each package from
[`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py) so
the decorators run on import.

---

## Step 6 — Configure Reference and Old Policies

`DirectPreferenceRayTrainer` enables the reference policy for
direct-preference losses.

For algorithms that use one trainable policy state, normal LoRA or full-weight
configuration is enough. DPO follows this path.

For algorithms that need an old rollout policy in addition to the trainable
policy, declare policy-state adapters:

```bash
actor_rollout_ref.model.policy_state_adapters='["default","old"]'
actor_rollout_ref.rollout.rollout_adapter=old
```

DiffusionNFT uses this pattern. At startup, the trainer copies `default` into
`old`; after actor updates it refreshes the old adapter with copy or EMA based
on:

```bash
algorithm.old_policy_decay_schedule=<schedule>
algorithm.old_policy_decay=<optional_decay>
algorithm.old_policy_update_interval=<steps>
```

The shared `LoRAAdapterMixin` handles adapter selection, copy, and EMA updates.
Avoid adding algorithm-specific adapter plumbing unless the shared helpers are
insufficient.

---

## Step 7 — Wire a Launch Script

Create `examples/<algo>_trainer/` with a runnable script and README.

For offline paired DPO-style algorithms, include the dataset class and pair
flags:

```bash
algorithm.trainer_type=direct_preference \
algorithm.sample_source=offline \
algorithm.paired_preference=true \
actor_rollout_ref.model.algorithm=dpo \
actor_rollout_ref.model.model_type=diffusion_dpo_model \
actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo \
data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_dpo_dataset \
```

For online DiffusionNFT-style algorithms, include online rollout, old policy,
and loss-specific knobs:

```bash
algorithm.trainer_type=direct_preference \
algorithm.sample_source=online \
algorithm.paired_preference=false \
actor_rollout_ref.model.algorithm=diffusion_nft \
actor_rollout_ref.model.model_type=diffusion_nft_model \
actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
actor_rollout_ref.rollout.rollout_adapter=old \
actor_rollout_ref.rollout.calculate_log_probs=False \
```

Keep loss-specific worker knobs under
`actor_rollout_ref.actor.diffusion_loss`. Keep trainer-level data-flow knobs
under `algorithm`.

---

## Step 8 — Add Smoke Tests

Add an end-to-end smoke test under `tests/special_e2e/`:

- Use [`tests/special_e2e/run_sd35_offline_dpo.sh`](../../tests/special_e2e/run_sd35_offline_dpo.sh)
  as the reference for offline pair training.
- Use [`tests/special_e2e/run_diffusionnft_qwen_image.sh`](../../tests/special_e2e/run_diffusionnft_qwen_image.sh)
  as the reference for online direct-preference training.

Register the script in
[`tests/gpu_smoke/run_gpu_smoke_tests.sh`](../../tests/gpu_smoke/run_gpu_smoke_tests.sh).
The test should exercise trainer routing, sample-source routing, loss dispatch,
FSDP engine dispatch, and any algorithm-specific adapter contract.

---

## Final Checklist

- [ ] `algorithm.trainer_type=direct_preference` is set.
- [ ] `algorithm.sample_source` is set to `offline` or `online`.
- [ ] `algorithm.paired_preference=true` is used only for adjacent
      chosen/rejected pair batches.
- [ ] Loss class is registered with `@register_diffusion_loss("<name>")` and
      added to `DiffusionLossConfig.valid_modes`.
- [ ] Online algorithms that need rollout-to-actor transformation implement
      `DiffusionLossFn.prepare_actor_batch(...)`.
- [ ] FSDP engine is registered with `@EngineRegistry.register(model_type=...)`
      or an existing compatible direct-preference engine is reused.
- [ ] Launch script sets `actor_rollout_ref.model.model_type` to the matching
      engine key.
- [ ] Model and rollout adapters are registered only for the contexts the
      algorithm uses.
- [ ] Old-policy algorithms declare `policy_state_adapters` and
      `rollout_adapter=old`.
- [ ] Example README documents whether the algorithm is offline or online and
      lists the key config flags.
- [ ] Smoke test covers the selected data source, trainer, loss, engine, and
      adapter path.
