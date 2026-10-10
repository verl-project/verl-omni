# DGPO

Last updated: 10/10/2026.

DGPO ([paper](https://arxiv.org/abs/2510.08425)) is an online RL method for
diffusion models that learns from **group-level preferences**. It samples a
group of images per prompt with a deterministic ODE sampler, scores them with a
reward, and trains with a forward-process (flow-matching) objective on the final
samples, so no reverse-process likelihoods or stochastic rollouts are needed.

## Algorithm

For a prompt $c$, the rollout policy samples a group $G$ of clean latents
$x_0^{1:|G|}$. Rewards are centered within the group into advantages $A_i$ and,
with `algorithm.norm_adv_by_std_in_grpo=True`, divided by the group reward standard
deviation (the batch one with `algorithm.global_std=True`).
Every sample of the group is re-noised with the **same** noise $\epsilon$ and the
**same** timestep $t$, $x_t = (1-t)x_0 + t\epsilon$, and scored by its
flow-matching error under the policy and a reference model:

$$
d_i = \mathrm{mean}\left(\big((\epsilon - x_0^i) - v_\theta(x_t^i, c, t)\big)^2\right),
\qquad
d_i^{\mathrm{ref}} = \mathrm{mean}\left(\big((\epsilon - x_0^i) - v_{\mathrm{ref}}(x_t^i, c, t)\big)^2\right),
$$

where the mean runs over latent elements, as in the reference implementation, so
$\beta$ is set on that per-element scale.

The group preference score and the loss are

$$
s_G = \frac{\beta}{|G|}\sum_{i \in G} A_i\,(d_i - d_i^{\mathrm{ref}}),
\qquad
\mathcal{L}(\theta) = \mathbb{E}_i\left[\sigma(s_G)\,A_i\,d_i\right],
$$

with $s_G$ treated as a constant. Samples whose old-policy ratio
$\exp(d_i^{\mathrm{old}} - d_i)$ leaves $[1-\epsilon_c, 1+\epsilon_c]$ in the
direction of $A_i$ keep a detached $d_i$, and an optional prediction-space MSE to
the reference model regularizes the update.

## How VeRL-Omni Implements DGPO

DGPO reuses the online direct-preference stack of DiffusionNFT.

| Layer | What it does | Code |
|---|---|---|
| Rollout adapter | Samples with the `old` LoRA adapter at noise level 0 and returns clean latents and the timestep schedule. | `verl_omni/pipelines/sd3_dgpo/` |
| Batch preparation | Makes groups contiguous, computes group advantages, draws shared timesteps and group-shared noise. | `DGPOLoss.prepare_actor_batch` |
| Actor loss | Group-weighted flow-matching loss with old-policy clipping and reference MSE. | `verl_omni/trainer/diffusion/diffusion_algos.py` |
| FSDP engine | `diffusion_nft_model`: old, current and reference velocities at the shared $(x_t, t)$. | `verl_omni/workers/engine/fsdp/diffusers_impl.py` |

The group score is computed inside one micro batch, so every rollout group must
sit whole in a micro batch on one rank. Each rank receives a contiguous slice of
`ppo_mini_batch_size * n / dp` rows and cuts it into contiguous micro batches, so
`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` must be a multiple of
`actor_rollout_ref.rollout.n` and `ppo_mini_batch_size` must be divisible by the
data-parallel size, as must the number of prompts per actor update. The v1
trainer also needs `trainer.v1.sampler.drop_incomplete_groups=True` so failed
rollouts do not leave partial groups. The trainer checks these, the LoRA `old`
adapter settings, `use_kl_loss=False` and `enable_timestep_staging=False` at start,
disables actor shuffling for DGPO, and the loss raises an error if a micro batch
still holds a partial group.

Training timesteps and noise are drawn once per training step and reused across
`ppo_epochs`.

### Differences from the reference implementation

- The rollout uses the `old` adapter from the first step; the reference samples with
  the current policy for its first `switch_ema_ref` steps.
- The `old` adapter EMA is refreshed once per training step rather than once per
  optimizer step.
- The example rolls out with the SD3 flow-matching Euler scheduler; the reference
  uses a DPM-Solver multistep scheduler.
- The reference all-reduces group scores across ranks; here every group must sit
  whole in one micro batch on one rank (see above).
- The reference accumulates gradients over all groups of a rollout epoch into one
  optimizer step; here each `ppo_mini_batch_size` is one optimizer step.
- The reference keeps a separate EMA of the trained weights for evaluation and
  checkpoints; there is no such EMA here.
- The example's group size, resolution, LoRA rank and validation steps follow the
  SD3.5 FlowGRPO OCR recipe rather than the reference run.

## Configuration

```bash
actor_rollout_ref.model.algorithm=dgpo
actor_rollout_ref.model.model_type=diffusion_nft_model
algorithm.trainer_type=direct_preference
algorithm.sample_source=online
actor_rollout_ref.model.policy_state_adapters='["default","old"]'
actor_rollout_ref.rollout.rollout_adapter=old
actor_rollout_ref.rollout.calculate_log_probs=False
actor_rollout_ref.rollout.algo.noise_level=0.0
```

### Core Parameters

- `actor_rollout_ref.actor.diffusion_loss.dgpo_beta`: $\beta$ in the group
  score. Default `100.0`.
- `actor_rollout_ref.actor.diffusion_loss.dgpo_clip_range`: $\epsilon_c$ for the
  old-policy ratio; `0` disables clipping. Default `0.01`.
- `actor_rollout_ref.actor.diffusion_loss.ref_kl_coef`: coefficient of the
  prediction-space reference MSE.
- `actor_rollout_ref.actor.diffusion_loss.adv_clip_max`: advantage clamp.
- `algorithm.train_timestep_range`: rollout schedule indices `[start, end)` that
  training timesteps are drawn from; `null` uses the whole schedule.
- `algorithm.train_timestep_count`: number of schedule indices drawn per step and
  shared by the whole batch; `null` applies `algorithm.timestep_fraction`.
- `algorithm.old_policy_decay_schedule`: `linear_to_0_3` ramps the `old` adapter
  EMA decay to 0.3 as in the reference implementation.

## Reference Example

[`examples/dgpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/dgpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh)
post-trains `stabilityai/stable-diffusion-3.5-medium` with LoRA on the OCR task
using `Qwen/Qwen2.5-VL-3B-Instruct` as the reward model. Its data, reward and LoRA
settings match the SD3.5 FlowGRPO OCR recipe; it rolls out with guidance scale 4.5
as in the reference implementation and uses a learning rate of `3e-4`. See the
[example README](https://github.com/verl-project/verl-omni/blob/main/examples/dgpo_trainer/README.md)
for data preparation.
