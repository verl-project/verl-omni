# Centered Reward Distillation

Last updated: 10/24/2026.

CRD ([paper](https://arxiv.org/abs/2603.14128),
[code](https://github.com/yuanzhi-zhu/CRD)) is an online RL method for diffusion
models. It matches centered external rewards to implicit rewards estimated
from denoising errors. It uses separate sampling and moving-reference policies
and reward-adaptive regularization toward the CFG-guided pretrained model.

## Algorithm

For each prompt $c$, the sampling policy generates a group of $K$ clean latents
$x_0^i$ with raw rewards $r_i$. Forward-process training samples noise
$\epsilon_i$ and a timestep $t$:

$$
x_t^i = (1-t)x_0^i + t\epsilon_i,
\qquad v_i = \epsilon_i - x_0^i.
$$

The current policy $\theta$ and moving reference $\bar\theta$ predict velocities
at the same $(x_t^i,c,t)$. The released adaptive estimator uses

$$
d_\phi^i =
\frac{\operatorname{mean}[(v_\phi(x_t^i,c,t)-v_i)^2]}
{\max\!\left(\operatorname{sg}\!\left[
\operatorname{mean}|v_\phi(x_t^i,c,t)-v_i|\right], a_{\min}\right)},
\qquad u_i = d_{\bar\theta}^i-d_\theta^i,
$$

where the means reduce latent elements and $\operatorname{sg}$ denotes
stop-gradient. The target velocity, moving reference and adaptive denominators
are detached. Group-normalized advantages $A_i$ are clipped and mapped to
bounded rewards, with $C=\texttt{adv_clip_max}$:

$$
q_i = \frac12 + \frac{\operatorname{clip}(A_i,-C,C)}{2C}.
$$

This integration uses uniform within-prompt centering:

$$
\widetilde u_i = u_i - \operatorname{sg}\!\left[\frac1K\sum_{j\in G}u_j\right],
\qquad
\widetilde q_i = q_i - \frac1K\sum_{j\in G}q_j,
$$

$$
\mathcal L_{\mathrm{CRD}} =
\frac{C}{\beta}\operatorname{mean}_i
\left[(\beta\widetilde u_i-\widetilde q_i)^2\right].
$$

The pretrained anchor predicts with classifier-free guidance (CFG), independently
of the guidance used for sampling:

$$
v_{\mathrm{init}}^\gamma = v_{\mathrm{init}}(x_t,\varnothing,t)
+\gamma\left[v_{\mathrm{init}}(x_t,c,t)-v_{\mathrm{init}}(x_t,\varnothing,t)\right].
$$

The full objective adds the reference implementation's reward-adaptive
velocity-space KL surrogate:

$$
\lambda_i = \lambda_{\min}+r_i(\lambda_{\max}-\lambda_{\min}),
\qquad
\mathcal L = \mathcal L_{\mathrm{CRD}}
+\operatorname{mean}_i\left[\lambda_i
\operatorname{mean}\left[(v_\theta-v_{\mathrm{init}}^\gamma)^2\right]\right].
$$

Raw rewards must be finite and in $[0,1]$. Higher rewards strengthen the anchor
penalty. Only the current policy receives gradients.

## How VeRL-Omni Implements CRD

CRD reuses the online direct-preference trainer and `NFTDiffusersFSDPEngine`,
selected by `model_type=diffusion_nft_model`.

| Layer | What it does | Code |
| --- | --- | --- |
| Rollout adapter | Samples deterministic ODE endpoints and retains clean latents and positive/negative prompt embeddings. | `verl_omni/pipelines/sd3_crd/vllm_omni_rollout_adapter.py` |
| Batch preparation | Keeps prompt groups contiguous, computes advantages, selects shared timesteps and draws independent noise per sample. | `CRDLoss.prepare_actor_batch` |
| Training adapter | Constructs velocity targets and applies CFG only to the pretrained anchor. | `verl_omni/pipelines/sd3_crd/diffusers_training_adapter.py` |
| Actor loss | Centers rewards within each prompt group and adds reward-adaptive anchor regularization. | `verl_omni/trainer/diffusion/diffusion_algos.py` |
| Trainer | Initializes policy snapshots before checkpoint loading and refreshes both EMA states after actor updates. | `verl_omni/trainer/diffusion/v1/trainer_base.py` |

The four policy roles share one pretrained transformer:

| Role | State | Update |
| --- | --- | --- |
| Current policy | `default` LoRA adapter | Optimizer |
| Sampling policy | `old` LoRA adapter | `algorithm.old_policy_*` EMA |
| Moving loss reference | `moving_reference` LoRA adapter | `algorithm.reference_policy_*` EMA |
| Pretrained anchor | LoRA disabled | Frozen |

The engine selects `moving_reference` for the detached loss-reference prediction;
`old` retains the existing rollout/export meaning. The current policy and moving
reference predict conditionally. The pretrained prediction runs with adapters
disabled and no gradients. Rollout retains negative prompt embeddings even when
sampling uses CFG 1, so the anchor can use CFG 4.5.
Validation also uses the `old` sampling adapter, with the evaluation generation settings.

Each complete prompt group must fit in one micro batch on one rank. The trainer
disables actor shuffling and rejects incomplete groups, dynamic batches and
timestep staging. The initial scope is online V1 synchronous training with
FSDP/FSDP2 LoRA. Checkpoints use the existing named-adapter and optimizer state;
snapshot initialization precedes restoration so resumed EMA weights are kept.

## Configuration

```bash
actor_rollout_ref.model.algorithm=crd
actor_rollout_ref.model.model_type=diffusion_nft_model
algorithm.trainer_type=direct_preference
algorithm.sample_source=online
actor_rollout_ref.model.policy_state_adapters='["default","old","moving_reference"]'
actor_rollout_ref.rollout.rollout_adapter=old
actor_rollout_ref.rollout.calculate_log_probs=False
actor_rollout_ref.rollout.algo.noise_level=0.0
trainer.use_v1=True
trainer.v1.trainer_mode=sync
```

### Core Parameters

- `actor_rollout_ref.actor.diffusion_loss.crd_beta`: implicit reward scale
  $\beta$. Default `1.0`.
- `actor_rollout_ref.actor.diffusion_loss.adv_clip_max`: advantage bound $C$.
- `actor_rollout_ref.actor.diffusion_loss.adaptive_weight_min`: minimum adaptive
  denominator $a_{\min}$.
- `actor_rollout_ref.actor.diffusion_loss.ref_kl_coef`: maximum anchor coefficient
  $\lambda_{\max}$; the recipe uses `0.1`.
- `actor_rollout_ref.actor.diffusion_loss.crd_kl_min`: minimum anchor coefficient
  $\lambda_{\min}$. Default `0.0`; the recipe uses `0.0001`.
- `actor_rollout_ref.model.pipeline.reference_guidance_scale`: pretrained-anchor
  CFG $\gamma$. Default `4.5`.
- `algorithm.old_policy_decay_schedule` / `old_policy_decay` /
  `old_policy_update_interval`: sampling-policy EMA settings.
- `algorithm.reference_policy_decay_schedule` / `reference_policy_decay` /
  `reference_policy_update_interval`: independent moving-reference EMA settings.
- `algorithm.train_timestep_range` / `train_timestep_count`: rollout schedule
  range and number of timesteps selected for forward-process training.

Fixed decay values override schedules. Update intervals count trainer iterations.
Keep raw rewards in $[0,1]$ and retain whole groups in each micro batch.

## Reference Example

The [SD3.5-Medium OCR recipe](../examples/crd_trainer.md) uses LoRA and
Qwen2.5-VL-3B as the reward model. Its README contains setup and run commands.
