# GRPO-Guard

Last updated: 05/08/2026.

GRPO-Guard ([paper](https://arxiv.org/abs/2510.22319)) is an extension of
[Flow-GRPO](flowgrpo.md) that stabilizes the importance-ratio estimate used in
the policy loss. The standard Flow-GRPO ratio
$\rho = \exp(\log p_\theta - \log p_{\text{old}})$ can become numerically
unbalanced when only a single Monte-Carlo noise sample $z$ is used per
denoising step, causing high-variance gradients and aggressive clipping.

GRPO-Guard adds a **ratio-mean bias** correction that explicitly penalises
drift in the reverse-SDE proposal mean of the current policy relative to the
rollout policy, and rescales the per-step loss by $1 / (\sqrt{-dt})^2$ so the
gradient magnitude is consistent across denoising steps.

## Algorithm

For step $t$ with proposal mean $\mu_\theta(x_t)$ from the current policy and
$\mu_{\text{old}}(x_t)$ from the rollout policy, SDE noise scale
$\sigma_t = \mathrm{std\\_dev\\_t}$, and $\sqrt{-dt}$:

$$
b_t = \frac{\lVert \mu_\theta - \mu_{\text{old}} \rVert_{\text{mean}}^2}
            {2 (\sqrt{-dt}\, \sigma_t)^2}
$$

$$
\rho_t = \exp\big((\log p_\theta - \log p_{\text{old}} + b_t) \cdot
                  (\sqrt{-dt}\, \sigma_t)\big)
$$

$$
\mathcal{L}^{\text{guard}}_t =
  \frac{1}{(\sqrt{-dt})^2}\;
  \mathbb{E}\big[\max(-A_t \rho_t,\ -A_t \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon))\big]
$$

The squared-norm in $b_t$ is averaged over the channel and spatial dimensions
of the latent (see `GRPOGuardLoss` in
[`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py)).

## Configuration

GRPO-Guard reuses the entire Flow-GRPO training stack — only the actor loss
mode changes. Refer to [Flow-GRPO](flowgrpo.md) for advantage estimator,
rollout, sampling, batch-size, and reward configuration.

To enable GRPO-Guard:

- `actor_rollout_ref.actor.diffusion_loss.loss_mode=grpo_guard`
- `actor_rollout_ref.rollout.algo.sde_type=sde`

A typical small clip ratio works well with the additional bias term:

- `actor_rollout_ref.actor.diffusion_loss.clip_ratio=2e-6`

KL regularisation against a frozen reference policy still works the same way
as Flow-GRPO (`actor_rollout_ref.actor.use_kl_loss=True`,
`actor_rollout_ref.actor.kl_loss_coef=...`).

## Example script

A 4-card collocated training script is provided:

```bash
bash examples/grpoguard_trainer/run_qwen_image_ocr_lora.sh
```

It reuses the Flow-GRPO Qwen-Image OCR setup and only flips the actor loss
mode, the clip ratio, and the experiment name. Dataset and model preparation
follow the same instructions as the [Flow-GRPO quick-start](../start/flowgrpo_quickstart.md).

## References

- [Flow-GRPO: Online policy gradient RL for flow matching models](https://arxiv.org/abs/2505.05470)
- [GRPO-Guard: ratio-bias regularisation for diffusion-model RL](https://arxiv.org/abs/2510.22319)
