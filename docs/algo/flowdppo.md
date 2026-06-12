# Flow-DPPO

Last updated: 06/12/2026.

Flow-DPPO ([paper](https://arxiv.org/abs/2606.11025)) is an extension of
[Flow-GRPO](flowgrpo.md) that replaces PPO-style ratio clipping with an
asymmetric divergence mask. It keeps Flow-GRPO's stochastic reverse-SDE rollout,
group-relative advantages, and per-step log-prob ratios.

## Algorithm

For step $t$ with proposal mean $\mu_\theta(x_t)$ from the current policy and
$\mu_{\text{old}}(x_t)$ from the rollout policy, SDE noise scale
$\sigma_t = \mathrm{std\\_dev\\_t}$, and $\sqrt{-dt}$:

$$
D_t =
\frac{\lVert \mu_\theta - \mu_{\text{old}} \rVert_{\text{mean}}^2}
     {2 (\sqrt{-dt}\, \sigma_t)^2}
$$

The update is masked only when it is outside the divergence threshold and moving
farther from the rollout policy:

- positive advantage, $\rho_t > 1$, and $D_t > \epsilon_D$
- negative advantage, $\rho_t < 1$, and $D_t > \epsilon_D$

Corrective updates stay active. See `FlowDPPOLoss` in
[`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py).

## Configuration

Flow-DPPO reuses the entire Flow-GRPO training stack — only the actor loss mode
and divergence threshold change. Refer to [Flow-GRPO](flowgrpo.md) for
advantage estimator, rollout, sampling, batch-size, and reward configuration.

To enable Flow-DPPO:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_dppo`
- `actor_rollout_ref.actor.diffusion_loss.kl_mask_threshold=1e-5`
- `actor_rollout_ref.rollout.algo.sde_type=sde`

`actor_rollout_ref.actor.diffusion_loss.add_kl_coefficient=True` normalizes the
mean drift by the scheduler's SDE noise scale `std_dev_t * sqrt_dt`, matching
the Flow-SDE log-prob variance used during Qwen-Image training.

## Example script

A 4-card collocated training script is provided:

```bash
bash examples/flowdppo_trainer/run_qwen_image_ocr_lora.sh
```

It reuses the Flow-GRPO Qwen-Image OCR setup and only flips the actor loss mode,
the divergence threshold, and the experiment name. Dataset and model preparation
follow the same instructions as the [Flow-GRPO quick-start](../start/flowgrpo_quickstart.md).

## References

- [Flow-GRPO: Online policy gradient RL for flow matching models](https://arxiv.org/abs/2505.05470)
- [Flow-DPPO: Divergence proximal policy optimization for diffusion models](https://arxiv.org/abs/2606.11025)
- [UniRL Flow-DPPO implementation](https://github.com/Tencent-Hunyuan/UniRL/blob/main/unirl/algorithms/flowdppo.py)
- [GRPO-Guard: ratio-bias regularisation for diffusion-model RL](https://arxiv.org/abs/2510.22319)
