# Flow-DPPO

Flow-DPPO ([paper](https://arxiv.org/abs/2606.11025), [UniRL code](https://github.com/Tencent-Hunyuan/UniRL/blob/main/unirl/algorithms/flowdppo.py)) is a policy-gradient algorithm for flow matching models. It keeps FlowGRPO's stochastic reverse-SDE rollout, group-relative advantages, and per-step log-prob ratios, but replaces PPO ratio clipping with an exact Gaussian divergence mask.

At each trainable denoising step, the SDE transition is Gaussian. Flow-DPPO computes the KL between the old and current reverse-step means:

```text
KL(old || new) = mean((mu_new - mu_old)^2 / (2 * sigma_t^2))
```

The policy update is masked only when it is both outside the divergence trust region and moving farther from the old policy:

- positive advantage, `ratio > 1`, and `KL > kl_mask_threshold`
- negative advantage, `ratio < 1`, and `KL > kl_mask_threshold`

All other updates remain active, including corrective updates that move the current policy back toward the old policy.

## Configuration

Use the standard diffusion policy-gradient trainer:

```bash
algorithm.trainer_type=policy_gradient
algorithm.adv_estimator=flow_dppo
actor_rollout_ref.model.algorithm=flow_dppo
actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_dppo
```

Important knobs:

- `actor_rollout_ref.actor.diffusion_loss.kl_mask_threshold`: divergence threshold for the asymmetric mask. The default is `1e-5`.
- `actor_rollout_ref.actor.diffusion_loss.add_kl_coefficient`: when `True`, normalize the mean drift by the scheduler's SDE noise scale `std_dev_t * sqrt_dt`. This matches the Flow-SDE log-prob variance used during Qwen-Image training.
- `actor_rollout_ref.rollout.algo.sde_type`: Flow-DPPO can use the same sampler family as FlowGRPO. The Qwen-Image example uses `sde`.

## Example

Run the Qwen-Image OCR LoRA recipe:

```bash
bash examples/flowdppo_trainer/run_qwen_image_ocr_lora.sh
```

For a short local run, override paths and steps:

```bash
MODEL_PATH=/path/to/Qwen-Image \
REWARD_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
TOTAL_TRAINING_STEPS=2 \
bash examples/flowdppo_trainer/run_qwen_image_ocr_lora.sh
```
