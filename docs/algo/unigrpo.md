# UniGRPO

Last updated: 09/22/2026.

[UniGRPO](https://arxiv.org/abs/2603.23500) jointly optimizes autoregressive reasoning and image generation in a shared model. The BAGEL recipe generates a thinking chain with the understanding experts, conditions image generation on the prompt and that chain, and uses the image reward to train both paths. It builds on the existing [BAGEL FlowGRPO recipe](../examples/bagel/flowgrpo_trainer_bagel.md) and [GRPO-Guard](grpo_guard.md).

## Shared reward and joint objective

For each prompt $p$, sample $G$ thinking chains $c_i$ and one image $x_i$ per chain ($M=1$). PickScore evaluates the image against the original prompt, producing reward $R_i$. The recipe computes a group-normalized advantage:

$$
A_i = \frac{R_i - \operatorname{mean}_{j=1}^{G} R_j}{\operatorname{std}_{j=1}^{G} R_j + \epsilon}.
$$

The same scalar advantage supervises every thinking token and the sampled image SDE steps for that prompt/chain pair. The AR path uses the clipped GRPO surrogate with token likelihood ratio $r_{i,k}=\exp(\log\pi_\theta(c_{i,k}|p,c_{i,<k})-\log\pi_{\mathrm{old}}(c_{i,k}|p,c_{i,<k}))$:

$$
\mathcal L_{\mathrm{AR}} = -\mathbb E_{i,k}\left[\min\left(A_i r_{i,k}, A_i\operatorname{clip}(r_{i,k},1-\delta,1+\delta)\right)\right].
$$

The implementation averages token losses within each sample, then averages samples. The image path uses the GRPO-Guard surrogate, including its reverse-SDE mean-drift correction and timestep scaling, plus a velocity regularizer:

$$
\mathcal L_{\mathrm{image}} = \mathcal L_{\mathrm{Guard}} + \lambda\,\mathbb E_{i,t}\left[\|v_\theta(x_{i,t},t,p,c_i)-v_{\mathrm{ref}}(x_{i,t},t,p,c_i)\|^2_{\mathrm{mean}}\right].
$$

The velocity target comes from a frozen snapshot taken before the first image update. Velocity MSE replaces latent KL; there is no text KL term. Setting `ratio_norm=false` selects the plain FlowGRPO image surrogate instead of GRPO-Guard. See `ar_grpo_loss`, `UniGRPOJointUpdater`, and `UniGRPOLoss` for the exact reductions and numerical implementation.

Gradients from both losses accumulate in the same transformer before **one optimizer step**. Backward executes per sample to bound activation memory; “two backwards” describes the two loss branches, not a fixed count of autograd calls for a multi-sample batch. The understanding/base parameters use learning rate `1e-6`; parameters matching `moe_gen` use `3e-5` through configuration-driven optimizer groups.

## Sampling and replay

The implemented recipe uses torch-only native sampling. Each actor rank synchronizes a flat bf16 replica from its FSDP2 training weights, generates a variable-length thinking chain with KV caching, then samples an image. This avoids FSDP collectives inside variable-length decoding. The replica is parked on CPU before training replay to release accelerator memory.

Before returning the rollout, `record_old_logp` recomputes AR and image log probabilities on the training module. This anchors the update-zero ratio to one despite numerical differences between replica sampling and FSDP replay; the denominator is the replayed training-policy likelihood, not the unmodified replica likelihood. Later minibatch updates use these fixed denominators and become off-policy. The default recipe uses two joint updates per rollout batch.

The training defaults are 512×512 images, 25 denoising steps, a three-step SDE window in the early high-noise portion of the schedule, noise level 0.8, and no training CFG. The standard validation loop is skipped because native rollout has no separate validation server.

## Framework integration

The recipe selects the shared `PPODiffusersFSDPEngine` with `model_type=diffusion_model`. The `(OmniBagelForConditionalGeneration, unigrpo)` adapter supplies two extension hooks:

- `fsdp2_sharding_units(module)` selects transformer layers and the root leaves used by functional AR/image forwards. The engine applies its precision, offload and sharding policies. Other adapters return `None` and keep the default wrapping path.
- `build_engine_hooks(module, model_config, optimizer_config)` creates `BagelUniGRPOHooks`. It supplies joint backward and generation through the `DiffusionEngineHooks` contract. Other adapters return `None` and keep the existing training loop.

The hooks accumulate gradients without owning an optimizer. Zeroing gradients, clipping, the optimizer step, scheduling and checkpoint management remain in the shared engine. Explicit leaf sharding uses shard-aware norm clipping over the FSDP mesh, avoiding per-parameter DTensor reductions. Optimizer parameter groups preserve the configured optimizer implementation and options; the first matching name substring wins.

For `algorithm.trainer_type=unigrpo`, `rollout.name=native` selects `NativeRolloutWorker`, a thin subclass of the shared worker that dispatches `generate` requests. It initializes the parent with the actor role to reuse model setup, updates and checkpoints without creating a separate rollout engine. BAGEL-specific token and trajectory handling remains inside the model adapter and hook implementation.

## Configuration and usage

```bash
bash examples/unigrpo_trainer/bagel/run_bagel_unigrpo.sh
```

Relevant selectors and defaults:

```yaml
algorithm:
  trainer_type: unigrpo
  adv_estimator: flow_grpo
actor_rollout_ref:
  model:
    model_type: diffusion_model
    algorithm: unigrpo
  actor:
    strategy: fsdp2
    optim:
      _target_: verl_omni.workers.config.diffusion.FSDPDiffusionOptimizerConfig
      lr: 1.0e-6
      param_group_lrs: {moe_gen: 3.0e-5}
      override_optimizer_config: {foreach: false}
    diffusion_loss:
      loss_mode: unigrpo
      clip_ratio: 1.0e-6
      mse_weight: 1.5e-5
      ratio_norm: true
  rollout:
    name: native
    n: 8
```

See the [BAGEL recipe](../examples/bagel/unigrpo_trainer_bagel.md) for dataset preparation and resource requirements. Full fine-tuning with FSDP2 and CUDA is the validated path; this recipe does not establish FSDP1, NPU, LoRA or vLLM rollout support. It implements the $M=1$ shared-advantage setting, and does not claim to reproduce the paper's absolute benchmark scores. The regularizer snapshot is held by the hooks rather than checkpointed: restarting from a checkpoint initializes a new reference snapshot before its first image update.

## References

- [UniGRPO paper](https://arxiv.org/abs/2603.23500)
- [UniRL reference integration](https://github.com/Tencent-Hunyuan/UniRL/pull/148)
