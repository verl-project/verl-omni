(metrics)=
# Diffusion Training Metrics

The table below describes metrics specific to diffusion FlowGRPO training, logged each step to your configured backend (console / W&B).

**Variables.** $B$ = `train_batch_size`; $n$ = `rollout.n`, images generated per prompt; $\sigma_i$ = reward standard deviation within group $i$; $r$ = probability ratio $\pi_\theta / \pi_{\theta_\mathrm{old}}$ per (image, denoising-timestep) pair; $\varepsilon$ = `clip_ratio`; $N$ = number of GPUs; $t_\mathrm{step}$ = wall-clock time per training step.

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| critic/rewards/zero_std_ratio | $\frac{1}{B}\lvert\{i : \sigma_i = 0\}\rvert$ | GRPO derives its learning signal from relative rewards within a group; $\sigma_i = 0$ means group $i$ contributes no gradient regardless of absolute reward. A persistently high value (e.g. $> 0.5$) indicates reward saturation or poorly calibrated task difficulty. |
| critic/rewards/std_mean | $\frac{1}{B}\sum_{i=1}^{B} \sigma_i$ | Tracks average reward diversity across the batch. A declining trend is an early warning of saturation, typically visible before zero_std_ratio spikes. |
| actor/pg_clipfrac_higher | $\hat{P}(r > 1 + \varepsilon)$ | The policy is reinforcing high-advantage denoising steps beyond the clip threshold. pg_clipfrac_higher $\gg$ pg_clipfrac_lower signals upward-dominant learning and can guide tuning of clip_ratio or the learning rate. |
| actor/pg_clipfrac_lower | $\hat{P}(r < 1 - \varepsilon)$ | The policy is suppressing low-advantage denoising steps beyond the clip threshold. Asymmetry between higher and lower clipfrac reveals the dominant learning direction. |
| timing_per_image_ms | Latency (ms/image) per stage: rollout, reference log-prob, old log-prob, advantage computation, actor update | Identifies which stage dominates step time; use to focus optimization effort. |
| perf/throughput | $\dfrac{B \times n}{t_\mathrm{step} \times N}$ (images / GPU / s) | Overall training throughput. Use alongside timing_per_image_ms to evaluate scaling efficiency and detect regressions across runs. |
