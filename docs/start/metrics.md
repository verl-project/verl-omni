(metrics)=
# Training Metrics

Last updated: 05/15/2026

The table below describes metrics specific to diffusion FlowGRPO / GRPO-Guard training, logged each step to your configured backend (console / W&B).

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| zero_std_ratio | $\frac{1}{B}\lvert\{i : \sigma_i = 0\}\rvert$ | GRPO derives its learning signal from relative rewards within a group; $\sigma_i = 0$ means group $i$ contributes no gradient regardless of absolute reward. A persistently high value (e.g. $> 0.5$) indicates reward saturation or poorly calibrated task difficulty. |
| std_mean | $\frac{1}{B}\sum\limits_{i=1}^{B} \sigma_i$ | Tracks average reward diversity across the batch. A declining trend is an early warning of saturation, typically visible before zero_std_ratio spikes. |
| pg_clipfrac_higher | $\hat{P}(r > 1 + \varepsilon)$ | The policy is reinforcing high-advantage denoising steps beyond the clip threshold. pg_clipfrac_higher $\gg$ pg_clipfrac_lower signals upward-dominant learning and can guide tuning of the clip ratio or learning rate. |
| pg_clipfrac_lower | $\hat{P}(r < 1 - \varepsilon)$ | The policy is suppressing low-advantage denoising steps beyond the clip threshold. Asymmetry between higher and lower clipfrac reveals the dominant learning direction. |
| ratio_mean | $\mathbb{E}[\rho_t]$ | Mean importance ratio across the batch. Should stay close to 1; persistent drift indicates the current policy is diverging from the rollout policy. |
| ratio_std | $\mathrm{Std}(\rho_t)$ | Spread of the importance ratio. High values signal high-variance gradient updates and may indicate the clip ratio or learning rate is too large. |
| timing_per_image_ms | Latency (ms/image) per stage | Covers rollout, reference log-prob, old log-prob, advantage computation, and actor update; identifies which stage dominates step time and where to focus optimization effort. |
| throughput | $\dfrac{B \times n}{t_\mathrm{step} \times N}$ (images / GPU / s) | Overall training throughput. Use alongside timing_per_image_ms to evaluate scaling efficiency and detect regressions across runs. |

**Variables.**

- $B$ — number of prompts per training batch
- $n$ — number of images generated per prompt
- $\sigma_i$ — reward standard deviation within group $i$
- $\rho_t$ — importance ratio $\pi_\theta / \pi_{\theta_\mathrm{old}}$ per (image, denoising-timestep) pair
- $r$ — shorthand for $\rho_t$ in clipping expressions
- $\varepsilon$ — clip ratio
- $N$ — number of GPUs
- $t_\mathrm{step}$ — wall-clock time per training step
