(metrics)=
# Diffusion Training Metrics

The table below describes the metrics that are specific to diffusion FlowGRPO training and are logged to your configured backend (console / W&B) during each training step.

| Metric | Definition | Implication |
|--------|------------|-------------|
| `critic/rewards/zero_std_ratio` | Fraction of prompt groups (out of `train_batch_size`) where all $n$ generated images received identical rewards, i.e. within-group $\sigma = 0$ | GRPO's learning signal comes from *relative* rewards within a group; a group with $\sigma = 0$ contributes no gradient regardless of the absolute reward value. A persistently high ratio (e.g. $> 0.5$) indicates the reward model is saturated or the task difficulty is poorly calibrated — the policy is not receiving useful training signal. |
| `critic/rewards/std_mean` | Mean of per-group reward standard deviations across the batch: $\frac{1}{B}\sum_{i=1}^{B}\sigma_i$ | Complements `zero_std_ratio` — tracks the average reward spread across the whole batch before full collapse. A declining `std_mean` is an early warning of saturation *before* `zero_std_ratio` spikes. |
| `actor/pg_clipfrac_higher` | Fraction of $(image, timestep)$ pairs where $\pi_\theta / \pi_{\theta_\mathrm{old}} > 1 + \varepsilon_\mathrm{clip}$ | The policy is trying to increase the probability of high-advantage denoising steps beyond what the clip threshold permits. `pg_clipfrac_higher` $\gg$ `pg_clipfrac_lower` signals an upward-dominant learning direction and can guide tuning of `clip_ratio` or the learning rate. |
| `actor/pg_clipfrac_lower` | Fraction of $(image, timestep)$ pairs where $\pi_\theta / \pi_{\theta_\mathrm{old}} < 1 - \varepsilon_\mathrm{clip}$ | The policy is suppressing low-advantage denoising steps more aggressively than the clip allows. Together with `pg_clipfrac_higher`, the asymmetry between the two values reveals the dominant learning direction. |
| `timing_per_image_ms/{stage}` | Per-image wall-clock latency (ms) for each pipeline stage: `gen` (rollout), `ref` (reference log-prob), `old_log_prob`, `adv` (advantage computation), `update_actor` | Identifies which stage dominates step time and where to focus optimisation effort. |
| `perf/throughput` | Images processed per GPU per second: $(B \times n) \;/\; (t_\mathrm{step} \times N_\mathrm{GPU})$, where $B$ is `train_batch_size`, $n$ is `rollout.n` | Overall training throughput. Use alongside `timing_per_image_ms` to evaluate scaling efficiency and spot regressions between runs. |
