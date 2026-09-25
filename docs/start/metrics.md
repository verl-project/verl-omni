(metrics)=
# Training Metrics

Last updated: 09/24/2026

Metrics are logged each step to your configured backend (console / W&B). The diffusion RL
trainers share the group-statistic metrics, and each objective adds its own block below. Names
are shortened from their log keys: the actor-side rows are logged under `actor/`, the
reward-group statistics under `critic/rewards/`, and the timing and throughput rows under their
own prefixes.

## FlowGRPO / GRPO-Guard

The table below describes metrics specific to diffusion FlowGRPO / GRPO-Guard training.

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

## DiffusionNFT

DiffusionNFT consumes the same rollouts and group statistics, but replaces the PPO-style
ratio objective with a reward-weighted two-branch fit to the clean latent. One branch
(`positive_loss`) pulls the prediction toward the rewarded direction and the other
(`negative_loss`) pushes an implicit negative away from it, blended by the reward probability
$r_i$. The remaining metrics report the two gradient-scale terms that balance that update and
the reference anchor.

With $\hat{x}_0^{\pm}$ the two branch targets, the branch losses are
$L_i^{\pm} = \mathrm{mean}\left((\hat{x}_0^{\pm} - x_0)^2 / w^{\pm}\right)$, where
$w^{\pm} = \max\left(\overline{\lvert \hat{x}_0^{\pm} - x_0 \rvert}, \epsilon_w\right)$ is an
adaptive per-sample weight that keeps early-training gradients from being dominated by the
largest residuals.

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| total_loss | $\text{policy loss} + \lambda \cdot \text{ref KL loss}$ | The objective actually backpropagated. Read it next to `policy_loss` to see how much of the step is the reference anchor rather than the reward. |
| policy_loss | $A \cdot \overline{\dfrac{r_i L_i^{+} + (1 - r_i) L_i^{-}}{\beta}}$ | The reward-weighted fit that drives learning; should track the reward curves. A flat or rising series while reward improves means the learning rate or `ref_kl_coef` is holding the update back. |
| positive_loss | $\overline{L_i^{+}}$ | Loss of the rewarded branch. Falling values mean the policy reproduces the rewarded sample more closely. |
| negative_loss | $\overline{L_i^{-}}$ | Loss of the implicit negative branch. This branch, not `positive_loss`, is what makes DiffusionNFT preference-like rather than plain distillation; if it stops moving, the negative side of the objective has collapsed. |
| reward_prob_mean | $\overline{r_i}$ | Mean reward weight, with $r_i = \mathrm{clip}(A_i / A, -1, 1) / 2 + \tfrac{1}{2} \in [0, 1]$. Near $0.5$ is expected for group-normalized advantages. A persistent excursion toward $0$ or $1$ means the group statistics have collapsed to a single sign and the batch carries little contrast. |
| contraction_scale | $\overline{\lvert A \beta t (v_\theta - v_\mathrm{old}) \rvert}$ | The reward-free term pulling the policy back toward the rollout policy. It grows with the policy's drift and is a stabilizer, not learning signal. |
| reward_term_scale | $\overline{\lvert 2 A (r_i - \tfrac{1}{2})(\hat{x}_0^{\text{old}} - x_0) \rvert}$ | Magnitude of the reward-driven term, i.e. the whole learning signal. $A$ cancels out of it, so reading this next to `contraction_scale` shows which of the two dominates. |
| log10_signal_ratio | $\log_{10}(\text{reward term}) - \log_{10}(\text{contraction term})$ | How far the reward term dominates the contraction; roughly $\ge 0$ means the reward, not the pull-back, is steering the update. Logged in log space because `contraction_scale` is exactly $0$ on a micro-batch whose $v_\theta - v_\mathrm{old}$ rounds to zero (common right after a LoRA re-init), and a raw ratio would then poison the batch mean. Recover the ratio as $10^{\text{log10 signal ratio}}$. |
| old_deviate | $\overline{(v_\theta - v_\mathrm{old})^2}$ | Squared drift from the rollout policy, the quantity `contraction_scale` penalizes. A rising trend warns the update has moved too far off-policy. |
| ref_kl_loss | $\overline{(v_\theta - v_\mathrm{ref})^2}$ | Deviation from the reference policy. Named `kl`, but implemented as a mean squared difference. |
| ref_kl_contribution | $\lambda \cdot \text{ref KL loss}$ | The reference anchor's share of `total_loss`. It should shape the update without swamping the reward signal; compare it against `policy_loss`. |

**Variables.**

- $v_\theta$ — velocity predicted by the policy being trained
- $v_\mathrm{old}$ — velocity predicted by the rollout (old) policy
- $v_\mathrm{ref}$ — velocity predicted by the reference policy
- $\hat{x}_0^{\pm}$ — clean-latent target of the positive / negative branch
- $L_i^{\pm}$, $w^{\pm}$ — adaptive-weighted branch loss and its weight, defined above
- $\epsilon_w$ — the adaptive weight floor (`adaptive_weight_min`)
- $r_i$ — reward probability of sample $i$, mapped from the clipped advantage
- $\beta$ — `mix_beta`
- $A$ — `adv_clip_max`
- $\lambda$ — `ref_kl_coef`
- $t$ — flow time
