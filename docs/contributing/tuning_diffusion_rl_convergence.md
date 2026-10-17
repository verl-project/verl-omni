# Debugging a Diffusion RL Run That Does Not Converge

Last updated: 10/15/2026

Use this guide when a run trains without errors but the reward stays flat or gets worse.
If the run disagrees numerically with diffusers or with the rollout engine, start from
[Common Pitfalls](common_pitfalls.md) instead: those bugs look like convergence problems
but are fixed in code, not by tuning.

Work the sections in order. Each one rules out a cheaper explanation before the next one
spends GPU hours.

## 1. Gate the task before tuning anything

A run cannot improve a reward that is already at its ceiling or that does not vary within a
group. Check both from the step-0 validation and the first training step.

- **Headroom.** Compare the step-0 validation mean with the reward's maximum. Boogu-Image
  renders text so well that four OCR datasets started at 0.91–0.99 and never moved; a denser
  layout that started at 0.60 was the first usable one. If the baseline is near the ceiling,
  change the data, not the hyperparameters.
- **Within-group signal.** `critic/rewards/zero_std_ratio` should be near 0 and
  `critic/rewards/std_mean` should be large relative to the gains you hope to see. Group-relative
  advantages are zero when every sample in a group gets the same reward.
- **Deterministic reward.** Generative reward models sample by default, which adds noise to
  every comparison. Search the reward module for environment overrides of temperature and
  seed (for example `GENRM_OCR_TEMPERATURE`) and pin them for the whole investigation.

## 2. Make validation able to detect a change

Unpaired means across validation runs hide effects of a few hundredths. Fix the prompts and
the seed, then compare the same prompt across steps.

- Fix the validation seed with `+actor_rollout_ref.rollout.val_kwargs.seed=42`. The `+` is
  required because the key is not in the struct config.
- Compare each prompt with itself: report the mean paired difference, a bootstrap 95% CI and
  the win/loss rate. The validation JSONL rows carry `input`, `score` and one
  `reward/<function>/<component>` key per reward component.

  ```python
  import json, random, statistics as st

  def load(path):
      return {row["input"]: row for row in map(json.loads, open(path))}

  base, new = load("validation/0.jsonl"), load("validation/20.jsonl")
  diff = [new[k]["score"] - base[k]["score"] for k in base if k in new]
  boot = sorted(st.mean(random.choices(diff, k=len(diff))) for _ in range(4000))
  print(st.mean(diff), boot[100], boot[3899], sum(d > 0 for d in diff) / len(diff))
  ```

- Every batch draws different prompts, so per-step training reward is noisy. Compare
  10-step window averages, not individual steps.
- Validation cost scales with prompts × denoising steps, and doubles under text CFG. Size it
  so a pass takes minutes: on the HPSv2.1 + PickScore + CLIPScore mix (scores around 1.6–1.8),
  256 prompts gave a paired 95% CI of about ±0.02; unpaired comparisons are several times
  wider. 40 denoising steps are enough for a converged sampler.

## 3. Classify the failure from the training metrics

[Metrics](../start/metrics.md) defines each key. Read them as trajectories, not single values.

| Pattern | Reading | Next step |
| ------- | ------- | --------- |
| Step-N validation images identical to step 0, or the LoRA sync log reports dropped modules | Weights do not reach the rollout | Fix the sync before tuning. Search the log for `LoRA sync: bound`; it must report `0 dropped`. |
| `actor/ref_kl_loss` plateaus, `actor/grad_norm` decays, training and validation reward flat | The reference anchor holds the policy in place | Lower `ref_kl_coef` (see section 6). |
| Distance from the reference grows, validation reward falls with a CI below 0 | The policy moves in a harmful direction | Look at the images (section 5); suspect a structural mismatch before the learning rate. |
| Training reward rises, validation flat or falling | Rollout/validation mismatch or reward hacking | Look at the images; compare validation at the rollout step count. |

To confirm that the anchor dominates, run two steps with `ref_kl_coef=0` and the same seeds.
If `actor/grad_norm` drops by an order of magnitude, the anchor term was setting the update
direction. Adam normalizes step size, so a small `ref_kl_contribution` in the loss value does
not mean a small share of the gradient.

## 4. Run one-variable ablations with short gates

- Change one setting against a reference run that shares the data seed and validation seed.
  When a bug fix and a hyperparameter change land together, their effects cannot be separated
  later: PR #666 fixed LoRA sync and rebalanced the objective at once, and the rebalanced
  recipe was later treated as necessary for stability without that being shown.
- A paired validation at step 10 is usually enough to read the direction. Stop arms whose CI
  is entirely below 0; extend only arms that are improving.
- Compute step time from `timing_s/step` and `timing_s/testing` before choosing the step
  budget, and prefer more short arms over one long run.

## 5. Look at the images whenever the reward moves

Put step-0 and step-N validation images for the same prompts side by side. The image usually
names the failure faster than any metric:

- washed-out, low-contrast, blurry outputs: the guidance effect is eroding (section 6);
- saturated colors, repeated textures, text-like noise: reward hacking;
- unchanged images: weights are not reaching the rollout (section 3).

## 6. DiffusionNFT-specific lessons

- **Train CFG-free.** DiffusionNFT regresses the trained velocity onto targets built from the
  policy's own samples. Under text CFG the trained quantity is `v_u + s * (v_c - v_u)`, an
  extrapolation that is not the flow of any data distribution, so training it erodes the
  guidance. On Boogu-Image with the same settings, `guidance_scale=4.0` fell from 1.811 to
  1.745 in 10 steps and the images washed out, while `guidance_scale=1.0` rose from 1.593 to
  1.83–1.84 over 60 steps on both the V0 and V1 trainers. This matches the DiffusionNFT paper and the Qwen-Image recipe
  (`true_cfg_scale=1.0`). The evidence is one model and one 10-step comparison; treat it as a
  strong default, not a proof.
- **Judge CFG-free gains against the base model at its best guidance.** Part of a CFG-free
  gain is the policy learning to work without guidance. The claim that matters is the paired
  comparison against the base model at its usual guidance scale on the same prompts and seeds.
- **Start from a weak anchor.** `ref_kl_coef=10` with `lr=1e-4` held a Boogu-Image run at its
  baseline for 80 steps, and two different rewards both stayed flat. Start from the Qwen-Image
  values (search `examples/diffusionnft_trainer/qwen_image/` for `ref_kl_coef`) and add anchor
  strength only after a collapse has been reproduced with the LoRA sync verified.
