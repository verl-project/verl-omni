# Qwen-Image DiffusionNFT Nightly

DiffusionNFT on local tiny-random Qwen-Image weights. Shared run steps,
baselines, thresholds, and failure triage are in `tests/nightly/README.md`.

This is the v1-trainer form of `tests/special_e2e/run_diffusionnft_qwen_image.sh`:
same algorithm flags and old-adapter rollout, with the OCR reward model enabled,
stretched to 20 deterministic steps. `timestep_fraction=1.0` keeps every
training timestep, and the actor still permutes their order per row.
`calculate_log_probs` is off, so `timing_s/old_log_prob` and
`perf/mfu/actor_infer` are absent.

Rollout rows arrive in per-run `uuid4` uid order, and DiffusionNFT draws the
`train_timesteps` permutation in that row order. The nightly debug hook
stable-sorts the actor batch by `extra_info.repeat_index` before
`prepare_actor_batch`, so the same prompt keeps the same timestep order on the
next run. Rows of one prompt keep their incoming order. Dump comparison still
realigns both payloads by `repeat_index` before comparing `latents_clean` and
`train_timesteps`. A baseline dumped before that sort assigns timesteps by uid
order and will not match; replace `baseline/qwen_image_diffusion_nft/` after
the change.

Scores come from `genrm_ocr.compute_score_ocr` on the tiny Qwen-VL reward
model, with greedy decoding. The precision signal is the rollout and the
old-adapter actor update: clean latents, train timesteps, and LoRA gradients.
`batch.reward_prob` and `batch.advantages` are informational.

```bash
bash tests/nightly/qwen_image_diffusion_nft/run_qwen_image_diffusion_nft.sh
```
