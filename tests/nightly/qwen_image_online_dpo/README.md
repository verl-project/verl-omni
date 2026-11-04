# Qwen-Image Online DPO Nightly

Online DPO on local tiny-random Qwen-Image weights. Shared run steps,
baselines, thresholds, and failure triage are in `tests/nightly/README.md`.

This is the v1-trainer form of `tests/special_e2e/run_online_dpo_qwen_image.sh`:
paired preference, `loss_mode=dpo`, and the Qwen-Image DPO pipeline, stretched
to 20 deterministic steps. The nightly pins FSDP2, native/SDPA attention,
fixed seeds, and greedy OCR sampling. The L2 smoke leaves the strategy at the
FSDP default and probes FlashAttention.

`prepare_actor_batch` stable-sorts rows by `extra_info.repeat_index`, then
keeps the highest-reward and lowest-reward rollout of each prompt uid. Flow
noise and timesteps are drawn in that order, so the same prompt gets the same
draw on the next run. The nightly uses `rollout.n=4`. Pairing drops rows, so
`prepare_actor_batch_selected_indices` is not a full permutation and dump
comparison skips the pre-pair restore. Rows are aligned by `repeat_index`
only. A baseline dumped before that sort assigns noise by uid order and will
not match; replace `baseline/qwen_image_online_dpo/` after the change.

Direct-preference steps do not recompute SDE `old_log_probs`. Performance
comparison includes `timing_s/prepare_actor_batch` and `timing_s/ref` and does
not include `timing_s/old_log_prob` or `perf/mfu/actor_infer`. Reward-derived
tensors stay informational because OCR scores are not bit-exact.

```bash
bash tests/nightly/qwen_image_online_dpo/run_qwen_image_online_dpo.sh
```
