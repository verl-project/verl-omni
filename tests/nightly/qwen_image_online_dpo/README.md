# Qwen-Image Online DPO Nightly

Online DPO on local tiny-random Qwen-Image weights. Shared run steps,
baselines, thresholds, and failure triage are in `tests/nightly/README.md`.

This is the v1-trainer form of `tests/special_e2e/run_online_dpo_qwen_image.sh`:
paired preference, `loss_mode=dpo`, and the Qwen-Image DPO pipeline, stretched
to 20 deterministic steps. The nightly pins FSDP2, native/SDPA attention,
fixed seeds, and greedy OCR sampling. The L2 smoke leaves the strategy at the
FSDP default and probes FlashAttention.

The nightly hook sorts by `extra_info.repeat_index` before pairing so noise
follows dataset order. Pairing keeps the top and bottom reward of each prompt
(`rollout.n=4`). Dump comparison realigns those rows by the same key. The
nightly reward model returns stable scores, so chosen and rejected do not swap.
Refresh `baseline/qwen_image_online_dpo/` if the baseline predates the hook sort.

Direct-preference steps do not recompute SDE `old_log_probs`. Performance
comparison includes `timing_s/prepare_actor_batch` and `timing_s/ref` and does
not include `timing_s/old_log_prob` or `perf/mfu/actor_infer`. Reward-derived
tensors stay informational because OCR scores are not bit-exact.

```bash
bash tests/nightly/qwen_image_online_dpo/run_qwen_image_online_dpo.sh
```
