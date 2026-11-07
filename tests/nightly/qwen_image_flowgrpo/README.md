# Qwen-Image FlowGRPO Nightly

FlowGRPO on local tiny-random Qwen-Image weights. Shared run steps, baselines,
thresholds, and failure triage are in `tests/nightly/README.md`.

This case is policy-gradient FlowGRPO with an SDE window of `[0, 2]`.
`log_probs`, latents, and timesteps are bound to the sample at rollout, so the
shared `extra_info.repeat_index` reorder at comparison time is enough. The loss
does not draw them again in batch order.

Precision tensors on top of the shared actor-forward and LoRA-gradient dumps:

- `log_probs`, `old_log_probs`, `latents`, `all_latents`, and `all_timesteps`.

Performance comparison includes `timing_s/old_log_prob` and
`perf/mfu/actor_infer`.

```bash
bash tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh
```
