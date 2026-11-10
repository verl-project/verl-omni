# Qwen3-Omni Thinker GSPO Nightly

GSPO LoRA on a local tiny-random Qwen3-Omni Thinker. Shared run steps,
baselines, thresholds, and failure triage are in `tests/nightly/README.md`.

This is the v1-trainer form of
`tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_smoke.sh`:
`verl_omni.trainer.main_omni`, `pipeline_name=qwen3_omni_moe`, and the same
Thinker module filter, stretched to 20 steps on 1 GPU. The script builds the
tiny checkpoint and a repeated GSM8K prompt before training.

Rollout is greedy (`temperature=0`, `do_sample=false`, `seed=42`) so the same
prompt keeps the same token ids. Dump comparison reads two payload kinds:

- `actor_update_batch`: the actor mini-batch, including `old_log_probs`,
  `ref_log_prob`, and `rollout_log_probs` when those fields are present.
- `lora_gradients`: LoRA parameter gradients before the optimizer step.

`batch.responses` and `batch.advantages` stay informational because the shared
comparator treats those names that way. A token-id change still fails the run
when it moves `old_log_probs` or the LoRA gradients. Rule reward scores are
compared. There is no OCR reward model.

The language-model MFU counter is collected. `collect_metrics.py` warns that
this model and algorithm pair is not in the diffusion MFU allowlist.

```bash
NUM_GPUS=1 \
POLICY_MODEL_PATH=/path/to/tiny-random/Qwen3-Omni \
bash tests/nightly/qwen3_omni_thinker_gspo/run_qwen3_omni_thinker_gspo.sh
```
