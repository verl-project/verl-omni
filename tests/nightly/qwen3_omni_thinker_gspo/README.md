# Qwen3-Omni Thinker GSPO Nightly

GSPO LoRA on a local tiny-random Qwen3-Omni Thinker. Shared run steps,
baselines, thresholds, and failure triage are in `tests/nightly/README.md`.

This is the v1-trainer form of
`tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_smoke.sh`:
`verl_omni.trainer.main_omni`, `pipeline_name=qwen3_omni_moe`, and the same
Thinker module filter, stretched to 20 steps on 1 GPU. The script builds the
tiny checkpoint and exports `L3_DATA_KIND=math`. `run_l3_case.sh` rewrites the
distinct GSM8K prompts on every run through `tests/nightly/create_sample_data.py`.
Each row gets `extra_info.repeat_index` so dump comparison can realign prompts
across runs. `data_source` stays `openai/gsm8k` for the naive GSM8K rule
reward; `--data_source` does not apply to this kind.

Rollout is greedy (`temperature=0`, `do_sample=false`, `seed=42`). Dump
comparison reads two payload kinds:

- `actor_update_batch`: the actor mini-batch, including `old_log_probs`,
  `ref_log_prob`, and `rollout_log_probs` when those fields are present.
- `lora_gradients`: LoRA parameter gradients before the optimizer step.

`batch.responses`, `batch.advantages`, and `batch.old_log_probs` stay
informational. Nested `old_log_probs` pads inconsistently on the tiny model;
`rollout_log_probs` covers token-id drift. A token-id change still fails the
run when it moves `rollout_log_probs`, `ref_log_prob`, or the LoRA gradients.
Rule reward scores are compared. There is no reward model.

The language-model MFU counter is collected. `collect_metrics.py` warns that
this model and algorithm pair is not in the diffusion MFU allowlist.

```bash
NUM_GPUS=1 \
POLICY_MODEL_PATH=/path/to/tiny-random/Qwen3-Omni \
bash tests/nightly/qwen3_omni_thinker_gspo/run_qwen3_omni_thinker_gspo.sh
```
