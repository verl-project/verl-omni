# MiniCPM-o 4.5 Thinker GSPO Nightly

GSPO LoRA on a local tiny-random MiniCPM-o 4.5 Thinker. Shared run steps,
baselines, thresholds, and failure triage are in `tests/nightly/README.md`.

This is the v1-trainer form of
`tests/special_e2e/run_gspo_minicpmo_thinker_lora_v1_smoke.sh`:
`verl_omni.trainer.main_omni`, `pipeline_name=minicpmo_4_5`, the AVQA dataset
class, and the same frozen-tower exclusion, stretched to 20 steps on 1 GPU.
The script builds the tiny checkpoint and exports `L3_DATA_KIND=avqa`.
`run_l3_case.sh` rewrites the repeated image-audio parquet on every run
through `tests/nightly/create_sample_data.py`. Each row gets
`extra_info.repeat_index` so dump comparison can realign prompts across runs.

Rollout is greedy (`temperature=0`, `do_sample=false`, `seed=42`). Dump
comparison reads two payload kinds:

- `actor_update_batch`: the actor mini-batch, including `old_log_probs`,
  `ref_log_prob`, and `rollout_log_probs` when those fields are present.
- `lora_gradients`: LoRA parameter gradients before the optimizer step.

`batch.responses`, `batch.advantages`, and `batch.old_log_probs` stay
informational. Nested `old_log_probs` pads inconsistently on the tiny model;
`rollout_log_probs` covers token-id drift. A token-id change still fails the
run when it moves `rollout_log_probs`, `ref_log_prob`, or the LoRA gradients.
The choice reward is compared. There is no reward model.

The builder needs the MiniCPM-o 4.5 remote code and tokenizer. It does not
download the full checkpoint. The language-model MFU counter is collected.
`collect_metrics.py` warns that this model and algorithm pair is not in the
diffusion MFU allowlist.

```bash
NUM_GPUS=1 \
POLICY_MODEL_PATH=/path/to/tiny-random/MiniCPM-o-4_5 \
bash tests/nightly/minicpmo_4_5_thinker_gspo/run_minicpmo_4_5_thinker_gspo.sh
```
