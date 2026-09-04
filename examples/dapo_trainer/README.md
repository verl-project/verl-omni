# Qwen3-Omni Thinker DAPO Trainer

Last updated: 08/28/2026

This example provides the first Qwen3-Omni Thinker DAPO milestone on the V1
omni trainer: GPU LoRA training on multimodal AVQA with clip-higher,
token-level policy gradient, GRPO advantages, and the AVQA choice reward.

The Phase 1 launcher intentionally disables dynamic sampling:

```text
algorithm.filter_groups.enable=false
```

It also does not enable the overlong reward buffer. Those components are kept
out of this baseline so the token-level DAPO policy path can be validated
independently. In this Phase 1 recipe, DAPO refers to vanilla token-level
policy loss with asymmetric clipping, GRPO advantages, and no KL penalty. The
registered naive reward manager calls the AVQA `choice_reward`; the reward
manager name alone does not select the optimization algorithm.

**Phase 2 (#446): overlong reward buffer.** Overlong shaping is wired through
`reward.reward_kwargs` and only applies with `reward.reward_manager.name=dapo`
(`source=register`) — it is a no-op under the `naive` manager this example
uses. See `tests/special_e2e/run_dapo_qwen3_omni_thinker_lora_v1_smoke.sh` for
a working `name=dapo` recipe with overlong shaping enabled, and
`tests/utils/test_dapo_overlong_reward_on_cpu.py` for the reward-shape
contract: `reward.reward_kwargs.overlong_buffer_cfg.{enable,len,penalty_factor,log}`
and `reward.reward_kwargs.max_resp_len`.

**Phase 3 (#446): dynamic sampling.** The upstream V1 `PPOTrainer` replay
buffer already implements group filtering — this is config-only, no new
trainer code. `run_qwen3_omni_thinker_dapo_dynamic_sampling_lora_v1.sh` sets
`algorithm.filter_groups.enable=true` with `metric=acc` (the naive/DAPO
reward managers always populate `reward_extra_info["acc"]`, see
`verl.experimental.reward_loop.reward_manager.naive`), so the trainer drops
uniform-reward groups (all-correct or all-wrong) and keeps generating until
`data.train_batch_size` qualified prompts are collected, bounded by
`algorithm.filter_groups.max_inflight_gen_batches`. Group filtering requires a
streaming reward path (`reward.reward_model.enable=false`, the default), so
this recipe uses the `dapo` reward manager rather than Phase 1's `naive` one.

## Run

Download and extract the AVQA-R1-6K data, then convert it from the repository
root:

```bash
python examples/gspo_trainer/data_process/avqa.py \
    --input_dir /path/to/raw/AVQA_R1 \
    --output_dir ~/data/avqa_r1_6k
```

The converted parquet stores absolute image and audio paths. Every Ray worker
must mount the converted dataset and its media files at the same absolute path
used during conversion.

Install the audio and multimodal processing dependencies on every Ray worker,
then launch:

```bash
pip install -e ".[audio]"
pip install qwen-vl-utils
```

```bash
bash examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_v1.sh
```

The default model is `~/models/Qwen/Qwen3-Omni-30B-A3B-Instruct`. Override the
model, data, or any Hydra setting without editing the script:

```bash
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/validation.parquet \
bash examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_v1.sh \
    trainer.total_training_steps=2
```

Validation runs once before training and every 10 steps by default. It uses
greedy decoding (`n=1`, `do_sample=false`, `temperature=0`, `top_p=1`,
`top_k=-1`) over the full validation split. Plot
`val-core/avqa_r1_6k/reward/mean@1` against the trainer step for the directly
comparable in-trainer validation curve.

Only the Thinker LoRA adapters are trained. Talker, code2wav, code predictor,
visual projection, and audio-tower modules are excluded, and the vision tower
is frozen, matching the existing GSPO V1 baseline.
