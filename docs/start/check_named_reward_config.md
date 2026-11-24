# Check named reward configuration

Last updated: 11/24/2026

Run this preflight with the same trainer configuration root and reward overrides
you plan to use for training:

```bash
python tools/check_named_reward_config.py \
  --config-dir verl_omni/trainer/config \
  --config-name diffusion_trainer \
  '++reward.models.pickscore.backend=native' \
  '++reward.models.pickscore.placement.devices=[0]' \
  '++reward.models.pickscore.executor.model=pkg://verl_omni.utils.reward_score.scorers.pickscore:PickScoreScorer' \
  '++reward.reward_functions.ps.path=pkg://verl_omni.utils.reward_score.scorers.pickscore' \
  '++reward.reward_functions.ps.name=compute_score_pickscore_batched' \
  '++reward.reward_functions.ps.model=pickscore'
```

The two config options default to this checkout's `verl_omni/trainer/config` root and
`diffusion_trainer`. Set `--config-name omni_trainer` for the Omni root.
Hydra composes the full trainer configuration before the check. A standalone
YAML file must supply the required trainer defaults; the preflight does not
invent missing fields.

The example uses the current PickScore scorer and glue paths from the
[scorer integration guide](../contributing/integrating_a_reward_scorer.md).
Checking this config does not establish that the model can run on the chosen
hardware.

Exit code `0` prints one JSON object to stdout. It contains the ordered model
names and backends, model-backed term names and references, and the existing
`requires_reward_role` decision. An empty `reward.models` mapping is valid for
this scoped report. Configuration errors print a message to stderr and exit `2`.

The check uses the current named-model schema and term-reference validators,
including the exclusion between enabled legacy `reward.reward_model` and
`reward.models`. `--help` uses only Python's standard library; validation uses
normal repository and package imports and requires their installed dependencies.
It does not resolve configured scorer or model import strings,
load weights, start Ray, or allocate a device. It does not check every reward key,
scorer keyword argument, reward-manager selection, import availability, pool
capacity, or whether a training run will succeed. Keep existing trainer and
reward-loop checks for those decisions.
