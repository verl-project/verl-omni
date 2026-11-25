# MiniCPM-o 4.5 streaming duplex GRPO (LoRA, 4×H800 colocated)

Streaming full-duplex GRPO over replayed two-track event episodes — the
trainer side of [RFC verl-omni #905](https://github.com/verl-project/verl-omni/issues/905),
consuming the duplex action-capture surface of
[vllm-omni #10627](https://github.com/vllm-project/vllm-omni/pull/10627).
See [the duplex GRPO doc](../../../docs/algo/duplex_grpo.md) for the full
design; this README covers running it.

## What trains

- **Episodes are real duplex sessions.** Each sample opens a vLLM-Omni session
  with `action_capture: true` and a per-episode seed, feeds the user track at
  the native 1 s cadence, and closes before the weight sync. The engine seals
  one `DuplexActionRecordSet` per response/listening scope with the exact
  sampling-distribution log-probability of every action.
- **Axes**: `pause` (stay silent through the user's intra-utterance silence),
  `turn` (take the floor promptly after the user's utterance ends), and
  `interruption` (yield promptly after the user's interrupting speech
  *begins* — the clip is injected relative to the policy's own speak onset,
  and a policy that was not speaking when it begins, or never spoke, takes
  the floor score; the harness never cancels model output).
- **Loss**: GRPO (`adv_estimator=grpo`, group = `rollout.n=16` episodes per
  segment) in its Dr.GRPO form: group advantage `R − mean(R)` without std
  division (`algorithm.norm_adv_by_std_in_grpo=false`) and a constant token
  denominator (`loss_agg_mode=seq-mean-token-sum-norm`,
  `loss_scale_factor=1024`), so 15-tie + 1-outlier groups on the near-binary
  1 Hz rewards are not amplified by √G. Text/control-token actions only,
  validity mask 1 on sampled actions and 0 on forced/re-emitted ones. KL on
  (β 0.05, `low_var_kl`) against the starting checkpoint; LoRA r=32 with
  frozen audio/vision towers.
- **Rewards**: CPU rule rewards from timestamps
  (`verl_omni/utils/reward_score/duplex_turn_taking.py`). The scorer also
  emits the trip-wire columns (`duplex_speak_ratio`, `duplex_silence_share`,
  `duplex_duplicate_ngram_share`) and a per-episode `duplex_episode_failed`
  flag whenever a pre-registered threshold (`telemetry.TRIP_WIRE_DEFAULTS`)
  breaches, plus `duplex_pacing_corrupted` whenever an engine-injected silent
  unit landed inside the scored window; the trainer aggregates these columns
  per step, so silence collapse, repetitive speech, or a rising drop rate
  shows up in the logged metrics. The agent loop drops a pacing-corrupted
  episode (mask 0, flagged the same way) and keeps it out of its group's
  baseline; the advantage is gated per token to the scored event's units on
  pause and interruption episodes.

## Prerequisites

1. **A vLLM-Omni build with the capture surface** — verl-omni's pinned
   revision is turn-based (issue
   [#919](https://github.com/verl-project/verl-omni/issues/919) owns the pin
   policy). For this experiment, override the pin in your training env:

   ```bash
   uv pip install --no-deps "vllm-omni @ git+https://github.com/vllm-project/vllm-omni@<capture-revision>"
   ```

   The rollout fails loudly (`requires a vLLM-Omni build with the duplex
   action-capture surface`) when the installed revision predates it.
2. **Mined episode parquet** from `kyutai/DailyTalkContiguous` — see the
   [mining section of the doc](../../../docs/algo/duplex_grpo.md#data). Build
   it on CPU before any training:

   ```bash
   python scripts/duplex_build_episodes.py --manifest manifest.jsonl \
       --output duplex_episodes.parquet
   ```

   Defaults: 5 % of dialogues held out by `crc32(dialogue_id)` (a stable,
   per-dialogue split), axes balanced to ≈1:1:1 inside each split
   (`--per-axis-cap auto`), pauses mined from 0.6–1.8 s same-speaker
   silences, interrupting clips ≥ 1.5 s.
3. **The M0 entry gate** — before any gradient step, run the calibration
   audit (`scripts/duplex_calibration_audit.py`) and the parity check
   (`scripts/duplex_m0_parity_check.py`) on the unmodified checkpoint. The
   calibration gate is two-sided: every axis's normalized mean reward must
   land in [0.30, 0.80] of the axis band.

## Run

```bash
export MODEL_PATH=/path/to/MiniCPM-o-4_5
export TRAIN_FILE=/path/to/duplex_episodes_train.parquet
export VAL_FILE=/path/to/duplex_episodes_val.parquet
bash run_minicpmo_4_5_duplex_grpo_lora_v1.sh
```

Defaults: 4×H800, colocated sync trainer, rollout TP=2, batch 16 segments ×
16 episodes, 200 steps. Sizing note (RFC §3.8): a 3–5 s segment contains only
~3–5 listen/speak decisions at the model's 1 Hz cadence, hence the larger
group.

## Evaluation

- Per-axis reward means and the trip-wire columns from the scorer, aggregated
  per training step by the trainer's reward-extra metrics (the
  `duplex_episode_failed` share is the degeneracy alarm;
  `reward/duplex_pacing_corrupted` is the pacing drop rate).
- FDB-v1 rule metrics over the dumped replays:

  ```bash
  python -m verl_omni.utils.dataset.duplex.fdb_metrics "${OUTPUT_DIR}/rollout/*.jsonl"
  ```
- Exit criteria (English duplex non-regression): every axis's val `R_norm`
  moves by at least −0.03 with one axis improving, the trip-wire columns stay
  inside their bands, `actor/kl_loss` stays bounded per step, and a 10–20
  episode transcript spot-check of the dumped replays finds no silence
  collapse or repetitive speech.

## Validation status

CPU-tested at contract level (`tests/utils/dataset/duplex/`, the strategy and
agent-loop suites); GPU parity (actor-recomputed logprobs vs captured records)
is the M0 gate script, and end-to-end training is **to be** validated on the
GPU servers with this recipe — neither has run yet at PR time. This is PoC
scope: capability verification, not SOTA.
