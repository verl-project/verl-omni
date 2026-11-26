# Streaming duplex GRPO (MiniCPM-o 4.5)

Last updated: 11/26/2026.

Streaming **full-duplex GRPO** over real duplex sessions — the trainer side of
[RFC #905](https://github.com/verl-project/verl-omni/issues/905), consuming
the duplex action-capture and policy log-probability surface of
[vllm-omni #10627](https://github.com/vllm-project/vllm-omni/pull/10627).
Goal: capability verification at PoC scale (one 4×H800 node) — time-aligned
trajectory capture, actor/rollout log-probability parity on streaming
actions, event-driven loss masking, temporal rule rewards, and weight sync
between duplex rollout batches.

## Why a duplex rollout is different

A full-duplex model listens and speaks concurrently; turn-taking, pause
retention, and barge-in yielding are the model's own per-second decisions. A
duplex episode therefore cannot be a prompt/response pair: it is a **session**
whose input track is fed at the native 1 s cadence while the policy decides
per 1 s unit (`<|listen|>` to stay silent; a spoken unit emits its content
tokens and the boundary `<|chunk_eos|>`). Training on those decisions needs three things the turn-based
paths do not have:

1. **Capture** — the engine records every action with the log-probability of
   the *exact distribution the runtime sampled* (post forbidden-token mask,
   repetition penalty, and temperature/top-k/top-p truncation), the forcing
   taxonomy (`sampled` / `forced` / `reemit`), input-chunk origins
   (`client` / engine-injected silence), and the derived per-response seed.
2. **Replay** — the trainer rebuilds the engine's unit-window sequence
   token-for-token and re-derives the audio embeddings from the fed PCM, so
   the actor conditions on exactly what the policy saw.
3. **Sessions, not generates** — the rollout server drives
   `DuplexOmni` sessions (open → paced appends → per-response record draining
   → close), and weight sync is fenced while sessions hold appended data.

## Architecture

```
dataset (mined episodes) ──▶ omni_duplex_agent ──▶ rollout server (DuplexOmni)
                                  │                       │  sessions + capture
                                  │  record sets (wire)   ▼
                                  │               DuplexStrategy.run_duplex_episode
                                  ▼
                     DuplexTrajectory ──▶ replay (unit windows + mel features)
                                  │              │
                                  │              ├──▶ AgentLoopOutput + extra_fields["duplex_replay"]
                                  │              │        (actor: explicit audio_bounds media branch)
                                  │              └──▶ rule rewards / telemetry / FDB metrics
```

| Piece | Where | Notes |
| --- | --- | --- |
| Wire mirror of `DuplexActionRecordSet` | `verl_omni/utils/dataset/duplex/records.py` | Strict validation; contract-tested against upstream when the capture surface is installed |
| Trajectory assembly | `.../duplex/trajectory.py` | Scopes in order; validity mask (`sampled`→1, `forced`/`reemit`→0); unit decision timeline |
| Unit-window replay | `.../duplex/replay.py` | `<unit>` + 10 bare audio placeholders + the unit's action tokens + `</unit>` per record (one record per 1 s unit); engine silent units replay as 1 s of digital silence; mel via the processor's `process_audio_streaming` |
| Event mining | `.../duplex/mining.py` | pause / turn / interruption from word-level timestamps |
| Rule rewards | `verl_omni/utils/reward_score/duplex_turn_taking.py` | Per-axis, timestamp-only, tuned bands: the turn axis measures from the user's turn end; the interruption axis measures from the interrupting speech *onset* (full credit within 1 s, −1 past 3 s) with a silence floor — a policy that was not speaking at the onset, or never spoke, takes −1. Emits the trip-wire columns, `duplex_pacing_corrupted`, and the per-episode `duplex_episode_failed` flag |
| Trip-wire definitions | `.../duplex/telemetry.py` | The shared threshold set (`TRIP_WIRE_DEFAULTS`), the reward bands, and the timing definitions every consumer calls — `speak_ratio`, `silence_share`, `duplicate_ngram_share`, the onset-anchored `interruption_yield_latency`, and the pacing predicate `pacing_corrupted` (an engine-injected silent unit inside the scored window); the scorer, the FDB metrics and the agent loop share them |
| Calibration gate | `.../duplex/calibration.py` + `scripts/duplex_calibration_audit.py` | Two-sided mid-band entry gate before any gradient: each axis's normalized mean must land in [0.30, 0.80] of the axis band |
| FDB-v1 rule metrics | `.../duplex/fdb_metrics.py` | static-protocol timing metrics over dumped replays |
| Rollout bridge | `verl_omni/pipelines/minicpm/duplex_rollout_adapter.py` | `pipeline_mode="duplex"`; `minicpmo_4_5_duplex` |
| Session driver | `verl_omni/workers/rollout/vllm_rollout/vllm_omni_duplex_strategy.py` | 1 s-cadence submits next to the engine; per-response draining; close before weight sync; reports the fed-chunk indices bracketing the interrupting speech |
| Agent loop | `verl_omni/agent_loop/omni_duplex_agent_loop.py` | `omni_duplex_agent`; one Ray call per episode; re-anchors interruption episodes onto the unit clock and drops pacing-corrupted episodes (mask 0, out of the group baseline) |
| Advantage gates | `verl_omni/trainer/omni/advantage_gates.py` | `OmniPPOTrainer._compute_advantage` applies two generic agent-loop columns after verl's scoring: `advantage_mask` (per-token policy-gradient gate, the replay's `pg_mask`) and `advantage_exclude` (a dropped episode leaves its GRPO group's baseline and takes advantage 0); reports `advantage/gated_token_share` and `advantage/excluded_sample_share` |
| Recipe | `examples/grpo_trainer/minicpm_o/duplex_grpo/` | GRPO LoRA, 4×H800 colocated |

### The replay layout

The engine's duplex data plane feeds, per unit, `<unit>` + ten bare
audio-embedding placeholders, samples the unit's action tokens (one per
decode step — a spoken unit carries its content tokens and the boundary
`<|chunk_eos|>`), and closes the unit with `</unit>` when the next opens.
The capture ledger folds a unit's decode steps into one record, so the
replay builds exactly one `<unit>` block per ingested 1 s unit
(engine-injected silence included) — intra-unit tokens and the re-emitted
terminator never add one. The trainer replay mirrors
that layout exactly — the standard processor rendering (`<|audio_start|> …
<|audio_end|>` marker spans) is a *different* sequence, so duplex batches
carry **explicit `audio_bounds`** from the replay instead of marker-derived
ones (`MiniCPMThinkerAdapter.prepare_model_inputs` takes the duplex branch
when the batch carries `extra_fields["duplex_replay"]`; packed batches re-base
the bounds on the position-id resets). The prompt is the session's
system-context prefix; the response is the interleaved unit-window sequence
with mask 1 only at sampled action positions. The payload also carries
**`pg_mask`**, the policy-gradient gate over the response: on pause and
interruption episodes the action tokens of units that decided strictly before
the scored event's start (`pause_window_s[0]` / `interrupt_speech_onset_s`)
are 0, so a yield failure does not penalize what the policy said beforehand.
Turn episodes are not gated — the premature-barge-in decision the −1 scores
precedes the turn end. Only the advantage takes the gate; the KL term rides
the full `response_mask`. Unit mels are re-derived through
the streaming processor and encoded through the streaming Whisper encoder
with the KV history carried across the episode's units, exactly as the engine
does; the reference block goes through the offline encoder on both sides.

### Log-probability semantics

One measure on both sides — the two-stage policy's pre-truncation joint:

- **Captured logprobs** ride the replay payload (`extra_fields["duplex_replay"]`)
  as the parity/audit channel: the M0 entry gate compares them against the
  same joint evaluated on the actor logits (`scripts/duplex_m0_parity_check.py`).
  They are the two-stage policy's pre-truncation joint, not a plain softmax:
  `log P(chunk_eos)` on the raw softmax for a boundary hit, and
  `log(1 − P) + log_softmax(mask(z) / T)[a]` for a content token, with the
  forbidden set `tts_pad + bad_token_ids + chunk_eos` masked out of stage 2.
- **Training logprobs** — `old_log_probs` (the ratio and the reference KL),
  the update forward's `log_probs`, and `ref_log_prob` — are recomputed by
  verl's actor forwards over the replayed sequence (`loss_mode=vanilla`, no
  `rollout_correction.bypass_mode`) under that same joint
  (`verl_omni/utils/dataset/duplex/policy_measure.py`), at every response
  position the mask keeps: forbidden set masked out of stage 2, boundary rows
  `log P(chunk_eos)` on the raw softmax, content rows
  `log(1 − P) + log_softmax(mask(z) / T)[a]`, greedy rows at `T = 1`. Positions
  with mask 0 keep verl's full-vocabulary value. The gradient follows the
  measure: the stage-2 term has exactly zero gradient on the forbidden logits,
  the stage-0 term depends on every logit (it is a raw softmax) and moves the
  content logits only through their aggregate mass `1 − P` — the relative
  log-odds among content tokens come from stage 2 alone. The dispatch is the
  `prepare_model_outputs` hook on `OmniFSDPEngine`; this pipeline is its only
  exerciser, and every other adapter takes verl's output unchanged. The
  rollout never exports its captured values as a log-probability column, so
  verl's `rollout_log_probs` channel stays empty and rollout-correction has
  nothing to engage on.

The objective is Dr.GRPO-style: the group advantage is `R − mean(R)` over the
`rollout.n=16` episodes of a segment without the std division (binary,
1 Hz-quantized rewards produce 15-tie + 1-outlier groups whose std-normalized
advantages are amplified by √G), the per-token losses aggregate under
a constant denominator (`loss_agg_mode=seq-mean-token-sum-norm` with
`loss_scale_factor` pinned to the rollout horizon), and the KL term
(`low_var_kl`, β 0.05) rides the full response mask against the starting
checkpoint. A pacing-corrupted (dropped) episode leaves the baseline: each
group's mean runs over its clean episodes only (the clean rows are rescored
through verl's GRPO estimator on that sub-batch) and the dropped episode's
advantage is 0. The per-token `pg_mask` gate scales the advantage after that.

### Weight-sync fencing

Every session the driver opens is closed before the episode returns, so the
engine's fence (`weight_sync_sessions_open` on `start_weight_update` /
`add_lora` while sessions with appended data are open) is a backstop, not the
mechanism. Pre-warmed sessions may stay open (open performs no forward).

## Data

Single PoC corpus: `kyutai/DailyTalkContiguous` (stereo two-track,
word-level timestamps). Mining (pure timestamp arithmetic, CPU):

- **pause** — same-speaker silences of 0.6–1.8 s (window: 2 s lead + tail);
- **turn** — the tail of each user utterance ≥ 1 s (3 s lead, 2 s tail);
- **interruption** — user utterances ≥ 1.5 s that overlap a model utterance in
  the corpus; the *live* injection point is the policy's own speak onset plus a
  delay, so group members may see slightly different overlaps (accepted).

Episodes are bounded (≤ 8 s mined windows; the session horizon caps the
tail). At the model's 1 Hz decision cadence a 3–5 s segment holds ~3–5
decisions, hence `rollout.n=16` groups.

Build the parquet on CPU, before any training:

```bash
python scripts/duplex_build_episodes.py --manifest manifest.jsonl \
    --output duplex_episodes.parquet
```

The manifest is one JSON line per dialogue (`dialogue_id`, the user track's
mono `user_audio`, `user_words`, `model_words`). Defaults: the val split
holds 5 % of dialogues by `crc32(dialogue_id)` — stable across processes,
by dialogue so none straddles the split — and `--per-axis-cap auto` then
caps every axis to the smallest one inside each split, so train and val are
both ≈1:1:1 across pause / turn / interruption (`0` keeps the natural,
turn-heavy mix; an integer caps during mining). The pause band is
`--min-pause-s 0.6` to `--max-pause-s 1.8`, the interrupting clip at least
`--min-interrupt-s 1.5`, and every row carries its `dialogue_id` and
`commit=true`. This checkpoint does not open a response on its own; the
dataset reads `commit` and the driver commits only when it is true. A
lead-in that would start before the track is clamped to 0, and event
anchors stay relative to that clamped window.

## Entry gates and validation

1. **Calibration** (`scripts/duplex_calibration_audit.py`): the unmodified
   checkpoint's per-axis normalized mean reward must land inside the mid band
   — at least 0.30 and at most 0.80 of the axis band (`--max-mean-ratio` for
   the ceiling) — before any gradient step. A floored axis means the event
   construction or reward band is misaligned with the checkpoint; a
   saturated one leaves its groups almost no variance to learn from.
2. **M0 parity** (`scripts/duplex_m0_parity_check.py`): same-seed episodes
   replay deterministically; the captured pre-truncation joint matches the
   same joint evaluated on the actor logits — per-row |delta| ≤ 0.05 nats and
   mean |delta| ≤ 0.01 nats over every sampled row — including a
   `truncated_interruption` episode; every forced class maps to mask 0.
3. **In-loop trip-wires**: the rule scorer emits `duplex_speak_ratio`,
   `duplex_silence_share`, and `duplex_duplicate_ngram_share` per episode
   (the shared telemetry definitions) plus `duplex_episode_failed` = 1 when
   any pre-registered threshold breaches (`telemetry.TRIP_WIRE_DEFAULTS`:
   duplicate 3-gram share ≤ 0.4, silence share inside 0.2–0.9, speak ratio
   inside 0.05–0.8; rule-reward degeneracy is the documented failure mode of
   duplex RL). The trainer aggregates every numeric reward-extra column per
   step — means with NaN-skip, and `inf` latencies surface as
   `<key>_never_share` — so a breach is visible (and the failed-share
   rises) instead of silently averaged away. The same loop drops an episode
   whose scored window an engine-injected silent unit landed in (pacing
   corrupted: the driver missed the 1 s deadline and the runtime's silence
   clock injected a unit) onto a mask-0 output flagged
   `duplex_pacing_corrupted` = 1, while clean episodes report 0, so the
   trainer's per-step mean `reward/duplex_pacing_corrupted` is the drop
   rate; the dropped episode leaves its group's baseline and takes advantage
   0 (`advantage/excluded_sample_share` reports the excluded share,
   `advantage/gated_token_share` the share of response-masked positions the
   pause/interruption gate zeroed).
4. **FDB-v1 rule metrics** over dumped replays
   (`python -m verl_omni.utils.dataset.duplex.fdb_metrics dump.jsonl`).
5. **Exit criteria** (English duplex non-regression against the base
   checkpoint): every axis's validation normalized mean reward (`R_norm`, the
   calibration audit's `normalized_mean` on the val episodes) moves by
   `Δ R_norm ≥ −0.03` with at least one axis improving; the trip-wire columns
   stay inside their bands; the per-step policy KL on the action tokens
   (verl's `actor/kl_loss`) stays bounded; and a 10–20-episode transcript
   spot-check of dumped replays after the run finds no silence collapse or
   repetitive speech.

A 4×H800 run of this loop (two consecutive steps, balanced episodes, the
capture branch) measured cold start ≈ 10 min and step time 324 s then
350 s (`gen` 210/272 s, `ref` 12 s, `update_actor` 27/32 s). Rewards sat
at mean −0.668 (min −1, max 0); `actor/kl_loss` 0; advantages mean
+0.0014; `response_length/mean` 85.1; old-logprob mismatch count 0.
Budget about 6 min/step, so 100 steps is about 10–11 h on that node,
plus about 0.5–1 min per eval. The script default remains
`TOTAL_TRAINING_STEPS=200`; override it per launch. Use the balanced
parquet (`--per-axis-cap auto`), not the 94 % turn-skewed full mine.

Calibration on 146 balanced val episodes: pause `normalized_mean` 0.367
(inside [0.30, 0.80]), turn 0.098, interruption 0.000. Turn was 46/51
exactly −1 and interruption 46/46 exactly −1. Under per-unit
auto-response the base policy speaks through the user's utterance, so
the turn floor (speech before the user's turn ends) and the interruption
floor (not speaking at the onset) both fire. The bands stay as specified.
The axes are not a learning signal until one of these lands: grade an
early barge-in instead of flooring it; keep today's bands as a yield
axis and add a speak-through axis; or mine turn episodes whose user
utterance is followed by silence. The M0 tolerances stay max |Δ| ≤ 0.05
nats and mean ≤ 0.01. The gate scripts now build an engine with
`enable_prefix_caching=false` and score through the same media
normalization as training. A uniform residual, if one remains, is the
bf16 engine tower versus the fp32 training tower, and that chooses
whether the tolerance or the cast changes.

`VERL_OMNI_DUPLEX_MAX_SESSIONS` is the engine admission cap. The rollout
adapter cannot see `rollout.agent.num_workers`, so the recipe exports the
same count (default 16). The reward manager is the registered
`minicpm_naive` (`reward.reward_manager.source=register`); the importlib
default looks the name up as a module attribute and misses it.

CPU tests cover the whole contract surface
(`tests/utils/dataset/duplex/`, the strategy / agent-loop / adapter suites);
the wire mirror is cross-checked against the installed upstream capture
implementation whenever it is available (`test_contract_on_cpu.py` skips at
the pin — see [#919](https://github.com/verl-project/verl-omni/issues/919)).

## Compatibility and the vLLM-Omni pin

The engine surface this consumes ships in vllm-omni #10627. verl-omni's
pinned vLLM-Omni revision stays turn-based; every duplex path imports the
capture surface lazily and fails loudly (with the required revision named)
when it is missing, so pinned installs keep working for everything else. The
pin bump itself is deliberately out of scope here — issue
[#919](https://github.com/verl-project/verl-omni/issues/919) owns it.

## Scope and future work

PoC scope: audio-only episodes, replay (no closed-loop user), rule rewards,
GRPO. Future work on the same trajectory contract: duplex DPO, streaming
duplex OPD (#566), LLM-judge rewards, backchannel axis, vision-in-episodes,
long-lived sessions across weight sync, scale-up corpora.
