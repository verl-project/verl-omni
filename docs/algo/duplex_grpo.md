# Streaming duplex GRPO (MiniCPM-o 4.5)

Last updated: 11/19/2026.

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
whose input track is fed at the native 1 s cadence while the policy emits one
action token per unit (`<|listen|>` to stay silent, content/boundary tokens
to speak). Training on those decisions needs three things the turn-based
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
| Unit-window replay | `.../duplex/replay.py` | `<unit>` + 10 bare audio placeholders + action + `</unit>` per unit; engine silent units replay as 1 s of digital silence; mel via the processor's `process_audio_streaming` |
| Exact-distribution recompute | `.../duplex/sampling_logprobs.py` | Ports the engine's truncation math; the M0 parity gate's core |
| Event mining | `.../duplex/mining.py` | pause / turn / interruption from word-level timestamps |
| Rule rewards | `verl_omni/utils/reward_score/duplex_turn_taking.py` | Per-axis, timestamp-only, tuned bands |
| Telemetry + trip-wires | `.../duplex/telemetry.py` | duplicate n-gram share, silence share, speak ratio |
| Calibration gate | `.../duplex/calibration.py` + `scripts/duplex_calibration_audit.py` | mid-band entry gate before any gradient |
| FDB-v1 rule metrics | `.../duplex/fdb_metrics.py` | static-protocol timing metrics over dumped replays |
| Rollout bridge | `verl_omni/pipelines/minicpm/duplex_rollout_adapter.py` | `pipeline_mode="duplex"`; `minicpmo_4_5_duplex` |
| Session driver | `verl_omni/workers/rollout/vllm_rollout/vllm_omni_duplex_strategy.py` | 1 s-cadence submits next to the engine; per-response draining; close before weight sync |
| Agent loop | `verl_omni/agent_loop/omni_duplex_agent_loop.py` | `omni_duplex_agent`; one Ray call per episode |
| Recipe | `examples/grpo_trainer/minicpm_o/duplex_grpo/` | GRPO LoRA, 4×H800 colocated |

### The replay layout

The engine's duplex data plane feeds, per unit, `<unit>` + ten bare
audio-embedding placeholders, samples one action token, and closes the unit
with `terminator + </unit>` when the next opens. The trainer replay mirrors
that layout exactly — the standard processor rendering (`<|audio_start|> …
<|audio_end|>` marker spans) is a *different* sequence, so duplex batches
carry **explicit `audio_bounds`** from the replay instead of marker-derived
ones (`MiniCPMThinkerAdapter.prepare_model_inputs` takes the duplex branch
when the batch carries `extra_fields["duplex_replay"]`; packed batches re-base
the bounds on the position-id resets). The prompt is the session's
system-context prefix; the response is the interleaved unit-window sequence
with mask 1 only at sampled action positions.

### Log-probability semantics

Captured logprobs are of the sampled distribution, not a plain softmax. For
MiniCPM-o 4.5's two-stage draw the joint is `log P(chunk_eos)` (raw softmax)
for a boundary hit and `log(1 − P) + stage-2` (post-truncation renormalized)
for a content token. Old log-probs ride the rollout capture (the in-tree
processed-logprobs convention); the reference KL and the M0 parity gate
recompute under the same transform from the recorded distribution inputs
(`sampling_logprobs.py`).

### Weight-sync fencing

Every session the driver opens is closed before the episode returns, so the
engine's fence (`weight_sync_sessions_open` on `start_weight_update` /
`add_lora` while sessions with appended data are open) is a backstop, not the
mechanism. Pre-warmed sessions may stay open (open performs no forward).

## Data

Single PoC corpus: `kyutai/DailyTalkContiguous` (stereo two-track,
word-level timestamps). Mining (pure timestamp arithmetic, CPU):

- **pause** — user intra-utterance silences ≥ 0.8 s (window: 2 s lead + tail);
- **turn** — the tail of each user utterance ≥ 1 s (3 s lead, 2 s tail);
- **interruption** — user utterances that overlap a model utterance in the
  corpus; the *live* injection point is the policy's own speak onset plus a
  delay, so group members may see slightly different overlaps (accepted).

Episodes are bounded (≤ 8 s mined windows; the session horizon caps the
tail). At the model's 1 Hz decision cadence a 3–5 s segment holds ~3–5
decisions, hence `rollout.n=16` groups.

## Entry gates and validation

1. **Calibration** (`scripts/duplex_calibration_audit.py`): the unmodified
   checkpoint's per-axis normalized mean reward must be ≥ 0.4 of the axis
   band before any gradient step.
2. **M0 parity** (`scripts/duplex_m0_parity_check.py`): same-seed episodes
   replay deterministically; actor-recomputed logprobs match the captured
   records within tolerance on all sampled tokens, including a
   `truncated_interruption` episode; every forced class maps to mask 0.
3. **In-loop trip-wires**: duplicate 3-gram share ≤ 0.4, silence share and
   speak ratio inside pre-registered bands (rule-reward degeneracy is the
   documented failure mode of duplex RL).
4. **FDB-v1 rule metrics** over dumped replays
   (`python -m verl_omni.utils.dataset.duplex.fdb_metrics dump.jsonl`).

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
