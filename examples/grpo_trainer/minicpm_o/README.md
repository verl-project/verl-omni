# MiniCPM-o 4.5 Duplex GRPO

Last updated: 11/23/2026.

Implements the four unblocked trainer-side deliverables of [RFC #905](https://github.com/verl-project/verl-omni/issues/905): event-mining dataloader, rule reward scorers, FDB-v1 harness, and the GRPO LoRA recipe scaffolding.

The **duplex rollout bridge** (`pipeline_mode="duplex"`) and **trajectory assembly** depend on the engine-side action-capture surface ([vllm-project/vllm-omni#10536](https://github.com/vllm-project/vllm-omni/issues/10536) P0/P1) and are explicitly out of scope here. Everything in this directory is independently developable and testable.

## Components

| Component | Location | Purpose |
|---|---|---|
| Event miner | `verl_omni/utils/dataset/duplex/event_miner.py` | Mine pause/turn/interruption segments from DailyTalkContiguous |
| Calibration audit | `verl_omni/utils/dataset/duplex/calibration.py` | Entry gate: verify base-policy rewards land in mid band |
| Rule rewards | `verl_omni/utils/reward_score/duplex/rule_rewards.py` | Three-axis scoring + telemetry + trip-wires |
| FDB harness | `verl_omni/utils/reward_score/duplex/fdb_harness.py` | Regression gate: FDB v1 rule timing metrics |
| GRPO recipe | `run_minicpmo_4_5_duplex_grpo_lora_v1.sh` | Training script (needs engine capture to run) |

## Data preparation

```bash
# Download the corpus (2,541 dialogues, ~22 h, CC-BY-SA-4.0)
hf download kyutai/DailyTalkContiguous --local-dir ~/data/DailyTalkContiguous

# Mine event-typed segments (3–5 s pause/turn/interruption from word-level timestamps)
python verl_omni/utils/dataset/duplex/event_miner.py \
    --input_dir ~/data/DailyTalkContiguous \
    --output_dir ~/data/dailytalk_events
```

## Reward design (RFC #905 §3.5)

Three axes, each normalized to [-1, 0]:

| Axis | Formula | Event type |
|---|---|---|
| Pause violation | −1 if model speaks during tagged pause | pause |
| Turn latency | −(delay from turn end to model speech), full-credit band 1 s | turn |
| Interruption yield | −(delay from overlap end to model silence), deadline 2 s | interruption |

Telemetry + trip-wires (silence collapse, repetition degeneracy):
- `duplicate_trigram_share` ≤ 0.30
- `silence_ratio` in [0.05, 0.85]
- `speak_ratio` in [0.10, 0.90]

## Calibration entry gate (RFC #905 §3.6)

Before any gradient step, run the unmodified checkpoint over the mined event set. Per-axis mean normalized reward must be ≥ −0.6 (i.e. ≥ 40% of the way to optimal). An axis that floors the base model means the event construction or reward band is misaligned and must be fixed before training.

```bash
python verl_omni/utils/dataset/duplex/calibration.py \
    --trajectory_dir ~/data/dailytalk_events/base_rollout \
    --min_mean_reward -0.6
```

## Regression gate (RFC #905 §3.6)

Full-Duplex-Bench v1 static protocol, restricted to rule-based timing metrics:

| Metric | Threshold |
|---|---|
| Pause violation rate | ≤ 0.20 |
| Average turn latency | ≤ 3.0 s |
| Average yield latency | ≤ 2.0 s |
| Responded after overlap | ≥ 0.60 |

## Training

The recipe requires the duplex rollout bridge (engine action capture) before the first run. Once available:

```bash
MODEL_PATH=~/models/openbmb/MiniCPM-o-4_5 \
TRAIN_FILE=~/data/dailytalk_events/train.parquet \
bash examples/grpo_trainer/minicpm_o/run_minicpmo_4_5_duplex_grpo_lora_v1.sh
```

Topology: 4×H800 colocated sync trainer (rollout TP=2, LoRA merge, gradient checkpointing), matching the validated AVQA recipe.

## References

- [RFC #905](https://github.com/verl-project/verl-omni/issues/905) — duplex GRPO design
- [vllm-omni#10536](https://github.com/vllm-project/vllm-omni/issues/10536) — engine action-capture surface
- [DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous) — training corpus
- [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) — regression gate
- Kyutai interactivity alignment (arXiv 2606.11167) — replay-based duplex RL precedent
