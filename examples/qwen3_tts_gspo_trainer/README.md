# Qwen3-TTS Talker GSPO Trainer

RL post-training of `Qwen/Qwen3-TTS-12Hz-1.7B-Base`: the talker learns to speak text lines
better, scored by audio reward functions on the decoded waveform. FSDP actor (talker codec-0
path) + vLLM-Omni AR rollout, GSPO token-level policy loss + GDPO per-dimension advantage
estimator (both from base verl).

Validated end-to-end with a full training pass over a ~25k-line voice dataset on 1 node x 8 H200
(reward rises on all dimensions; rollout and actor codec log-probs match to Pearson >= 0.999).

## What is trained / frozen

- Trainable: the talker (codec-0 main path: `talker.model` + `talker.codec_head`).
- Frozen: sub-talker / `code_predictor` (codebooks 1-15), `speaker_encoder`, `code2wav`.
- Reward is computed on the fully decoded 24 kHz waveform; gradient flows only to the talker.
- Inverts the GSPO Thinker recipe (`examples/gspo_trainer/`), which trains the Thinker and
  excludes the talker.

The talker is not a plain causal LM (2-channel input, speaker slot, 16 codebooks), so
`verl_omni/models/transformers/qwen3_tts.py` (loaded via `external_lib`) installs a custom
teacher-forced codec-0 forward; the pure math lives in `qwen3_tts_forward.py` and is CPU-tested.

## Reward

`MultiAudioRewardManager` decodes the rollout's codec tokens reward-side once (decoder selected
by `reward.audio.codec`, so other codec models can be added) and passes the waveform to every
function declared under `reward.reward_functions`. Each entry loads one reward function that
owns one scoring model, so backends can be reweighted, tweaked, or dropped independently in
config; entry fields beyond path/name/weight are forwarded to the function as kwargs
(e.g. `whisper_model`, `device`).

| entry | reward form | model | module in verl_omni/utils/reward_score/tts/ |
|-------|-------------|-------|---------------------------------------------|
| `asr` | intelligibility exp(-2.5 * CER); extras: `stab` truncation / repetition / outlier penalty | whisper (GPU torch-native by default) | asr_reward.py |
| `sim` | speaker similarity (1 + cos) / 2 vs the clone/target ref | 3D-Speaker ERes2Net | spk_sim_reward.py |
| `emo` | P(tagged emotion), or 1 - P(neutral) untagged | emotion2vec_plus_large | emo_reward.py |

The manager emits `reward/<key>` per entry plus `reward/<key>/<extra>` for dict extras, and
trains on the weighted sum. With `adv_estimator: gdpo`, the dimensions named by
`algorithm.gdpo_reward_keys` are instead z-scored within each prompt group before fusing, so one
noisy dimension cannot dominate. Backends degrade gracefully (a failed backend scores 0.0,
below any scored clip).

Plain `AudioRewardManager` handles the single-function case: the standard
`default_compute_score_audio` path dispatches `data_source` `tts` to
`verl_omni/utils/reward_score/tts/__init__.py::compute_score`, a fixed-weight composition of
the same three modules.

## Data

`data_process/tts_content_synth.py` extracts every assistant message from a conversations JSONL
into verl parquet, cloning one fixed `--ref_audio` voice for each line:

```bash
python data_process/tts_content_synth.py \
  --input conversations.jsonl \
  --ref_audio clone_voice.wav \
  --output_dir ~/data/tts_voice_synth
```

`data_process/tts_verbalize.py` has the matching written-to-spoken transforms (URLs, emails) to
apply to text before synthesis; the reward side folds text with the consistent Whisper-standard
normalizer.

Precompute the clone voice's x-vector once (it feeds both rollout and actor, so generation and
the teacher-forced recompute condition on the same speaker):

```bash
python data_process/precompute_spk_embed.py --ref clone_voice.wav --out ~/data/tts_voice_synth/spk_embed.json
```

## Install

On top of the standard verl-omni GPU install, the recipe needs the TTS reward extras and the
qwen-tts package (reward-side code2wav + the trainable model classes):

```bash
pip install -e ".[tts]"
pip install --no-deps qwen-tts==0.1.1 sox
```

qwen-tts 0.1.1 pins transformers==4.57.3, which vllm 0.24 no longer supports; --no-deps plus a
small compat patch runs it on this repo's transformers 5.x. `patches/` carries that patch and the
gaps in the other pinned rollout dependencies, each an idempotent script to run once after
install (and again after any package reinstall, which restores pristine files):

```bash
for p in patches/patch_*.py; do python "$p"; done
```

| script | fixes |
|--------|-------|
| patch_qwen_tts_tf5.py | qwen-tts 0.1.1 modeling code on transformers 5.x |
| patch_vllm_omni_codec.py | vllm-omni 0.24.0 AR postprocess + output modality for "codec" stages |
| patch_verl_unpad_fallback.py | base verl: hard flash_attn import in attention_utils; only needed on envs without flash-attn |

Two further base-verl gaps this recipe hits are handled in process by the `external_lib` module
(`verl_omni/models/transformers/qwen3_tts.py`), so they need no script: the forward_only ref is
switched to manual param offload (FSDP-native CPUOffload leaves the leaf embedding tables on CPU
for the custom talker forward), and the composite-config `save_pretrained` is made non-fatal (it
trips transformers' to_diff_dict with KeyError 'dtype').

## Run

```bash
TRAIN_FILE=~/data/tts_voice_synth/train.parquet \
VAL_FILE=~/data/tts_voice_synth/test.parquet \
SPK_EMBED=~/data/tts_voice_synth/spk_embed.json \
bash run_qwen3_tts_gspo.sh
```

Any config field can be overridden on the CLI (standard hydra), e.g.
`bash run_qwen3_tts_gspo.sh trainer.total_epochs=2 reward.num_workers=4`.

## Files

```
examples/qwen3_tts_gspo_trainer/
├── config/qwen3_tts_gspo.yaml       recipe (inherits verl ppo_trainer; gspo + gdpo)
├── qwen3_tts_stages.yaml            vLLM-Omni single-stage AR talker config
├── run_qwen3_tts_gspo.sh            launch (volatile overrides only)
├── patches/                         one-time env fixes for the pinned deps (see Install)
└── data_process/
    ├── tts_content_synth.py         conversations jsonl to verl parquet
    ├── tts_verbalize.py             written to spoken text transforms
    └── precompute_spk_embed.py      clone-voice x-vector JSON

verl_omni/models/transformers/qwen3_tts.py         talker actor patch (external_lib)
verl_omni/models/transformers/qwen3_tts_forward.py pure codec-0 forward math (CPU-tested)
verl_omni/reward_loop/reward_manager/audio.py      AudioRewardManager (codec decode)
verl_omni/reward_loop/reward_manager/multi.py      MultiAudioRewardManager (weighted sub-rewards)
verl_omni/utils/reward_score/tts/                  asr / spk sim / emo reward functions
```

## Notes / troubleshooting

- On-policy correctness: the stage config neutralizes the model's deploy sampling defaults
  (top_k=-1, repetition_penalty=1.0) so the rollout's processed logprobs match the actor's
  full-softmax recompute. Keep `rollout.calculate_log_probs: true` and watch the logprob diff
  metrics on the first steps of any new environment.
- Rollout stops at codec eos: the untrained talker rambles to max_tokens; the rollout server
  injects codec eos as a stop token (derived from the model config), and the stage config caps
  `max_tokens: 384` (about 32s) to bound reward cost.
- Reward throughput: keep `reward.audio.score_threads: 1` (the code2wav decode is not thread-safe
  under CUDA; parallelism comes from `reward.num_workers` processes instead).
- HF hub flakes across many ranks: if you see transient `config.json not found` errors at init
  with a warm cache, export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for the training job.
