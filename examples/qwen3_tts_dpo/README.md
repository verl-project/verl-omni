# Qwen3-TTS Talker Online DPO (LLM-audio judge)

Online DPO post-training of a Qwen3-TTS talker: for each prompt the actor generates two candidates,
a remote audio-capable LLM judges which reads the text more naturally, and the talker trains with a
pairwise DPO loss on its codec-0 sequence log-prob against a frozen reference. It reuses the GSPO
recipe's talker actor, vLLM-Omni AR rollout, stage config and dependency patches
(`../qwen3_tts_gspo_trainer/`); only the loss, the reward, and the batch invariants differ.

## How it works

- **Rollout**: `rollout.n=2` candidates per prompt (the vLLM-Omni AR talker, same stage config as GSPO).
- **Reward**: `AudioJudgeRewardManager` decodes both candidates reward-side, buffers them by `uid`, and
  sends the pair to a judge `/rank` endpoint in one call. Each candidate's rating becomes `sj_score`.
- **Loss**: `verl_omni.trainer.main_ppo` binds `tts_dpo_loss` on the actor (via the same `set_loss_fn`
  seam base verl uses for the critic). Per sequence it sums the current and reference log-probs over
  the response, forms the implicit reward `r = sum(log pi_theta - log pi_ref)`, and minimizes
  `-logsigmoid(dpo_beta * (r_chosen - r_rejected))` plus an optional `dpo_nll_lambda * NLL(chosen)`.
  Pairing is internal to the loss (group by `uid`, best vs worst by `sj_score`); a judge tie yields a
  zero-gradient pair. No base-verl patch is involved.

Launch with `python3 -m verl_omni.trainer.main_ppo` (not `verl.trainer.main_ppo`): the omni entry is
the only difference, adding the DPO loss binding; every other recipe still runs on the base entry.

## Load-bearing invariants (baked into the config)

Online DPO needs each prompt's two candidates to reach the loss together in one in-order micro-batch
on one data-parallel rank. The config enforces this and the loss asserts it:

- `reward.num_workers: 1` - both candidates of a prompt must rendezvous in one reward worker's event
  loop for the pairwise judge call. More workers split the group; the manager raises if it is not 1.
- `actor.use_dynamic_bsz: false` and `ppo_micro_batch_size_per_gpu` == per-rank candidate count - so
  the mini-batch is exactly one in-order micro-batch and a `uid` group is never split across micro-batches.
- `trainer.balance_batch: false` - seqlen balancing would scatter a prompt's two candidates across
  ranks; contiguous chunking keeps them together (needs `train_batch_size % dp == 0`).
- `actor.use_kl_loss: true` - only to build the reference worker so `ref_log_prob` exists. The DPO loss
  uses `ref_log_prob` directly; the PPO KL term is unused (`kl_loss_coef: 0.0`).

If the loss logs `formed no preference pairs`, a `uid` group was split: check the sizing above.

## Tuning

- `dpo_beta` (default 0.1): implicit-KL strength. Higher sharpens the chosen/rejected separation but can
  destabilize; lower is gentler.
- `dpo_nll_lambda` (default 0.0): an SFT anchor on the chosen sequence. Raise to 0.1-0.5 if the chosen
  log-prob collapses while `dpo_margin` keeps widening (the model gaming displacement).
- Judge by held-out eval, not the training curve: fresh pairs each step keep `dpo_acc`/`dpo_margin` noisy.

## The judge

`judge_server.py` is a standalone server speaking the `/rank` contract
(`{text, wavs_b64, debias} -> {scores, chosen, rejected}`). The provider is pluggable:

```bash
# Gemini (default)
GOOGLE_API_KEY=... python3 judge_server.py --provider gemini --model gemini-3.5-flash --port 8901

# Any OpenAI-audio-compatible endpoint (hosted, or a locally vLLM-served audio LLM)
python3 judge_server.py --provider openai --base-url http://localhost:8000/v1 \
    --model <audio-llm> --api-key-env OPENAI_API_KEY --port 8901
```

Point the run at it with `JUDGE_URL` (or `reward.judge_urls=[...]` for several replicas). Any server
that answers `/rank` works; the reward manager is provider-agnostic.

**Caveat**: an LLM judge discriminates weakly on same-text / same-voice pairs, so online DPO from it
can regress vs the SFT checkpoint even while training metrics look healthy. Treat this as clean
plumbing for an LLM-judge preference loop, and evaluate checkpoints on held-out data before shipping.

## Run

```bash
MODEL_PATH=<qwen3-tts SFT ckpt> \
TRAIN_FILE=~/data/tts_voice_synth/train.parquet \
VAL_FILE=~/data/tts_voice_synth/test.parquet \
SPK_EMBED=~/data/tts_voice_synth/spk_embed.json \
JUDGE_URL=http://localhost:8901 \
bash run_qwen3_tts_dpo.sh
```

Install and data prep are identical to `../qwen3_tts_gspo_trainer/` (same `[tts]` extra, qwen-tts, and
`patches/`); the judge additionally needs `google-genai` (Gemini) or `openai` (OpenAI-compatible).

## Files

```
examples/qwen3_tts_dpo/
├── config/qwen3_tts_dpo.yaml   recipe (inherits verl ppo_trainer; DPO loss + audio judge)
├── run_qwen3_tts_dpo.sh        launch (reuses ../qwen3_tts_gspo_trainer stage yaml + patches)
└── judge_server.py             generic /rank judge (Gemini or OpenAI-compatible)

verl_omni/trainer/main_ppo.py                        omni PPO/DPO entry (binds tts_dpo_loss)
verl_omni/workers/utils/losses.py                    tts_dpo_loss (pairwise codec-0 DPO)
verl_omni/workers/config/tts/actor.py                DPOPolicyLossConfig (dpo_beta / dpo_nll_lambda)
verl_omni/reward_loop/reward_manager/audio_judge.py  AudioJudgeRewardManager
```
