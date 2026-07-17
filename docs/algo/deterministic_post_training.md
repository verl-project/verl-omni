# Deterministic Post-Training

Last updated: 07/10/2026.

verl-omni RL training is **not bitwise reproducible** by default: identical configs run twice can produce different reward curves due to nondeterminism in sampling. This page describes how to make VLM reward-model (GRM) scoring reproducible, focusing on the standard vLLM VLM path.

## When to use

- **Debugging** — reproduce a training failure exactly.
- **Regression testing** — verify a code change has no silent effect on reward scores.
- **Research** — fair comparison of algorithmic changes.

## Enable

Set `reward.reward_model.full_determinism=true` and a `seed` under the `reward_model` block:

```yaml
reward:
  reward_model:
    enable: true
    full_determinism: true
    seed: 42
```

Or via Hydra overrides:

```bash
python -m verl_omni.trainer.main_diffusion \
  reward.reward_model.enable=true \
  reward.reward_model.full_determinism=true \
  reward.reward_model.seed=42 \
  [other config overrides...]
```

Sampling params (`temperature`, `top_k`, `top_p`) remain user-configured under `reward.reward_model.rollout`; the `full_determinism` flag does not override them.

## How it works

verl-omni's GRM scores images via `/v1/chat/completions` (a VLM such as Qwen3-VL transcribes the image). Reproducibility is achieved with a **per-request sampling seed**: `VisualRewardManager` injects `seed = reward_model.seed` into each GRM request's `sampling_params`. vLLM's OpenAI-compatible endpoint honors the `seed` field, so the same image + same sampling params + same seed yields the same transcription, and therefore the same reward score, across runs.

This is **controlled pseudorandom** sampling, not greedy decoding — `temperature`/`top_k`/`top_p` are respected, and the seed makes the randomness reproducible.

### Sampling params

`do_sample` is **not** a valid vLLM `SamplingParams` key (silently ignored), so it is not passed. Greedy decoding is achieved with `temperature=0`; controlled pseudorandom sampling uses `temperature>0` plus the per-request `seed`. `max_tokens` is always supplied (default `4096`) so generation length is stable.

### VLM scoring path

When RM is enabled and no `custom_reward_function.path` is provided, the path is auto-set to `compute_score_ocr`.

## Limitations

- **Best-effort determinism**: vLLM's per-request `seed` guarantees reproducibility at the sampling-logic level. Full bit-level reproducibility may still be affected by GPU-kernel non-determinism (reduction order, batch composition) on some hardware/versions.
- **Generative GRM only**: this covers the generative VLM reward path. Full E2E determinism (rollout + diffusion sampling) is tracked separately in [`rfc/deterministic-full-rfc.md`](../../rfc/deterministic-full-rfc.md).

## Verify

```bash
pytest tests/reward_loop/test_visual_reward_manager.py::test_deterministic_reward_reproducibility -v -s
```

Each run spins up its own `RewardLoopManager` (a fresh RM server process), so reproducibility is checked across independent processes. The test asserts (1) two runs with the **same seed** produce bitwise-aligned `rm_scores` and `genrm_response`, and (2) runs with **different seeds** produce different outputs (proving the seed drives sampling rather than being a no-op).
