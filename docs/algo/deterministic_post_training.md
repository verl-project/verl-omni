# Deterministic Post-Training

Last updated: 08/15/2026.

By default, verl-omni RL training is **not bitwise reproducible**: identical configs run twice can produce different reward curves due to nondeterminism in GPU kernels, batch composition, request routing, and sampling. This page documents the effort to make post-training deterministic across multiple aspects (reward inference, rollout generation, diffusion sampling, and eventually end-to-end bitwise-aligned reward curves).

**Supported: reward inference determinism** (VLM reward-model / GRM scoring). The remaining aspects are tracked as future work — see [Scope](#scope).

## Scope

| Component | Status |
|---|---|
| GRM inference — floating-point determinism | ✅ Supported |
| GRM inference — batch invariance (`VLLM_BATCH_INVARIANT`) | ✅ Supported |
| GRM inference — deterministic multi-replica routing (`NaiveRouter` crc32) | ✅ Supported |
| GRM inference — per-request sampling seed | ✅ Supported |
| Actor-rollout generation determinism | 🚧 Future |
| Diffusion sampling determinism | 🚧 Future |
| End-to-end bitwise-aligned reward curves | 🚧 Future |

For the upstream full-determinism feature (actor rollout, FSDP/Megatron, trainer backend), see verl's [`docs/advance/determinism.md`](https://github.com/verl-project/verl/blob/main/docs/advance/determinism.md). verl-omni will adopt those layers incrementally, and a single top-level determinism switch (covering both rollout and reward) will replace the current per-component `full_determinism` flags.

## When to use

- **Debugging** — reproduce a reward-score anomaly exactly.
- **Regression testing** — verify a code change has no silent effect on GRM scores.
- **Research** — fair comparison of reward-side algorithmic changes.

Leave determinism off for production training — deterministic kernels are slower, cuDNN benchmarking is disabled, and batch invariance can serialize.

## Enable

Set `reward.reward_model.full_determinism=true` and a `seed` under the `reward_model` block:

```yaml
reward:
  reward_model:
    enable: true
    full_determinism: true
    seed: 42
    rollout:
      name: vllm_omni
```

Or via Hydra overrides:

```bash
python -m verl_omni.trainer.main_diffusion \
  reward.reward_model.enable=true \
  reward.reward_model.full_determinism=true \
  reward.reward_model.seed=42 \
  [other config overrides...]
```

Sampling params (`temperature`, `top_k`, `top_p`) remain user-configured under `reward.reward_model.rollout`; the flag does not override them. Greedy decoding uses `temperature=0`; controlled pseudorandom sampling uses `temperature>0` plus the per-request `seed`.

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `reward_model.full_determinism` | `false` | Enables reward inference determinism (all layers below) |
| `reward_model.seed` | `42` | Floating-point + per-request sampling seed |
| `reward_model.rollout.name` | — | Use `vllm_omni` (the server with the `_post_init` determinism hook) |
| `reward_model.rollout.temperature` | — | User-configured sampling temperature |
| `reward_model.rollout.top_k` | — | User-configured top-k |
| `reward_model.rollout.top_p` | — | User-configured top-p |

## How it works

verl-omni's GRM scores images via `/v1/chat/completions` (a VLM such as Qwen3-VL transcribes the image). Reproducibility is enforced at three layers.

### Env export before `ray.init()`

When determinism is requested (currently `reward.reward_model.full_determinism=true`; a single top-level switch covering rollout + reward is future work — see [Scope](#scope)), `main_diffusion.run_diffusion` calls `_export_full_determinism_env` in the main process **before** `ray.init()` to set the determinism **switch** env vars:

- `VERL_FULL_DETERMINISM=1`, `VLLM_BATCH_INVARIANT=1`, `PYTHONHASHSEED=<seed>`

These are forwarded to every Ray actor via verl's `get_ppo_ray_runtime_env` (the runtime_env forward-list). They must be set before `ray.init()` so actors inherit them — too late if only set inside `_post_init`. The `NaiveRouter` subprocess (a `multiprocessing.Process` forked from an actor) reads `VERL_FULL_DETERMINISM` to activate crc32 routing, and the vLLM server reads `VLLM_BATCH_INVARIANT` to activate batch invariance.

Only switch flags are exported here (mirroring verl's `main_ppo`); the floating-point vars (`CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC`, `NCCL_DETERMINISTIC`, `NCCL_ALGO`, `NCCL_PROTO`, `VERL_DISABLE_FLASH_ATTN_CE`) are set inside each actor's `enable_full_determinism()`, so they do not pollute the normal training flow's environment.

### 1. Batch invariance

`VLLM_BATCH_INVARIANT=1` makes a request's output independent of which other requests are co-batched with it. This applies to the generative `/v1/chat/completions` path the GRM uses (unlike vLLM's `/classify` pooling endpoint, which is not covered by batch invariance — that path would require serializing with `max_num_seqs=1`; verl-omni's GRM does not use it).

Coverage is model- and hardware-dependent — see the [vLLM batch invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance/) (and [tested models](https://docs.vllm.ai/en/latest/features/batch_invariance/#tested-models)). If not covered, set `max_num_seqs=1` to serialize.

### 2. Floating-point determinism

The RM server (`vLLMOmniHttpServer`, reached via `rollout.name=vllm_omni`) applies `enable_full_determinism(seed)` in `_post_init`, before model load:

| Setting | Effect |
|---|---|
| `PYTHONHASHSEED` | freezes Python `hash()` / dict ordering |
| `CUBLAS_WORKSPACE_CONFIG=:16:8` | deterministic cuBLAS |
| `FLASH_ATTENTION_DETERMINISTIC=1` | deterministic flash-attn |
| `NCCL_DETERMINISTIC=1` / `NCCL_ALGO=Ring` / `NCCL_PROTO=Simple` | deterministic NCCL |
| `VERL_DISABLE_FLASH_ATTN_CE=1` | bypass the non-deterministic flash-attn Triton cross-entropy kernel in `logprobs_from_logits` (pure-PyTorch `log_softmax+gather`); not covered by `FLASH_ATTENTION_DETERMINISTIC` (attention backward only) or `torch.use_deterministic_algorithms` (Triton custom ops skip `warn_only`) |
| `VERL_SEED` | read by rollout worker subprocesses (verl `vllm_rollout/utils.py`) to apply `enable_full_determinism` per worker |
| seeded `random` / `numpy` / `torch` / `cuda` RNGs | fixed RNG state |
| `torch.use_deterministic_algorithms(True, warn_only=True)` | deterministic ops |
| `cudnn.deterministic=True, benchmark=False` | deterministic cuDNN |

`full_determinism`/`seed` live on the `reward_model` block (a non-structured config node), so the trainer propagates them onto `reward_model.rollout` via `OmegaConf.open_dict` — the server process's `self.config` is the rollout config and can read them.

Each RM replica uses the same `seed` (not `replica_rank + seed`), so every replica starts from identical state.

### 3. Deterministic routing

The reward router (verl's `NaiveRouter`) is least-loaded; under `VERL_FULL_DETERMINISM=1` it tie-breaks among equally-loaded replicas with `crc32(request_body) % len(candidates)`, so the same request lands on the same replica across runs. Combined with identical per-replica seeds and batch invariance, the same request yields the same output regardless of which replica serves it.

### 4. Per-request sampling seed

`VisualRewardManager` injects `seed = reward_model.seed` into each GRM request's `sampling_params`. vLLM's OpenAI-compatible endpoint honors the `seed` field, so the same image + same sampling params + same seed yields the same transcription, and therefore the same reward score. This is **controlled pseudorandom** sampling, not greedy — `temperature`/`top_k`/`top_p` are respected, and the seed makes the randomness reproducible.

`do_sample` is **not** a valid vLLM `SamplingParams` key (silently ignored), so it is not passed. `max_tokens` is always supplied (default `4096`) so generation length is stable.

### VLM scoring path

When RM is enabled, the trainer defaults `reward.custom_reward_function` to the OCR GRM scorer (`verl_omni/utils/reward_score/genrm_ocr.py::compute_score_ocr`) when no custom path is set. Override `path`/`name` to use a different scorer.

## Side Effects

- **Performance**: deterministic kernels are slower and cuDNN benchmarking is disabled; expect throughput loss.
- **Recommendation**: enable for debugging, regression testing, or research only.

## Limitations

- **Batch invariance not directly asserted**: `genrm_ocr` sends one image per `/v1/chat/completions` request serially, so requests are not explicitly co-batched and `VLLM_BATCH_INVARIANT` cannot be reliably triggered or observed in the test. Batch invariance is still set by `_post_init` for correctness; its coverage is model/hardware-dependent (see [vLLM batch invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance/)).
- **Nondeterministic fallbacks**: some GPU ops have no deterministic implementation. `warn_only=True` emits warnings; these are expected.
- **Multi-replica routing not asserted**: every replica uses the same seed, so replica outputs are identical regardless of routing, and crc32 routing (gated on `VERL_FULL_DETERMINISM`) cannot be distinguished from least-loaded routing by the assertions. Routing correctness is guaranteed by code, not by the test suite.
- **Hardware**: deterministic ops and batch invariance require specific hardware — see the [vLLM batch invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance/). On unsupported hardware, set `max_num_seqs=1`.
- **Generative GRM only**: this covers the generative VLM reward path. Full E2E determinism (rollout + diffusion sampling) is future work.

## Verify

```bash
pytest tests/reward_loop/test_visual_reward_manager.py -v -s -k deterministic_reward_reproducibility
```

Each run spins up its own `RewardLoopManager` (fresh RM server process(es) via `ray.init`/`ray.shutdown`). Floating-point determinism (`enable_full_determinism`), `VERL_SEED`, and `VLLM_BATCH_INVARIANT=1` are all set by the RM server's `_post_init` (gated on `full_determinism=true` in the rollout config, which the test propagates via `OmegaConf.open_dict`), so no determinism env vars are needed in the Ray runtime_env. Two variants cover single replica (`test_deterministic_reward_reproducibility_single_replica`) and multi replica (`test_deterministic_reward_reproducibility_multi_replica`). Each asserts:

- **Same seed → bitwise-aligned**: two independent runs with the same seed produce bitwise-equal `rm_scores` (`torch.equal`) and identical `genrm_response`.
