---
paths:
  - verl_omni/utils/reward_score/**
  - verl_omni/reward_loop/**
---

# Reward Rules

- `verl_omni/utils/reward_score/` — the **scorers**, one reward per file (9 plus
  `reward_utils.py`). Scorers own their own I/O, including HTTP and model loading.
- `verl_omni/reward_loop/reward_manager/{visual,multi}.py` — the **managers**, which
  resolve the configured scorer, feed it one sample at a time, and turn its return
  value into `rm_scores` + metrics.

## Scorers are selected by config, not by a dispatcher

The `default_compute_score_image` dispatcher in `reward_score/__init__.py` is a
**fallback with a single branch** (`jpeg_compressibility`). It runs only when
`reward.custom_reward_function.path` is unset. Adding a branch there is *not* how a
reward gets wired up — all nine other scorers are selected from the run config:

```bash
# single reward
reward.custom_reward_function.path=verl_omni/utils/reward_score/hpsv3_reward.py
reward.custom_reward_function.name=compute_score_hpsv3

# multiple rewards, weighted
reward.reward_manager.name=MultiVisualRewardManager
"+reward.reward_functions.drm.path=pkg://verl_omni.utils.reward_score.latent_http_scorer_client"
+reward.reward_functions.drm.name=compute_score
+reward.reward_functions.drm.weight=1.0
+reward.reward_functions.drm.required=true
+reward.reward_functions.drm.noise_level=0.4      # forwarded to the scorer as a kwarg
```

`path` accepts a file path or a `pkg://` module. Under `reward_functions.<key>`,
`{path, name, weight, required}` are reserved and **every other key is forwarded to
the scorer as a keyword argument** — that is how `server_url` and `noise_level`
arrive. `data.reward_fn_key` (default `data_source`) names the dataset column that
becomes the `data_source` argument.

The surrounding config — reward-model resource pools, worker counts, how the reward
loop plugs into the agent loop — is documented in `docs/algo/async_reward.md`, and
the remote scorer protocol in `docs/start/http_scorer.md`.

## Scorer signature

Managers call scorers **by keyword**, so parameter names are the contract:

```python
async def compute_score_<name>(
    data_source: str,
    solution_image,       # torch.Tensor, (C,H,W) or (N,C,H,W), float in [0, 1]
    ground_truth: str,    # prompt / reference text
    extra_info: dict,     # manager-injected: num_turns, rollout_reward_scores, ...
    **kwargs,             # forwarded config keys: server_url, model_name, ...
) -> dict:                # {"score": float, "<name>_raw": ...}
```

- **`async` is optional.** Managers detect it with `inspect.iscoroutinefunction` and
  push sync scorers to an executor. Write whichever fits.
- **Return a `dict` with `"score"`.** Extra keys become metrics verbatim
  (`MultiVisualRewardManager` namespaces them `reward/<key>/<field>`) — this is why
  scorers return `{"score": ..., "hpsv3_raw": ...}`. A bare float works but lands in
  `reward_extra_info["acc"]`. Never return a tensor or a tuple.
- Naming follows the entrypoint: `compute_score` for generic ones,
  `compute_score_<name>` when the file is model-specific (`compute_score_pickscore`,
  `compute_score_hpsv3`, `compute_score_ocr`, `compute_score_unified_reward`).

## Input contract

`solution_image` is a float tensor in `[0, 1]`; scale by 255 inside the scorer.
Don't assume more than that:

- Channels-last is accepted by some scorers (`hpsv3`, `pickscore` sniff
  `shape[-1] in (1, 3)`).
- Video arrives 5-D and is frame-subsampled via `extra_info["frame_interval"]`.
- `latent_http_scorer_client` receives a **latent**, not an image — the parameter
  name is a convention, not a guarantee.
- `extra_info` is mutated by the manager before the scorer sees it.

## Failure and heavy state

- **Raising is allowed.** `MultiVisualRewardManager` catches per-sub-reward and
  substitutes `0.0` unless that entry sets `required: true`. `VisualRewardManager`
  has no try/except, so a raise there propagates — raise only when the run genuinely
  cannot continue (`latent_http_scorer_client` does this after exhausting retries).
- **Network calls belong in the scorer**, not the manager — every `aiohttp` call in
  the repo is scorer-side. Managers only forward `reward_router_address` /
  `model_name` down.
- **Cache models in module-level globals**, not per call: `pickscore` uses a global
  plus an `asyncio.Queue` with one consumer; `hpsv3` uses a cached inferencer behind
  a threading lock. Heavy top-level imports are the existing norm in those files.
- `MultiVisualRewardManager` shares **one** `reward_router_address` across all
  sub-rewards — several distinct remote reward models are unsupported today.

## Tests

Scorer tests live in `tests/utils/reward_score/test_<module>_on_cpu.py`, manager
tests in `tests/reward_loop/`. Three of nine scorers are covered; a rule-based
scorer should be deterministic and CPU-testable with a synthetic tensor
([testing.md](testing.md)).

Adding a scorer: [add-reward-score](../skills/add-reward-score/SKILL.md).
