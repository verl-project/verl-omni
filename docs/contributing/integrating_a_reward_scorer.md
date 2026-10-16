(integrating_a_reward_scorer)=
# How to Integrate a Reward Scorer

Last updated: 10/15/2026.

This guide covers adding a **reward scorer** to VeRL-Omni: the function or class
that turns one generated sample into a score. Most scorers are a single
free-function file — start with the
[reward rule](../../.agents/rules/reward.md) and the
[add-reward-score](../../.agents/skills/add-reward-score/SKILL.md) skill; this
page is the reference you need **when the scorer holds state across requests**
(a local GPU model, a preference model, a judge session) and therefore becomes a
**class-based (stateful) scorer** under `verl_omni/utils/reward_score/scorers/`.

The layering mirrors the trainer side: a class-based scorer writes **scoring
semantics only** — batching, threading, placement, and lifecycle are
framework-owned.

---

## Free function or class?

| Your scorer… | Shape | Where it lives |
| --- | --- | --- |
| Is a pure rule/function over the sample | Free function | `verl_omni/utils/reward_score/<name>.py` (free `compute_score*`) |
| Calls a remote HTTP service, per sample | Free function | same — network calls belong in the scorer |
| Loads a model once and reuses it across samples | Class | `verl_omni/utils/reward_score/scorers/<name>.py` (class) |
| Needs a batch to be efficient (GPU forward per sample is waste) | Class | same |
| Is engine-served (vLLM router) | Free function + named `engine` model | see [named reward models](../algo/named_reward_models.md) |

The free-function shape is unchanged and remains the recommendation for the
majority of
scorers. The legacy flat modules are frozen in place — never move or rename
scorers. A class-based port lands as a **new** `scorers/<name>.py` that *imports* the
frozen module's math instead of forking it.

## The class-based protocol

One file, one class, plus registration:

```python
# verl_omni/utils/reward_score/scorers/<name>.py
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import register_scorer


@register_scorer("<name>", modality="image")  # modality is a docs/validation tag
class MyScorer(RewardScorer):
    max_batch_size = 16            # executor packs up to this many requests per forward
    supports_cpu_offload = True    # sleep() can park weights on CPU; wake_up() restores

    def __init__(self, model_path=..., ...):   # configuration only — no weights, no device
        ...

    def load(self, device):                    # construct weights, place on device
        ...

    def score_batch(self, batch):              # the one required method
        # batch: list of per-sample kwarg dicts (the free-function compute_score contract)
        # returns: one entry per request, IN INPUT ORDER — a score dict
        #          ({"score": float, ...metrics}) or an Exception isolated to that request
        ...

    # Optional lifecycle hooks. Assigning self.model (the weights module)
    # and self.device in load() inherits all three defaults; override a
    # hook only for extra bookkeeping. The executor — never the scorer —
    # calls them, and it owns gc + accelerator-cache clearing.
    def wake_up(self): ...                     # default: model.to(self.device)
    def sleep(self): ...                       # default: model.to("cpu")
    def close(self): ...                       # default: drop self.model
```

Then add the import to `scorers/__init__.py` (registration is an import side
effect, like `pipelines/__init__.py`), and select the scorer from a named native
model (see below).

**Contract:**

- `score_batch` is len-preserving and order-preserving: `out[i]` corresponds to
  `batch[i]`, whatever internal padding or reordering the forward did.
- Per-request preprocessing failures are returned as `Exception` entries in that
  request's slot; a *raised* exception means the whole microbatch failed.
- The scorer never manages a queue, consumer task, thread, or event loop, and
  never calls lifecycle methods — the executor sequences `load`/`wake_up`/
  `sleep`/`close` and groups requests into microbatches in front of
  `score_batch`. Lifecycle defaults: assigning `self.model` and `self.device`
  in `load` inherits the hook implementations; gc and accelerator-cache clearing
  stay with the executor.
- Frame budgets stay inside `score_batch`: the executor groups *requests*,
  never frames or media streams. Payload provenance follows the
  [named media contract](../advanced/diffusion_media_artifacts.md) — select
  streams by name from `extra_info["media_artifacts"]`, never by position or
  rank; contract violations are fatal, not zero scores.

## Worked example: PickScore

[`scorers/pickscore.py`](../../verl_omni/utils/reward_score/scorers/pickscore.py)
is the reference port. It imports the frozen module's math
(`_PickScoreInferencer`, `_pairwise_pickscore`, `_decoded_pickscore_image`),
declares `max_batch_size = 16` and `supports_cpu_offload = True`, constructs
weights in `load(device)`, and turns per-request image selection into isolated
`Exception` slots inside `score_batch`. The whole file has no queue, no
consumer, no threading state — compare it with the legacy
`pickscore_reward.py` consumer loop to see exactly what the framework absorbed.

## Selecting a class-based scorer from config

Declare a **native** named model whose `executor.model` resolves to the scorer
class, then reference it from a reward term that submits one sample at a time:

```bash
+reward.models.pickscore.backend=native
+reward.models.pickscore.placement.devices=[0]
+reward.models.pickscore.executor.model=pkg://verl_omni.utils.reward_score.scorers.pickscore:PickScoreScorer
"+reward.reward_functions.ps.path=pkg://verl_omni.utils.reward_score.scorers.pickscore"
+reward.reward_functions.ps.name=compute_score_pickscore_batched
+reward.reward_functions.ps.model=pickscore
```

Selection resolution order (name → `pkg.module:Attr` → `file.py:Attr`)
is fail-fast: an unknown name raises with the registered names listed, and a
reference that resolves to a non-`RewardScorer` class is rejected. The
`executor.model` config key currently requires an import-spec form;
bare registered names (`executor.model=pickscore`) are not yet accepted by
the config validation.

## Tests

`tests/utils/reward_score/scorers/test_<name>_on_cpu.py`, following the
[CPU-test conventions](testing_guide.md):

- registry: the registered name resolves; an unknown name raises listing registered names;
- `score_batch` on synthetic tensors with the invariants asserted: input order,
  len-preservation, per-request failure isolation, media-contract violations;
- lifecycle hooks with a fake model (device moves recorded), no GPU required;
- if the port ships a free-function glue entry point, assert the request dict it submits.

The executor-side contracts (admission, sleep/close drain, cancellation
settlement) are already pinned by
[`tests/reward_loop/test_scorer_microbatch_on_cpu.py`](../../tests/reward_loop/test_scorer_microbatch_on_cpu.py)
and
[`test_native_scorer_executor_on_cpu.py`](../../tests/reward_loop/test_native_scorer_executor_on_cpu.py) —
you do not need to retest them per scorer.

## Final Checklist

- [ ] One new file `scorers/<name>.py`; no edits to legacy flat modules — import
      their math instead of copying it.
- [ ] Class subclasses `RewardScorer`, is registered via `@register_scorer`, and
      is imported from `scorers/__init__.py`.
- [ ] `score_batch` is len- and order-preserving; per-request failures are
      `Exception` slots; frame budgets and media selection stay inside it.
- [ ] No queue/consumer/thread code anywhere in the file; lifecycle hooks are
      no-ops or one-liners.
- [ ] CPU test file covers registry + invariants + hooks; existing
      `tests/reward_loop/` and `tests/utils/reward_score/` tests stay green.
- [ ] An example launch script (or an edit to an existing one) selects the
      scorer through `reward.models.<name>` + `reward.reward_functions.<term>`.
