(integrating_a_reward_scorer)=
# How to Integrate a Reward Scorer

Last updated: 11/24/2026.

This guide covers adding a **reward scorer** to VeRL-Omni: the function or class
that turns one generated sample into a score. Most scorers are a single
free-function file — model-backed ones follow
[Named Reward Models](../algo/named_reward_models.md)
for selection and configuration. This page is the reference you need
**when the scorer holds state across requests** (a local GPU model, a preference
model, a judge session) and therefore becomes a **class-based (stateful)
scorer** under `verl_omni/utils/reward_score/scorers/`.

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


@register_scorer("<name>")
class MyScorer(RewardScorer):
    max_batch_size = 16            # executor packs up to this many requests per forward
    supports_cpu_offload = True    # sleep() can park weights on CPU; wake_up() restores

    def __init__(self, model_path=None):   # configuration only — no weights, no device
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
    # Inherited defaults: wake_up() moves model to self.device; sleep() moves
    # model to CPU; close() drops model. Keep those methods unless extra work is needed.
```

Then add the import to `scorers/__init__.py` (registration is an import side
effect, like `pipelines/__init__.py`), and select the scorer from a named native
model (see below).

Registered names become visible after their importing plugin initializes
successfully. A failed import discards its registrations, including aliases of
classes imported from other modules; successfully imported dependencies remain
registered.

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

(native-scorer-runtime-statistics)=
## Inspecting native scorer runtime statistics

Native scorer runtime counters can be read explicitly with
`NativeRewardExecutor.get_runtime_stats()` or `ScorerMicrobatcher.get_runtime_stats()`.
On a reward worker, `get_native_scorer_stats()` returns model-name to snapshot;
on `OmniRewardLoopManager`, it returns group-name to an ordered list of worker
snapshots. A legacy `infer()` executor returns `{}`. The query does not wake a
scorer or add values to per-sample `reward_extra_info`.
Before a scorer's first wake, its class has not been resolved: its executor
returns `{}`, and its worker omits that model's key. Querying stats does not load
the model just to identify its type.

Each snapshot is a detached dictionary of cumulative values for one executor
instance. Counts include enqueued, accepted, withdrawn, discarded, and
capacity-waiting requests; executed batches and batch requests; failed batches
and accepted requests settled with exceptions; and peak queue and batch sizes.
`capacity_wait_seconds` covers time waiting for queue capacity, including
cancelled or rejected calls. `queue_wait_seconds` covers only requests accepted
by the consumer. `forward_seconds` measures host elapsed time around the
scorer call, including errors and cancellation; it is not GPU kernel time.
Counters survive sleep and wake within an executor, but do not persist across
worker restarts. Counts describe actual scorer work, including padding requests
from static replica dispatch; they are not counts of distinct evaluation rows.
Worker snapshots are kept separate because their clocks and lifetimes need not
align.

For a synchronous, non-streaming scoring phase, query the existing manager before
and after `compute_rm_score` returns. Do not recreate workers between snapshots
or overlap this phase with other scoring calls:

```python
before = manager.get_native_scorer_stats()
scored = manager.compute_rm_score(batch)  # batch is a DataProto
after = manager.get_native_scorer_stats()

for group, workers in after.items():
    previous_workers = before.get(group, [])
    for replica, models in enumerate(workers):
        previous_models = previous_workers[replica] if replica < len(previous_workers) else {}
        for model_name, counters in models.items():
            previous = previous_models.get(model_name, {})
            delta = {
                key: counters[key] - previous.get(key, 0)
                for key in ("batch_requests", "batches", "queue_wait_seconds", "forward_seconds")
            }
            mean_batch_size = delta["batch_requests"] / delta["batches"] if delta["batches"] else 0
            print(group, replica, model_name, mean_batch_size, delta)
```

An absent pre-wake model snapshot starts at zero. Only subtract snapshots from
the same worker and executor instance; a worker restart invalidates that phase
comparison. Peak fields are lifetime maxima, not additive counters. All duration
fields are seconds: queue wait is summed over accepted requests, whereas forward
time is counted once per executed batch, including failed batches. Their sums
across replicas are not phase wall time. The example's mean batch size includes
padding requests and does not measure distinct samples or GPU utilization.
The manager query is a blocking Ray fan-out, not an atomic cross-worker snapshot;
use it at an explicit phase boundary, outside the hot scoring path.

## Recovery from a non-returning forward

The hosting worker's `sleep()` and `close()` stop admission and drain accepted
forwards before offloading or disposing weights. Cancellation of a lifecycle
caller does not interrupt a synchronous forward thread. Scorers must put finite
timeouts on backend I/O they own; cancelling a Python task cannot safely stop
such a thread.

If a forward exceeds the deployment's request or shutdown deadline, the job
supervisor/operator must terminate the hosting worker process and confirm its
death before recreating the worker and model. For a Ray actor, use
`ray.kill(worker, no_restart=True)` on the affected actor, then confirm actor
termination; for a colocated worker, terminate the containing process/job. Fail
its outstanding requests and restart the job. Recreating only the actor does
not replace handles held by the reward worker groups or the native owner.
Do not retry `sleep()`/`close()` in that process or offload weights while its
forward may still run. The executor does not automatically kill workers or impose
a universal scoring deadline: this escalation belongs to the deployment owner.

An async scorer's cancelled dependency is instead a failure of that microbatch;
it does not cancel the consumer or prevent subsequent requests from scoring.

## Worked example: PickScore

[`scorers/pickscore.py`](../../verl_omni/utils/reward_score/scorers/pickscore.py)
is the reference port. It imports the frozen module's math
(`_PickScoreInferencer`, `_pairwise_pickscore`, and the shared
`decoded_reward_image` selector),
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

`executor.model` accepts an import spec: `pkg.module:Class`,
`pkg://pkg.module:Class` or `/path/to/reward.py:Class`. Bare registered names
(`executor.model=pickscore`) are rejected by configuration validation; registry
name lookup is a separate API. A resolved `RewardScorer` subclass uses batch
scoring and scorer lifecycle hooks. Other native model classes or factories
retain the legacy `infer()` adapter; they are not validated as `RewardScorer`
subclasses. The legacy adapter requires an `infer()` method when invoked.
Class-based `RewardScorer` models require flat `placement.devices` (complete
replicas). `placement.replica_groups` supports only the legacy
`PickScoreNativeModel` adapter, not class-based scorers.
The named model's top-level `model_path`, when set,
is forwarded to the scorer constructor as a `model_path` kwarg —
`executor.kwargs.model_path` takes precedence; a scorer whose constructor
takes no `model_path` fails construction with a clear `TypeError`.

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
      During the migration window, example updates land with the post-window
      cleanup (#723) together with the legacy-path scripts.
