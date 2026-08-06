---
name: add-reward-score
description: Guide for adding a new reward scorer to verl-omni and wiring it into a run. Use when adding a reward function or reward model for image, video, or multimodal RL (flow-GRPO, DanceGRPO, DPO), including preference models and remote HTTP scorers.
---

# Add a Reward Scorer

One new file under `verl_omni/utils/reward_score/`, then select it from the run
config — there is no code registration step.

The signature, return, failure, and caching contracts live in the
[reward rule](../../rules/reward.md); read it first, this skill does not repeat it.
For the machinery around the scorer see `docs/algo/async_reward.md` (reward-loop
workers, resource pools, config reference) and `docs/start/http_scorer.md` (the
request/response protocol for a remote scorer service).

## Step 1 — Copy the closest existing scorer

```bash
grep -rn "^async def compute_score\|^def compute_score" verl_omni/utils/reward_score/*.py
```

`genrm_ocr.py::compute_score_ocr` is the most-copied remote pattern and the one most
`examples/` scripts use; `jpeg_compressibility.py` is the minimal rule-based one;
`hpsv3_reward.py` is the reference for a locally-loaded preference model.
`reward_utils.py` holds the tensor/PIL conversion helpers — use them rather than
re-deriving the conversion.

## Step 2 — Write it

Apache 2026 header, a module docstring crediting upstream if the score is adapted,
and one `compute_score` / `compute_score_<name>` entrypoint matching the rule's
keyword contract. Nothing else is required of the file.

## Step 3 — Select it from a run

The config keys and their semantics (`custom_reward_function` vs
`reward_functions.<key>` with `weight` / `required`, and how extra keys become
kwargs) are in the
[reward rule](../../rules/reward.md#scorers-are-selected-by-config-not-by-a-dispatcher).

Copy a working invocation from `examples/` rather than assembling one by hand —
`examples/flowgrpo_trainer/sd35/run_sd35_medium_drm_lora.sh` is the multi-reward
reference.

## Step 4 — Test

`tests/utils/reward_score/test_<name>_on_cpu.py`, asserting the score is finite and
in range for a synthetic tensor; mock the transport for an `async` scorer
([run-cpu-tests](../run-cpu-tests/SKILL.md)).

```python
def test_score_is_finite_on_cpu():
    out = <name>.compute_score(solution_image=torch.rand(2, 3, 64, 64))
    assert 0.0 <= out["score"]
```

<!--
MAINTAINER GUIDE — keep this procedural; contracts belong in rules/reward.md. Update
Step 1's prose when the set of reference scorers changes materially.
-->
