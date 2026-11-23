---
name: profile
description: "Route a verl-omni profiling request to the right tool, recipe, and processes for the trainer actually being run — legacy diffusion (main_diffusion), diffusion V1 sync / separate_async (main_diffusion_v1), or Omni PPO V1 (main_omni). Use when asked to profile training or rollout, choose between nsys / npu / torch.profiler / torch_memory, decide which role and ranks to profile, or predict where trace artifacts will land."
---

# Profile a run

`docs/perf/profiler.md` owns the config surface (`global_profiler` + per-role
`actor_rollout_ref.{actor,ref,rollout}.profiler`, `reward.reward_model.rollout.profiler`),
the seven recipes, and the lightweight-footprint recipe. Open it — do not work
from a remembered procedure. This skill adds what the guide leaves implicit:
which trainer and mode you are profiling, which process owns the phase, and
which combinations are validated versus unverified.

## Step 0 — Identify the entrypoint and trainer mode first

Read them off the launcher before choosing anything: the `python -m
verl_omni.trainer.*` line and any `trainer.v1.trainer_mode=` override. Recipes
and caveats are mode-specific; transferring one from another mode by memory is
the top source of empty captures and wrong support claims.

| Entrypoint | Selected by | Step-based profiling support |
| --- | --- | --- |
| `main_diffusion` (legacy v0) | diffusion launchers that call it directly | All tools: `nsys` (incl. controller step-scoped capture), `torch`, `torch_memory`, `npu` |
| `main_diffusion_v1` | `trainer.use_v1=True`; mode `trainer.v1.trainer_mode` = `sync` / `separate_async` | `torch` and `npu` validated; `nsys` unverified — the V1 launch path under nsys never worked, fix under review (#839) |
| `main_omni` (PPO V1) | every omni run except offline DPO (`algorithm.sample_source=offline` + `algorithm.trainer_type=direct_preference`) | `npu` documented (recipe 7); other tools come from upstream verl's PPO V1 — check `profiler.md` and the upstream guides before claiming them |

Label every plan item **validated / unverified / unsupported**, and re-check
`profiler.md` before claiming support — the matrix has moved twice recently
(#641, #852).

Mode constraints enforced at config load (fail-fast, not warnings):

- Diffusion V1 rejects controller-side `capture-range=cudaProfilerApi` when
  nsys steps are configured; worker-side capture settings are unaffected.
- Diffusion V1 rejects rollout profiling combined with
  `global_profiler.profile_continuous_steps=True`; actor-only continuous
  windows remain allowed.
- Rollout and reward servers always require `tool_config.torch.discrete=True`
  — the vLLM-Omni servers reject continuous mode.

## Step 1 — Pick the tool by the question

| The question you are answering | Tool | Where it is documented |
| --- | --- | --- |
| Where does the step's wall-clock go? (phase overlap, Python control flow, rank straggler) | `nsys` | `docs/perf/profiler.md` recipes 4, 4a — legacy v0 only |
| Which ops/kernels dominate, and CPU vs CUDA / NPU? | `torch` / `npu` | recipes 1, 2 / recipe 7 |
| What is holding GPU memory / who OOMs? | `torch_memory` | recipe 3 |
| Is config B more compute-efficient than A? | MFU (no profiler) | `docs/perf/diffusion_mfu.md` — read `perf/mfu/actor`, **relative** only |
| Live dashboards across replicas / TransferQueue during a long run? | RL-Insight | `docs/start/rl_insight.md` |

A profiler answers "where is the time/memory in this step". MFU answers "how
efficient is this config vs another on the same setup" — it is a metric, not a
trace, and it over-estimates LoRA (it counts the full DiT forward+backward).

## Step 2 — Profile ONE lightweight step, never a full run

A full FlowGRPO step trace is hundreds of MB and slow to open. Every `examples/`
recipe forwards `"$@"` to the same `diffusion_trainer` config and Hydra resolves
duplicates last-wins, so **append** footprint overrides instead of editing the
script — shrink `rollout.n`, `pipeline.num_inference_steps`, resolution, and
batch (see the guide's lightweight recipe). For diffusion V1 `sync` the
ready-made lightweight launcher is
`examples/flowgrpo_trainer/sd35/profile_sd35_medium_ocr_lora_v1.sh` (runs 5
steps, records only steps 3–4 after warmup).

Always pin these, or profiling is silently skipped:

```bash
trainer.total_training_steps=1 trainer.save_freq=-1 trainer.test_freq=-1 \
trainer.resume_mode=disable global_profiler.steps=[1]
```

The last step force-triggers save/validation when `save_freq`/`test_freq` > 0,
and a leftover checkpoint auto-resumes past the profiled step. For continuous
captures, an interior step is the steady-state sample (the first carries
profiler startup, the last closes the window).

## Step 3 — Enable the profiler on the process that owns the phase

Each phase runs in a different process; enabling the wrong `*.profiler` yields
an empty trace. Trainer steps drive start/stop for every role below.

| Phase | Config key | Artifacts land in |
| --- | --- | --- |
| actor train / backward | `actor_rollout_ref.actor.profiler` | `{save_path}` |
| ref log-prob | `actor_rollout_ref.ref.profiler` | `{save_path}` |
| generation | `actor_rollout_ref.rollout.profiler` | `{save_path}/agent_loop_rollout_replica_{rank}` |
| reward scoring | `reward.reward_model.rollout.profiler` | `{save_path}/reward_model/agent_loop_rollout_replica_{rank}` |

`nsys` is the exception: Ray writes `*.nsys-rep` under
`/tmp/ray/session_latest/logs/nsight/` on each node (fixed by Ray) — only
`torch` / `torch_memory` / `npu` honor `global_profiler.save_path`.

These keys already exist in the composed config, so override with plain
`key=value` — a `+key=value` append fails with "An item is already at ...".

`*.rollout.profiler.ranks` are global ranks within the rollout workers, not
replica IDs or device IDs: for replica world size `W = TP * DP * PP`, rank `r`
selects replica `r // W`. In diffusion V1 `separate_async`, standalone rollout
ranks start at `H * W` after the hybrid replicas. Concurrent NPU collectors
must sit on physically disjoint devices — check actual placement, not just
rank numbers.

## Routing examples

- **Diffusion V1 `sync`, PyTorch, actor + rollout** — recipes 1 + 5 (the
  options apply to V1 launchers unchanged), or the SD3.5 lightweight launcher
  above. Rollout profiling forces `profile_continuous_steps=False`. Actor and
  rollout are separate worker groups; check **both** artifact sets for
  CPU/CUDA events — a clean training exit alone does not prove a trace was
  collected.
- **Diffusion V1 `separate_async`, NPU** — recipe 7 with the same overrides;
  the standalone rollout replicas are separate processes on dedicated GPUs, so
  pick actor and rollout ranks on disjoint devices. `global_profiler.steps`
  are trainer time windows, not the policy version of replay-buffered samples
  — correlate by ranks and trace content only (fully-async profiling is an
  open RFC, #712).
- **Omni PPO V1, NPU, colocated actor + rollout** — recipe 7 with an omni
  launcher (e.g.
  `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_avqa_v1.sh`).
  Collecting both roles in one colocated run needs a verl revision containing
  verl#7722; the repo's current verl pin predates it (recipe 7's
  prerequisites have the details).
- **nsys controller / worker timelines** — recipes 4 and 4a are legacy-v0
  paths. Do not route V1 or omni requests to them.

## Capture plan — the handoff

Before launching, state the plan in one block: **question → tool →
entrypoint/mode → role config keys + ranks → expected artifact paths →
validated/unverified status and limitations**. Resolve any unknown field
against the launcher or `profiler.md` first — an unknown field at planning
time becomes an empty trace at collection time.

## Gotchas (each has bitten a real run)

- **`nsys` output path**: fixed by Ray (see Step 3), not `save_path`; on V1
  additionally unverified end-to-end (#839).
- **Report hygiene**: reports may embed env vars such as `HF_TOKEN` unless
  `discard-environment` is set — scrub before sharing.
- **Do not hand-roll** `torch.profiler` / timers inside adapter or pipeline
  code. Workers are wrapped with `verl.utils.profiler.DistProfiler` and
  driven around each profiled step; an ad-hoc timer is fine for a throwaway
  local check but must never reach a PR.

## Further reading

- `docs/perf/profiler.md` — authoritative config surface, recipes, support matrix.
- `docs/perf/diffusion_mfu.md` — MFU reporting and adding an estimator.
- `docs/start/rl_insight.md` — online observability dashboards.

<!--
MAINTAINER GUIDE — this file must stay a mode-aware router. Recheck the
support matrix whenever entrypoints, trainer modes, or profiler validation
status change (e.g. #839 for V1 nsys, #712 for async profiling), and keep the
recipes, commands, and config reference in docs/perf/profiler.md — link, do
not copy.
-->
