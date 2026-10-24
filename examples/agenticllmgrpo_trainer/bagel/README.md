# BAGEL Co-RL (joint UND + GEN) training

Last updated: 10/12/2026

Multi-turn agentic co-RL on one BAGEL-7B-MoT actor. The UND text path is a Hermes
tool-calling agent loop and the GEN path (`*_moe_gen`) is its own `generate_image` tool;
both are trained in one composite optimizer step (UND token GRPO + GEN FlowGRPO) behind
the single `python3 -m verl_omni.trainer.main_omni` entrypoint. See the
[design RFC](https://github.com/verl-project/verl-omni/issues/453).

Rollout is sequential and multi-turn. Per episode the actor samples UND turns and, when a
turn calls `generate_image`, fans out the GEN denoise group; the optimizer is one
composite step after the sibling gather.

| Symbol | Meaning |
| --- | --- |
| `J` | UND policy turns in one episode |
| `K` | `generate_image` calls in the same episode |
| `S` | FlowGRPO seeds per `generate_image` call |
| `N` | Sibling episodes per dataset task for token GRPO (`rollout.n`) |

Episodes may be paired (`J == K`), mixed (`J > K >= 1`, reflection-only turns) or gen-off
(`K == 0`, UND-only).

Fail-closed rules:

- UND must be the published BAGEL checkpoint emitting Hermes `<tool_call>`. If dual-role
  serving cannot be proven, stop; do not fall back to a Qwen actor or image sidecar.
- Do not edit the Mode (2a) recipe
  `examples/agenticllmgrpo_trainer/agent_llm/run_agenticrpco_grpo_lora.sh`.
- No mid-episode weight sync, and no update before the N-sibling gather completes.

## Prerequisites

- Install VeRL-Omni (see the [installation guide](../../../docs/start/install.md)).
- 4 GPUs or more. Run commands from the repository root.
- Download the checkpoint:

  ```bash
  huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir ~/models/ByteDance-Seed/BAGEL-7B-MoT
  ```

## Prepare the data

The recipe builds the UniCoT train/val parquet automatically; `REBUILD_UNICOT=1` forces a
rebuild. The parquet must carry `extra_info.reference_image_path` (UniCoT references for
the reward model); re-stamp with `stamp_unicot_reference_paths.py` when the reference set
changes. RPCO reward weights are build-time flags: `--w_reflect/--w_plan/--w_format/--w_tool/--w_result`.

## Tests

```bash
python -m pytest tests/agent_loop tests/trainer/omni tests/utils -q
bash tests/gpu_smoke/run_gpu_smoke_core.sh
```

## Dual-role serving spike

The trainer refuses to start without `agent.und_ar_serving_ready=True`. Prove the UND AR
replica decodes Hermes tool calls before spending a long run:

```bash
# Terminal A: standalone UND AR server (bagel_think deploy, Hermes-capable)
bash examples/agenticllmgrpo_trainer/bagel/run_bagel_und_ar_serve.sh

# Terminal B: Hermes tool-call probe against the UND replica
BAGEL_UND_URL=http://127.0.0.1:8094 \
  python3 examples/agenticllmgrpo_trainer/bagel/spike_und_hermes.py \
    --model-path "$BAGEL_MODEL_PATH"
```

A pass means UND decodes a `generate_image` tool call and the GEN replica stashes a
FlowGRPO trajectory (latents, timesteps and log-probs all present). If the base model's
tool-calling is too weak, the fix is an SFT pass, not a loosened parser.

## Run training

```bash
# 4-GPU default
bash examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh

# 1-GPU sanity (RM off by default — actor + Omni + RM cannot fit one card)
CUDA_VISIBLE_DEVICES=3 bash examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh
```

The recipe runs `${_REPO_ROOT}/.venv/bin/python3` (never a bare `python3`, which can resolve
to a different interpreter) and resolves `verl` from the venv's install. It also *offers*
the sibling `../verl` checkout first — the layout that predates the venv's flat
`tokenizer.py` — but only after probing that the candidate can satisfy this tree's imports.
A stale sibling is worse than none: on `PYTHONPATH` it silently shadows the venv's `verl`
and the run dies before touching a GPU with, e.g., `ImportError: cannot import name
'build_rollout_dist_profiler' from 'verl.utils.profiler'`. Point `_LOCAL_VERL` at a specific
checkout to force one, or at a path with no `verl/` in it to skip the probe.

Ladder: run 1-GPU first (loop + composite step, RM off), then 2-GPU, then the 4-GPU
acceptance run with `ENABLE_RM=1` (colocated Qwen RM, TP-sharded across the pool).
`BAGEL_UND_AR_SERVING_READY=1` is the recipe default — set `0` to re-hear the fail-closed
gate after any serving change. `ROLLOUT_TP` and the GEN deploy yaml are selected together
from the pool size and read back before launch, so they cannot drift apart.

## Default configuration

| Env | Default | Effect |
| --- | --- | --- |
| `BAGEL_MODEL_PATH` | `$HOME/models/ByteDance-Seed/BAGEL-7B-MoT` | Published BAGEL-7B-MoT checkpoint (must be untied `lm_head`). |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | All visible cards go to the actor+GEN hybrid pool; UND AR colocates on the same placement group. |
| `UND_N_GPUS` | `2` (>2-GPU) | Width of the actor-PG slice the UND AR replica gets. Must cover every logical device its stages ask for (Thinker `0`, DiT `1`) and divide the actor pool. |
| `ENABLE_RM` | `0` (1/2-GPU), `1` (>=4-GPU) | Colocated Qwen RM. `1` enables mid-loop scoring and image-grounded episode rewards; without it every step is `skip_gen=True` and the GEN lane is frozen. |
| `REWARD_MODEL` | UnifiedReward-2.0-qwen3vl-2b | Reward model for `reward.reward_model.model_path`. |
| `REWARD_TP` / `ROLLOUT_TP` | pool size / `2` for a >=4-card pool, else `1` | Tensor-parallel for RM / GEN rollout; the pool must divide both. `ROLLOUT_TP` shards the GEN checkpoint, and each value is paired with a matching deploy yaml (`bagel_corl_deploy.yaml` for TP=1, `bagel_corl_deploy_tp2.yaml` for TP=2) whose stage `devices` length must equal it. |
| `ROLLOUT_GPU_MEM_UTIL` / `UND_GPU_MEM_UTIL` | `0.25` / `0.40` (>2-GPU) | GEN / UND AR Thinker budgets. Only `ROLLOUT_TP` shards the GEN weights, and the AR budget sizes the Thinker KV cache above its init floor, so neither is a placement knob. |
| `BAGEL_UND_AR_SERVING_READY` | `1` | Fail-closed gate; set `0` to re-prove the spike. |
| `S` / `N` / `LORA_RANK` / `LORA_ALPHA` / `GEN_STEPS` / `GEN_HW` / `TRAIN_BSZ` | see script | Seeds per call / siblings / LoRA shape / denoise steps / generation size / batch. On the >=4-card branch these are `S=1`, `N=4`, `TRAIN_BSZ=4` (with `PPO_MINI=TRAIN_BSZ`, `PPO_MICRO=1`), i.e. **16 episodes per optimizer step** in one mini-batch: siblings sharpen the group baseline, `TRAIN_BSZ` widens the task mix. `S=1` with `N<2` is rejected — the group mean then equals its only member and every GEN advantage is zero. |
| `ENABLE_PEC` | `1` | Turns on the GEN conditioning cache and its routing affinity as one pair (see below). |
| `AGENT_WORKERS` | `8` | Agent-loop workers; also sizes `PEC_SIZE`. |
| `MAX_PROMPT_LEN` | `2048` (>=4-GPU), `1024` (2-GPU), `512` (1-GPU) | Pins `data.max_prompt_length`/`max_response_length` **and** `rollout.prompt_length`/`response_length`; the pairs must match. |
| `MAX_UND_TURNS` | `8` | Episode UND turn cap, in practice bounded by `2 x MAX_PROMPT_LEN`. |
| `REBUILD_UNICOT` | `1` | Rebuild train/val parquet before launch. |
| `ROLLOUT_TEMPERATURE` / `ROLLOUT_TOP_P` / `ROLLOUT_TOP_K` / `ROLLOUT_REPETITION_PENALTY` | `0.7` / `0.9` / `50` / `1.05` | UND AR sampling, mirroring `bagel_corl_deploy_ar.yaml`. The agent loop sends all four explicitly, so leaving them unset overrides the deploy defaults. |
| `NORM_ADV_BY_STD` | `False` | Dr.GRPO: the advantage is the raw centred reward, so the update scales with the signal instead of the group std (at `N=2`, std division makes every gap a fixed-amplitude push). Read on both lanes. |
| `LR_WARMUP_RATIO` / `LR_SCHEDULER` / `MIN_LR_RATIO` | `0.05` / `cosine` / `0.1` | Actor LR schedule: 5% warmup, then cosine decay to 10% of the `1e-4` peak, so the late steps move ~10x less. |
| `VAL_MAX_SAMPLES` | `32` | Caps the V1 val pass (`val_max_samples` defaults to `-1`, i.e. the whole val set). A binary dim is `1/VAL_MAX_SAMPLES` wide, so 8 samples quantize `val-core` in 0.125 steps and a 0.03 improvement is invisible; 32 resolves it to 0.031. |
| `AGENTIC_VAL_VIZ` | `1` | Fixed validation holdout. Each val step also renders samples `9001`/`9002` (cafe poster, `reflect`/`plan`) and `9003`/`9004` (Chinese poster, `reflect`/`plan`) with the deterministic 50-step ODE val kernel, under `rollout_images/val/step_XXXXXX/sample_900N.01/` — the same four artifacts the Mode (2a) RPCO recipe dumps. They are artifacts only: the holdout writes **no** TransferQueue row, so it cannot displace or shorten the val batch. `0` opts out. |
| `RUN_TS` / `RUN_NAME` / `RUN_DIR` | `date +%Y%m%d_%H%M%S` / `bagel_corl_$RUN_TS` / `outputs/$RUN_NAME` | Hydra run dir (`hydra.run.dir`), holding `.hydra/` and `main_omni.log`. Hydra refuses an existing dir, so pass the original stamp or point `RUN_DIR` at it. |
| `DEFAULT_LOCAL_DIR` | unset | Checkpoint dir; unset shares `checkpoints/${project_name}/${experiment_name}` across runs. |
| `BAGEL_CORL_TRACE_DUMP` | `1` | Per-episode trajectory dump under `<e2e_root>/<experiment_name>/rollout_trajectories/step_XXXXXX/`. `0` opts out. |

Training-side knobs (in the launch command): `actor.optim.lr` (UND),
`+actor_rollout_ref.model.lr_gen=3e-5` (GEN group), `loss_weight_und/gen`, `loss_mode`
(`flow_grpo`, or `grpo_guard` for GRPO-Guard ratio normalization) and `gen_regularizer`
(`latent_kl`). CFG is off in training (`cfg_text_scale=1.0`); re-enable for eval only via
val kwargs.

## Reward dimensions and divergence

The episode scalar is the weighted composite from `agentic_multidim_reward`:
`reflect`, `plan`, `format`, `tool`, `result`, plus the three dims that carry the judge's
magnitude and the episode's closure. All three publish as metrics, so a flat value is
visible in the validation row rather than inferred from the scalar:

- `reward_image` (weight 3.0) — the final trusted C/A verdict plus the capped rewrite gain,
  `min(1, max(0, final - first))`. It is the only dim with a continuous, task-specific
  magnitude, so it is weighted above the protocol dims rather than averaged with them.
- `reward_protocol` (1.0) — "nothing silently failed": every call block parsed, no
  `agentic_judge ok=0`, no hard block. Deliberately disjoint from `R_format`, which only
  checks that the trajectory has the right *shape*.
- `reward_done` (1.0) — the episode closed on a terminal verdict **and** a stop cue. Two
  Bagel-dialect details are load-bearing: the harness appends its stop cue to the verdict's
  *own line* (`... good_enough=YES. <prose> agentic_stop_decision_required=1`), and the
  checkpoint answers the forced reflection with `{"name": "Done"}`, not prose `Done.`.

`bagel_corl/divergence_warning` (plus a WARNING log, once per excursion) is raised by
`OmniBagelCoRLTrainerSync._divergence_tripwire` when `rollout_corr/kl` exceeds 20,
`actor/grad_norm` exceeds 100, or a reading is `nan`. Both limits are an order of magnitude
above any healthy run measured to date and below the readings that made a collapsing run
unrecoverable.

This recipe runs the tripwire **armed** — `run_agentic_bagel_rpco_lora.sh` exports
`BAGEL_CORL_DIVERGENCE_ABORT=1` — so a diverged step fails the run instead of warning. That
is deliberate here and only here: the recipe sets no KL anchor (`use_kl_loss` stays false)
and no entropy bonus (`entropy_coeff` stays 0), so a step that leaves the region cannot come
back on its own, and every remaining step would spend rollout + reward budget on a policy
that never gets used. `bagel_corl_prfix_t07_20261016_023749_a1` is the measurement behind
that default: `rollout_corr/kl` printed `41.8` against the limit of `20` at step 84 and the
run continued for another 115 steps producing nothing. Aborting forfeits no state *only*
because this recipe saves no checkpoint for the run — if checkpointing is ever enabled,
re-read that default before trusting it.

### Reading the training reward

`critic/score/mean` is not a critic. With `adv_estimator=grpo` there is no value network —
which is why `critic/advantages/mean == critic/returns/mean` to the digit — and `verl`'s
`compute_data_metrics` defines the key as `token_level_scores.sum(-1).mean()` over the
non-aborted episodes of that rollout batch. It is a **per-step sample mean over ~16–25
heterogeneous episodes**, and on `bagel_corl_prfix_t07_20261016_023749_a1` it was dominated
by rollout failure rather than policy quality: it correlated `-0.71` with `episode/K_zero`,
carried a std of `0.088` against a total drift of `-0.014` over 76 updates, and its 20-update
moving average never left `0.63–0.71` — while `val-core/unicot_reflection/reward/mean@1` rose
`0.729 → 0.829` over the same span, led by `reward_image` (`+0.169`, weight 3.0) and
`reward_done` (`+0.143`). Do not read a flat training scalar as a stalled policy.

Every training update therefore also publishes:

- `bagel_corl/reward_dim/<dim>/{mean,min,max}` — the same dims the validation pass logs as
  `val-aux/...`, so a step's scalar swing is attributable to a named term.
- `bagel_corl/reward_valid/{mean,count}` and `bagel_corl/reward_no_image/{mean,count}` — the
  episode reward split by `used_image_credit`, separating "quality when the episode produced
  a usable image" from "how often it did". `count` rides along because a mean over a
  shrinking subset is not comparable on its own; watch it before reading a jump in
  `reward_valid/mean` as progress.

Both are read from the **trained** reward, in the trainer's own order: `rm_scores` (the column
`trainer_base` copies into `token_level_scores`, reduced with `.sum(-1)` because the reward
rides on one trainable token), then `token_level_scores`, then — only as a fallback —
`und_reward`. That last one is a trap: `bagel_corl_tq.episode_from_agent_extra` stamps
`und_reward` from `AgentLoopOutput.reward_score` (the reward manager's composite over
`reward_extra_info`), while the trained signal is recomputed as `episode_und_reward` →
`image_credit_reward` over the GEN seeds. They are different numbers. Reading `und_reward` on
`bagel_corl_prfix_t07_20261016_102501_a1` put `reward_valid/mean` 0.19 below
`critic/score/mean` at every step, including the steps where all 16 episodes had image credit.

A dim absent from a batch publishes nothing rather than `0`, so a missing scorer stays
distinguishable from a scorer that returned zero. `reward_plan` is `0` on these lanes by
design: `agentic_multidim_reward` only fires it when `task_type == "plan"`.

## GEN conditioning reuse

`generate_image` submits `S` seeds that share one conditioning, and each seed otherwise
re-encodes it. The conditioning cache amortizes that so one encode serves the whole group.
All three knobs must be passed to Hydra as `+key=value` (append), because the fields live
on `DiffusionRolloutConfig` and are only grafted on at run time.

1. **The knobs are a pair.** `enable_prompt_embed_cache` turns the cache on and
   `enable_prompt_embed_cache_routing_affinity` pins one S-group to one replica; without
   the pin, seeds 2..S scatter across workers holding no entry. Both come from
   `ENABLE_PEC=1`.
2. **`prompt_embed_cache_size` covers the live working set**, hence the
   `S * N * AGENT_WORKERS` default. Too small and a group evicts its own entry mid-group.
3. **The identity is derived, never hand-set.** `cond_uid` is a SHA-256 over the
   conditioning token ids the GEN stage encodes; the UND row carries the episode's
   distinct `gen_cond_uids`.

Per-step metrics are read from the engine's own cache counters, and are omitted entirely
when the engine exposes none:

| Metric | Definition | Target |
| --- | --- | --- |
| `gen/cond_recompute_ratio` | conditioning prefills / `generate_image` calls | `1/S` |
| `gen/prompt_embed_cache_hit_rate` | hits / (hits + misses); bypasses excluded | `1 - 1/S` |
| `gen/cond_amortization` | `1 / cond_recompute_ratio` | `S` |

The engine's conditioning cache is invalidated at every weight publish (in addition to the
LoRA request cache), because an entry is a function of `(tokens, weights)` and a new merged
weight version is published each composite step. Serving a stale entry would condition on a
different policy than the one being trained.
