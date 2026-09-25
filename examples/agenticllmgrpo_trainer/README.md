# Agentic LLM RPCO trainer

Last updated: 09/11/2026

This recipe runs RPCO stage-3 GRPO on the UniCoT Self-Reflection and Breakdown
datasets. The trainable agent LLM calls two frozen services:

1. `generate_image` produces a candidate image.
2. `judge_image` returns correctness, aesthetics, findings, and a
   `good_enough` decision.
3. The agent stops or rewrites the image prompt.

RPCO combines reflection, format, tool, result, and improve rewards from
`verl_omni.utils.reward_score.agentic_multidim_reward` (PR #412). Trajectories
come from `image_gen_tool_agent` + `OmniAgentLoopManager` (PR #409). Parquet
rows are built by `build_unicot_agentic_rl` (PR #411). Reflect and plan rows run
the same loop and the same reward formula and differ only in their system
prompt, so their reward curves are comparable.

## Start the frozen tools

Use GPUs that are not assigned to the trainer:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash examples/agenticllmgrpo_trainer/agent_llm/run_image_gen_tool_server.sh
```

```bash
CUDA_VISIBLE_DEVICES=1 \
  bash examples/agenticllmgrpo_trainer/agent_llm/run_judge_image_tool_server.sh
```

Defaults listen on `127.0.0.1:8092` (image) and `127.0.0.1:8093` (judge).

## Run stage 3

The launcher rebuilds UniCoT parquet by default (`REBUILD_UNICOT=1`), then
starts GRPO via `verl.trainer.main_ppo` (Qwen3-VL is not an OmniModelBase
architecture; `agentic_image_gen` is injected with Hydra `+` overrides):

```bash
CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 TOTAL_STEPS=200 \
  bash examples/agenticllmgrpo_trainer/agent_llm/run_agentic_rpco.sh
```

By default the builder uses Hugging Face cache directories for:

- `Fr0zencr4nE/UniCoT-Self-Reflection-6K`
- `Fr0zencr4nE/UniCoT-Breakdown-3K`

Set `UNICOT_REFLECTION_DIR` and `UNICOT_BREAKDOWN_DIR` to explicit snapshot
directories when needed. The full parsed dataset is used unless
`UNICOT_TRAIN_SIZE` / `UNICOT_VAL_SIZE` is set for a smoke run (capped sizes
fail closed if the mix cannot be met).

Useful controls:

- `RPCO_INIT_CKPT`: initialize from a stage-1 agent checkpoint.
- `RPCO_W_REFLECT`, `RPCO_W_FORMAT`, `RPCO_W_TOOL`,
  `RPCO_W_RESULT`, `RPCO_W_IMPROVE`: per-dimension weights baked into parquet
  ground truth (`RPCO_W_TOOL_CALL` is accepted as an alias for `RPCO_W_TOOL`).
  `RPCO_W_IMPROVE` scores the judge-outcome lift across a rewrite chain: the mean
  `(correctness, aesthetics)` gain from the first `judge_image` to the preferred one.
  A rewrite chain that leaves the judge where it started earns nothing, however much
  its text changed. Set `RPCO_W_IMPROVE=0` to score without it.
  (Predecessor `RPCO_W_NOVELTY` and the baked `w_novelty` key still resolve as an
  alias, so parquet built before the swap keeps working without a rebuild.)
  A stale `RPCO_W_PLAN` / `w_plan` key from an older parquet is inert: there is no
  `plan` dimension any more, so it cannot shift the total.
- `AGENTIC_VLLM_OMNI_URL` / `AGENTIC_VLLM_URL`: sidecar endpoints, forwarded
  to Hydra `agentic_image_gen.vllm_omni_url` / `vllm_url`.
- `AGENTIC_E2E_ROOT`: dump root for traj / images (`agentic_image_gen.e2e_root`).
- `TEST_FREQ` and `VAL_BEFORE_TRAIN`: periodic validation.
- `VAL_ROLLOUT_N`: greedy validation rollout count; defaults to 1.
- `REBUILD_UNICOT=0`: reuse existing train/val parquet.
- `MLFLOW_TRACE_BACKEND`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACE_EXPERIMENT`,
  `TRACE_MAX_SAMPLES_PER_STEP_PER_WORKER`: MLflow rollout traces, on by default
  (see [Rollout tracing](#rollout-tracing) below).

Reward wiring (required by PR #412):

- `reward.reward_manager.name=naive` (text trajectory; not VisualRewardManager)
- `reward.custom_reward_function` → `agentic_multidim_reward.compute_score`

## Rollout tracing

Rollout traces go to MLflow. The run script already sets
`actor_rollout_ref.rollout.trace.backend=mlflow` plus `token2text=true`, appends
`mlflow` to `trainer.logger` (so scalar metrics land in both WandB and MLflow),
and builds the sqlite store under
`outputs/e2e/mlflow/<experiment_name>.db` before the workers start — the
first-writer race the [trace docs](https://verl.readthedocs.io/en/latest/advance/rollout_trace.html)
warn about.

```bash
# Four slashes after ``sqlite:`` — ``sqlite:///`` + the absolute path. With only
# three, SQLAlchemy treats the path as *relative* to your cwd and quietly opens a
# brand-new empty database there, so the UI shows an empty "Default" experiment.
mlflow ui --backend-store-uri "sqlite:///$PWD/outputs/e2e/mlflow/<experiment_name>.db"
```

Open the **`verl_omni_agentic`** experiment's Traces tab, not `Default`: traces
are written under `trainer.project_name`, so experiment 0 is always empty.

`trainer.project_name` is the MLflow experiment and `trainer.experiment_name` the
run name; filter traces by `tags.step`, `tags.sample_index`, `tags.rollout_n`, and
`tags.validate`. `TRACE_MAX_SAMPLES_PER_STEP_PER_WORKER` caps traces per worker
per step (default 5, all `rollout.n` siblings of a selected sample are traced), so
total traces per step is
`cap * num_workers * rollout.n`.

Two caveats worth knowing:

- `rollout.trace.backend` is a single value, not a list: `RolloutTraceConfig` is a
  first-init-wins singleton, so mlflow and WandB Weave are alternatives, never
  both. Set `MLFLOW_TRACE_BACKEND=""` for a trace-free run.
- Traces are exported asynchronously, so the UI can lag the newest step by a few
  seconds. Spin the viewer up *after* launching training so it reads a store that
  already has the sqlite schema.

## Scope note

The RFC also proposed an external per-dimension HTTP protocol. This recipe does
not add that protocol: RPCO dimensions are computed locally by
`agentic_multidim_reward`.
