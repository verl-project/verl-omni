# BAGEL Co-RL (joint UND + GEN) training

Last updated: 10/27/2026

One BAGEL-7B-MoT actor learns Hermes tool-calling UND turns with token GRPO and
`generate_image` calls with GEN FlowGRPO. The entrypoint is
`python3 -m verl_omni.trainer.main_omni`; see the
[design RFC](https://github.com/verl-project/verl-omni/issues/453).
Each composite optimizer step waits for all sibling episodes. No weights are
published mid-episode.

## Setup and launch

Install the [training dependencies](../../../docs/start/install.md), download the
published `ByteDance-Seed/BAGEL-7B-MoT` checkpoint, and run from the repository root.
The default acceptance topology uses four GPUs: actor and GEN share the pool,
UND AR uses two-device slices, and the reward model spans the pool with TP=4.
UND must remain the published BAGEL checkpoint emitting Hermes `<tool_call>`;
do not substitute a Qwen actor. The independent Mode (2a) recipe is unchanged.

```bash
export BAGEL_MODEL_PATH="$HOME/models/ByteDance-Seed/BAGEL-7B-MoT"
export REWARD_MODEL=/path/to/UnifiedReward-2.0-qwen3vl-2b
export UNICOT_REFLECTION_DIR=/path/to/UniCoT-Self-Reflection-6K
export UNICOT_BREAKDOWN_DIR=/path/to/UniCoT-Breakdown-3K
bash examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh
```

The launcher prefers `.venv/bin/python3`, falling back to `python3` on PATH. A sibling
`../verl` checkout is prepended only if its profiler import passes; `_LOCAL_VERL`
selects another checkout. Explicit `CUDA_HOME` wins over toolkit discovery. Keep
`nvcc` compatible with Flashinfer's `--compress-mode=size`; the launcher also adds
venv `ninja` and existing CUDA compatibility libraries for bare-shell startup.
`DEBUG=1` enables shell tracing.

The launcher rebuilds UniCoT parquet by default. Each row must carry
`extra_info.reference_image_path` for image rewards. `REBUILD_UNICOT=0` reuses
existing `UNICOT_TRAIN` / `UNICOT_TEST`; missing files still trigger a build.
The default capped split uses 3,000 training rows, reflection mix 0.65, validation
ratio 0.05, and seed 42. `UNICOT_TRAIN_SIZE=full` removes the cap. Reward weights
are builder flags (`--w_reflect/--w_plan/--w_format/--w_tool/--w_result`).
A seeded validation panel of 24 prompts per task excludes duplicate training
prompts; `VAL_PANEL_PER_TASK=0` instead uses the first `VAL_MAX_SAMPLES=32` rows.

## Defaults and overrides

Environment settings replace recipe defaults; final command-line Hydra overrides
win over both. Quote overrides containing spaces. Keep the GEN deploy yaml's
per-replica device count equal to `ROLLOUT_TP`; all parallel widths must divide
the actor pool. Smaller-pool branches remain in the script for smoke budgets,
but the default UND yaml still needs two devices. A one-device smoke requires a
matching UND deploy yaml, `UND_N_GPUS=1`, and explicit memory/topology review.

| Setting | Four-GPU default |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` / `UND_N_GPUS` / `ROLLOUT_TP` / `REWARD_TP` | `0,1,2,3` / `2` / `2` / `4` |
| `ENABLE_RM` / `CORL_REWARD_VERSION` | `1` / `v2`; RM-off branches use `v1` |
| `TRAIN_BSZ` / `N` / `S` | `4` prompts / `8` sibling episodes / `1` seed per call: 32 episodes per step |
| `PPO_MINI` / `PPO_MICRO` | `TRAIN_BSZ` / `1` |
| `GEN_STEPS` / `GEN_HW` / `NOISE_LEVEL` | `10` / `512` / `0.8` |
| `SDE_WINDOW_SIZE` / `SDE_WINDOW_RANGE` | `2` / `[0,2]` |
| `CFG_TEXT_SCALE` / `CFG_IMG_SCALE` | `4.0` / `1.5` |
| `LORA_RANK` / `LORA_ALPHA` | `8` / `16` |
| `MAX_PROMPT_LEN` / `MAX_UND_TURNS` / `MAX_GENERATE_PASSES` | `2048` / `8` / `3` |
| `ROLLOUT_TEMPERATURE` / `ROLLOUT_TOP_P` / `ROLLOUT_TOP_K` / `ROLLOUT_REPETITION_PENALTY` | `0.7` / `0.9` / `50` / `1.05` |
| `ROLLOUT_GPU_MEM_UTIL` / `UND_GPU_MEM_UTIL` / `REWARD_GPU_MEM_UTIL` | `0.25` / `0.40` / `0.15` |
| `ACTOR_LR` / `GEN_LR` | `3e-5` / `3e-5` |
| `LR_WARMUP_RATIO` / `LR_SCHEDULER` / `MIN_LR_RATIO` | `0.05` / `cosine` / `0.1` |
| `NORM_ADV_BY_STD` / `GEN_NORM_ADV_BY_STD` | `False` (raw centered UND) / `True` (batch-wide GEN std) |
| `USE_KL_LOSS` / `KL_LOSS_COEF` / `KL_LOSS_TYPE` | `True` / `0.001` / `low_var_kl`; UND reference is LoRA-off base |
| `ROLLOUT_RS` / `ROLLOUT_RS_THRESHOLD` / `GEN_CLIP_RATIO` | `seq_mean_k1` / `20` / `0.02` |
| `SAVE_FREQ` / `RESUME_MODE` / `MAX_CKPT_KEEP` | `25` / `auto` / `2` |
| `TOTAL_STEPS` / `TEST_FREQ` / `VAL_GEN_STEPS` | `200` / `30` / `50` (deterministic ODE validation) |

`S=1,N<2` is rejected because GEN advantages would be zero. `J` and `K` denote
actual UND turns and generation calls within an episode; legacy environment `J/K`
aliases retain their launcher precedence for `N/S`. `ENABLE_PEC=1` enables both
conditioning reuse and routing affinity; `PEC_SIZE=max(32,S*N*AGENT_WORKERS)` with
8 agent workers. Every weight publish invalidates conditioning and LoRA caches.
`BAGEL_UND_AR_SERVING_READY=1` is an explicit serving-readiness assertion; validate
Hermes tool calls after serving changes. It is not a serving probe.

`RUN_DIR` defaults to `outputs/bagel_corl_$RUN_TS`. `DEFAULT_LOCAL_DIR` overrides
the otherwise shared `checkpoints/<project>/<experiment>` checkpoint directory.
`AGENTIC_VAL_VIZ=1` adds four fixed holdout renders without adding queue rows.
The reward pool handles scoring by default; `JUDGE_SERVER=1` opts into the external
judge sidecar when needed. See the script for backend and threshold overrides.

## Reward and acceptance checks

The composite reward retains reflect, plan, format, tool, result, image, protocol,
and done dimensions. With RM enabled, `v2` uses trusted continuous GEN scores for
image credit. Inspect validation dimensions and `bagel_corl/reward_dim/*` alongside
`reward_valid` / `reward_no_image` means and counts; `critic/score/mean` is a noisy
rollout sample mean, not a learned critic or a convergence proof.

`BAGEL_CORL_DIVERGENCE_ABORT=1` arms the tripwire: excessive KL (>20), gradient norm
(>100), or nonfinite readings fail the step. Keep checkpointing enabled to bound
lost work. Preserve these defaults when comparing against a convergence baseline.

```bash
bash -n examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh
python -m pytest tests/trainer/omni/test_bagel_corl_recipe_on_cpu.py -q
torchrun --standalone --nproc_per_node=2 tests/special_e2e/test_bagel_corl_joint_gradient_cuda.py
```

The CPU test checks captured launch arguments, quoting, caller precedence, and
failure propagation. The manual CUDA test exercises production composite engine
routing and gradients with tiny forward adapters; it does not prove real BAGEL
serving or convergence. Before replacing a working baseline, also run real UND/GEN
rollouts, a composite update, synchronized serving, checkpoint restore, and a
matched-seed short evaluation on the target GPU environment.
