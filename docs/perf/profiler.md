# Profiling FlowGRPO / diffusion training in VeRL-Omni

Last updated: 07/10/2026.

VeRL-Omni reuses the profiler subsystem from upstream
[verl](https://github.com/verl-project/verl) (`verl.utils.profiler`) and exposes
the same configuration surface for the diffusion trainer. Three profiling tools
are supported:

| Tool          | Backend                                    | Use case                                  |
|---------------|--------------------------------------------|-------------------------------------------|
| `nsys`        | NVIDIA Nsight Systems                      | End-to-end CUDA / kernel timeline tracing |
| `torch`       | `torch.profiler`                           | PyTorch-level CPU / CUDA / op profiling   |
| `torch_memory`| `torch.cuda.memory._dump_snapshot`         | CUDA memory allocation snapshots          |

> supported by the diffusion trainer at this time.

## Configuration overview

Profiling is controlled by two layers of configuration that mirror upstream
verl conventions:

1. **Global** profiler config under `global_profiler` in
   [`diffusion_trainer.yaml`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/config/diffusion_trainer.yaml).
   Selects the tool, the steps to profile, the output directory, and global
   tool-specific options (e.g. nsys controller / worker options).
2. **Per-role** profiler config under `actor_rollout_ref.{actor,ref,rollout}.profiler`.
   Inherits defaults from
   [`profiler/profiler.yaml`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/config/profiler/profiler.yaml)
   and selects which ranks to profile and the role-local tool config.

A typical training step automatically calls `start_profile` before the step
begins and `stop_profile` after validation, so as long as the global
`steps` list contains the current step the profiler is engaged.

### Global profiler fields

```yaml
global_profiler:
  _target_: verl.utils.profiler.ProfilerConfig
  tool: null                     # one of: nsys, torch, torch_memory (null disables)
  steps: null                    # e.g. [1, 2, 5]
  profile_continuous_steps: False
  save_path: outputs/profile
  global_tool_config:
    nsys: { ... }                # see below
    torch_memory: { ... }
```

### Per-role profiler fields

```yaml
actor_rollout_ref:
  actor:
    profiler:
      tool: torch                # nsys, torch, torch_memory
      enable: False
      all_ranks: False
      ranks: []
      tool_config:
        nsys: { discrete: ... }
        torch:
          contents: []           # cuda, cpu, memory, shapes, stack
          discrete: False
        torch_memory:
          trace_alloc_max_entries: 100000
          stack_depth: 32
```

The same block exists under `actor_rollout_ref.ref.profiler` and
`actor_rollout_ref.rollout.profiler`. Generation runs in separate vLLM-Omni
server processes, not in the actor worker, so it has its own profiler driven
by `actor_rollout_ref.rollout.profiler` (see recipe 5).

All the profiler keys below already exist in the composed config, so use
plain `key=value` overrides — a `+key=value` append fails with "An item is
already at ...".

## Quick recipes

The following recipes add CLI overrides on top of
`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`.

### 1. PyTorch profiler — end-to-end

Capture a single trace per profiled step (combined CPU + CUDA activities).

```bash
global_profiler.tool=torch \
global_profiler.steps=[1,2,5] \
global_profiler.save_path=./outputs/profile \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.actor.profiler.all_ranks=True \
actor_rollout_ref.actor.profiler.tool=torch \
actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda] \
actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False
```

The traces land under `outputs/profile`. View them in
[Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing`.

### 2. PyTorch profiler — discrete (per-stage)

Discrete mode produces one database per `@DistProfiler.annotate`-decorated
function within a step, which is useful when zooming into a specific phase.

```bash
global_profiler.tool=torch \
global_profiler.steps=[3] \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.actor.profiler.ranks=[0] \
actor_rollout_ref.actor.profiler.tool=torch \
actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True \
actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda]
```

### 3. CUDA memory snapshots (`torch_memory`)

The `torch_memory` tool records allocation history and dumps a snapshot at the
end of each profiled step. Visualize the resulting JSON files at
[pytorch.org/memory_viz](https://pytorch.org/memory_viz).

```bash
global_profiler.tool=torch_memory \
global_profiler.steps=[1,2] \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.actor.profiler.all_ranks=True \
actor_rollout_ref.actor.profiler.tool=torch_memory
```

### 4. NVIDIA Nsight Systems (`nsys`)

Nsight requires `nsys` to be installed on every node and the `nvtx` Python
package available in the training environment (`pip install nvtx`).

```bash
global_profiler.tool=nsys \
global_profiler.steps=[1,2] \
global_profiler.profile_continuous_steps=True \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.actor.profiler.all_ranks=True \
actor_rollout_ref.actor.profiler.tool=nsys
```

When `global_profiler.tool=nsys` and `steps` is non-empty, the FlowGRPO
entrypoint launches the Ray TaskRunner under `nsys` using the
`controller_nsight_options` from `global_profiler.global_tool_config.nsys`.
Workers are launched with `worker_nsight_options`, including the required
`capture-range: cudaProfilerApi` flag.

`*.nsys-rep` files are written by Ray under
`/tmp/ray/session_latest/logs/nsight/` on each node (this path is fixed by
Ray). Open them with `nsys-ui`.

### 5. Rollout servers (vLLM-Omni)

Generation runs in vLLM-Omni server processes, so it needs its own profiler,
driven by `actor_rollout_ref.rollout.profiler`. The trainer starts/stops it
around the generation phase of each step in `global_profiler.steps`, and the
servers record through vLLM's built-in torch profiler (`discrete=True` is
required — the server rejects continuous mode):

```bash
global_profiler.tool=torch \
global_profiler.steps=[1] \
actor_rollout_ref.rollout.profiler.enable=True \
actor_rollout_ref.rollout.profiler.ranks=[0] \
actor_rollout_ref.rollout.profiler.tool=torch \
actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cpu,cuda] \
actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True
```

`ranks` selects rollout replicas (one replica per
`rollout.agent.num_workers`). Each profiled replica writes its trace to
`{save_path}/agent_loop_rollout_replica_{rank}`, next to the actor traces.
Combine with recipe 1 to capture the actor train phase and the rollout in the
same step.

## Lightweight profiling recipe

Profiling a full FlowGRPO step produces a large trace that is slow to open.
Every recipe script under `examples/` passes `"$@"` through to the same
`diffusion_trainer` config, and Hydra resolves duplicate overrides
last-wins — so appending overrides to any recipe shrinks its footprint
without editing the script. The following profiles a single lightweight step
of the SD3.5 OCR recipe (2 rollouts instead of 8, 4 denoising steps instead
of 10, 256px instead of 384px, train batch 4 instead of 8), capturing the
actor train phase and the rollout servers (recipes 1 and 5 combined):

```bash
bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.algo.sde_window_range=[0,4] \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.logger='["console"]' \
    global_profiler.tool=torch \
    global_profiler.steps=[1] \
    global_profiler.save_path=./outputs/profile_sd35 \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.ranks=[0] \
    actor_rollout_ref.actor.profiler.tool=torch \
    actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda] \
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.ranks=[0] \
    actor_rollout_ref.rollout.profiler.tool=torch \
    actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cpu,cuda] \
    actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True
```

Measured on 3×RTX 4090 against the recipe defaults: traces 163 MB → 32 MB,
profiled step 616 s → 70 s.

The `trainer.*` lines are not optional: the run's last step force-triggers
checkpoint saving and validation when `save_freq`/`test_freq` > 0, and a
leftover checkpoint auto-resumes past the profiled step, silently skipping
profiling.

### Adapting to other recipes

The `trainer.*`, `global_profiler.*` and `*.profiler.*` overrides above work
unchanged for any recipe. Re-derive the footprint overrides from the
recipe's own values, minding two couplings:

- **Denoising steps vs. SDE window**: a window reaching past the last
  denoising step produces ragged per-sample tensors and fails rollout
  post-processing. Pin `sde_window_range=[0,<num_inference_steps>]` and keep
  `num_inference_steps >= sde_window_size` (the SD3.5 recipe's window —
  size 3, range `[0,5]` — assumes 10 steps).
- **Batch vs. micro batch**: the diffusion engine chunks batches statically,
  so the per-GPU sample count (`train_batch_size * rollout.n / num actor
  GPUs`) must stay divisible by each micro batch size. The SD3.5 recipe's
  micro batch of 8 assumes 32 samples per GPU; the lightweight footprint
  leaves 4, hence 2.

## Implementation notes

* Workers are wrapped with `verl.utils.profiler.DistProfilerExtension`, which
  exposes `start_profile`/`stop_profile` Ray methods. The diffusion trainer
  invokes them around each profiled step, mirroring
  [`verl/trainer/ppo/ray_trainer.py`](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py).
* `global_profiler.profile_continuous_steps=True` keeps a single profiling
  database open across consecutive steps in `global_profiler.steps`, which is
  helpful for analysing inter-step behaviour.
* For the rollout servers, the trainer calls
  `llm_server_manager.start_profile()`/`stop_profile()` around the generation
  phase of profiled steps; the servers record through vLLM's built-in torch
  profiler (recipe 5).

## Further reading

* Upstream PyTorch profiler guide:
  [`docs/perf/torch_profiling.md` in verl](https://github.com/verl-project/verl/blob/main/docs/perf/torch_profiling.md)
* Upstream Nsight guide:
  [`docs/perf/nsight_profiling.md` in verl](https://github.com/verl-project/verl/blob/main/docs/perf/nsight_profiling.md)
