# Profiling FlowGRPO / diffusion training in VeRL-Omni

Last updated: 05/11/2026.

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
`actor_rollout_ref.rollout.profiler`. The rollout profiler is only active when
the rollout role is colocated with the actor (the hybrid engine setup used by
FlowGRPO today).

## Quick recipes

The following recipes add CLI overrides on top of
`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`.

### 1. PyTorch profiler — end-to-end

Capture a single trace per profiled step (combined CPU + CUDA activities).

```bash
+global_profiler.tool=torch \
+global_profiler.steps=[1,2,5] \
+global_profiler.save_path=./outputs/profile \
+actor_rollout_ref.actor.profiler.enable=True \
+actor_rollout_ref.actor.profiler.all_ranks=True \
+actor_rollout_ref.actor.profiler.tool=torch \
+actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda] \
+actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False
```

The traces land under `outputs/profile`. View them in
[Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing`.

### 2. PyTorch profiler — discrete (per-stage)

Discrete mode produces one database per `@DistProfiler.annotate`-decorated
function within a step, which is useful when zooming into a specific phase.

```bash
+global_profiler.tool=torch \
+global_profiler.steps=[3] \
+actor_rollout_ref.actor.profiler.enable=True \
+actor_rollout_ref.actor.profiler.ranks=[0] \
+actor_rollout_ref.actor.profiler.tool=torch \
+actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True \
+actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda]
```

### 3. CUDA memory snapshots (`torch_memory`)

The `torch_memory` tool records allocation history and dumps a snapshot at the
end of each profiled step. Visualize the resulting JSON files at
[pytorch.org/memory_viz](https://pytorch.org/memory_viz).

```bash
+global_profiler.tool=torch_memory \
+global_profiler.steps=[1,2] \
+actor_rollout_ref.actor.profiler.enable=True \
+actor_rollout_ref.actor.profiler.all_ranks=True \
+actor_rollout_ref.actor.profiler.tool=torch_memory
```

### 4. NVIDIA Nsight Systems (`nsys`)

Nsight requires `nsys` to be installed on every node and the `nvtx` Python
package available in the training environment (`pip install nvtx`).

```bash
+global_profiler.tool=nsys \
+global_profiler.steps=[1,2] \
+global_profiler.profile_continuous_steps=True \
+actor_rollout_ref.actor.profiler.enable=True \
+actor_rollout_ref.actor.profiler.all_ranks=True \
+actor_rollout_ref.actor.profiler.tool=nsys
```

When `global_profiler.tool=nsys` and `steps` is non-empty, the FlowGRPO
entrypoint launches the Ray TaskRunner under `nsys` using the
`controller_nsight_options` from `global_profiler.global_tool_config.nsys`.
Workers are launched with `worker_nsight_options`, including the required
`capture-range: cudaProfilerApi` flag.

`*.nsys-rep` files are written by Ray under
`/tmp/ray/session_latest/logs/nsight/` on each node (this path is fixed by
Ray). Open them with `nsys-ui`.

## Implementation notes

* Workers are wrapped with `verl.utils.profiler.DistProfilerExtension`, which
  exposes `start_profile`/`stop_profile` Ray methods. The diffusion trainer
  invokes them around each profiled step, mirroring
  [`verl/trainer/ppo/ray_trainer.py`](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py).
* `global_profiler.profile_continuous_steps=True` keeps a single profiling
  database open across consecutive steps in `global_profiler.steps`, which is
  helpful for analysing inter-step behaviour.
* In hybrid-engine FlowGRPO (the default), the rollout shares the actor
  worker, so configuring `actor_rollout_ref.actor.profiler` is usually enough
  to capture the full step.

## Further reading

* Upstream PyTorch profiler guide:
  [`docs/perf/torch_profiling.md` in verl](https://github.com/verl-project/verl/blob/main/docs/perf/torch_profiling.md)
* Upstream Nsight guide:
  [`docs/perf/nsight_profiling.md` in verl](https://github.com/verl-project/verl/blob/main/docs/perf/nsight_profiling.md)
