# Profiling FlowGRPO / diffusion training in VeRL-Omni

Last updated: 11/12/2026.

VeRL-Omni reuses the profiler subsystem from upstream
[verl](https://github.com/verl-project/verl) (`verl.utils.profiler`) and exposes
the same configuration surface for the diffusion trainer. Four profiling tools
are supported:

| Tool          | Backend                                    | Use case                                  |
|---------------|--------------------------------------------|-------------------------------------------|
| `nsys`        | NVIDIA Nsight Systems                      | End-to-end CUDA / kernel timeline tracing |
| `npu`        | Ascend NPU profiler                       | NPU / CPU operator and memory profiling  |
| `torch`       | `torch.profiler`                           | PyTorch-level CPU / CUDA / op profiling   |
| `torch_memory`| `torch.cuda.memory._dump_snapshot`         | CUDA memory allocation snapshots          |

## Configuration overview

Profiling is controlled by two layers of configuration that mirror upstream
verl conventions:

1. **Global** profiler config under `global_profiler` in
   [`diffusion_trainer.yaml`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/config/diffusion_trainer.yaml).
   Selects the tool, the steps to profile, the output directory, and global
   tool-specific options (e.g. nsys controller / worker options).
2. **Per-role** profiler config under `actor_rollout_ref.{actor,ref,rollout}.profiler`
   and `reward.profiler`.
   Inherits defaults from
   [`profiler/profiler.yaml`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/config/profiler/profiler.yaml)
   and selects which ranks to profile and the role-local tool config.

Trainers automatically start and stop collection for steps selected by the
global `steps` list. Diffusion V1 places these calls around `step()`, following
the upstream V1 trainer structure. Non-continuous windows therefore exclude
the subsequent checkpoint, weight-sync, and validation phases.

### Global profiler fields

```yaml
global_profiler:
  _target_: verl.utils.profiler.ProfilerConfig
  tool: null                     # one of: nsys, npu, torch, torch_memory (null disables)
  steps: null                    # e.g. [1, 2, 5]
  profile_continuous_steps: False
  save_path: outputs/profile
  relocate_results: False
  finish_hook_cmd: null
  finish_hook_all_ranks: False
  finish_hook_ranks: []
  global_tool_config:
    nsys: { ... }                # see below
    torch_memory: { ... }
```

The following optional fields control result relocation and finish commands.

| Field | Default | Purpose |
| --- | --- | --- |
| `relocate_results` | `False` | Gather supported backend artifacts into `save_path` when collection stops. |
| `finish_hook_cmd` | `null` | Optional shell command for training workers to run after profiling finishes. `null` disables the command. |
| `finish_hook_all_ranks` | `False` | Run the finish command on every profiled rank when enabled. |
| `finish_hook_ranks` | `[]` | Explicit profiled ranks on which to run the finish command. |

### Per-role profiler fields

```yaml
actor_rollout_ref:
  actor:
    profiler:
      tool: torch                # nsys, npu, torch, torch_memory
      enable: False
      all_ranks: False
      ranks: []
      tool_config:
        npu:
          contents: []           # npu, cpu, memory, shapes, module, stack
          level: level0
          analysis: True
          discrete: False
          profile_token_start: null
          profile_token_end: null
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
by `actor_rollout_ref.rollout.profiler`. Reward-loop Ray actors use a separate
configuration under `reward.profiler`. See recipe 5 for both roles.

In diffusion V1, distillation teachers participate in the training-worker
profiling lifecycle. Standalone teachers (`distillation.nnodes > 0`) use
`actor_rollout_ref.ref.profiler`; teachers colocated with the actor use the
actor's profiler configuration. Shared worker groups are started and stopped
once per profiling window.

Global values for `save_path`, `relocate_results`, and `finish_hook_*` provide
defaults for the per-role configs. Set a field under
`actor_rollout_ref.actor.profiler.<field>` to override it for the actor.
For example, `global_profiler.relocate_results=True` enables result relocation
for roles inheriting that value; an actor override of `False` disables it
for the actor only.

To configure an actor-only finish command, set
`actor_rollout_ref.actor.profiler.finish_hook_cmd` and
`actor_rollout_ref.actor.profiler.finish_hook_ranks='[0]'`. Rank 0 must also
be selected for actor profiling. See [Implementation notes](#implementation-notes)
for finish-command timing.

All the profiler keys below already exist in the composed config, so use
plain `key=value` overrides — a `+key=value` append fails with "An item is
already at ...".

## Quick recipes

The following GPU recipes add CLI overrides on top of
`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`.
The PyTorch actor and rollout options also apply to diffusion V1 GPU launchers
in `sync` and `separate_async` modes.

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

When `global_profiler.tool=nsys` and `steps` is non-empty, the diffusion
entrypoints (`main_diffusion` and `main_diffusion_v1`) launch the Ray
TaskRunner under `nsys` using the `controller_nsight_options` from
`global_profiler.global_tool_config.nsys`.
Workers use `capture-range: cudaProfilerApi`, and the trainer starts and stops
their collection around the configured steps. The controller records the full
TaskRunner lifetime when its Nsight options do not specify a capture range. To
restrict controller collection to the configured steps, add these options:

```bash
+global_profiler.global_tool_config.nsys.controller_nsight_options.capture-range=cudaProfilerApi \
+global_profiler.global_tool_config.nsys.controller_nsight_options.capture-range-end=null \
+global_profiler.global_tool_config.nsys.controller_nsight_options.kill=none
```

When controller `capture-range-end` is null, it is resolved to the number of
discrete profiled steps or contiguous step groups before Ray starts the
TaskRunner.

Diffusion V1 supports NPU (`tool=npu`), PyTorch (`tool=torch`), and Nsight
Systems (`tool=nsys`) profiling.

`*.nsys-rep` files are written by Ray under
`/tmp/ray/session_latest/logs/nsight/` on each node (this path is fixed by
Ray). Open them with `nsys-ui`.

### 4a. Continuous diffusion training timeline

Run the V0 profiler with the Qwen-Image benchmark to inspect a continuous
training window, including old-log-prob inference and actor updates:

```bash
bash examples/run_diffusion_benchmark_nsys_v0.sh \
    examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark.sh
```

Use the V1 profiler with the Wan2.2 v1 sync FSDP2 benchmark:

```bash
bash examples/run_diffusion_benchmark_nsys_v1.sh \
    examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_v1_fsdp2_benchmark.sh
```

Select the profiling entrypoint that matches the trainer generation. Both
runners take the benchmark recipe as their first positional argument. Remaining
arguments are forwarded as Hydra overrides. V0 profiles the controller and
actor workers. V1 also profiles rollout servers, reward-loop workers, and GPU
metrics. Both runners create one continuous capture after an unprofiled warmup
step.

The default run trains for four steps. Step 1 is an unprofiled model/engine
warmup, step 2 opens the continuous capture, step 3 is the steady-state
analysis interval, and step 4 closes the capture. The recipe profiles the Ray
`TaskRunner` and ActorRollout rank 0 with Python stack sampling and NVTX. In
V1, rollout and reward-loop profiling also default to rank 0. Select ranks
independently through each role's Hydra configuration, for example:

```bash
actor_rollout_ref.actor.profiler.ranks='[0,1]' \
actor_rollout_ref.rollout.profiler.ranks='[0,2]' \
reward.profiler.ranks='[0]'
```

Set the corresponding role's `profiler.all_ranks=True` to select all ranks.
Each selected process uses one profiler start/stop pair around the same
continuous window.

Rollout GPU ranks map to their owning replicas, and a selected replica's report
includes its local GPUs. For example, with TP=2 and DP=PP=1, both `[0]` and
`[0,1]` select replica 0 and produce one rollout report containing GPU ranks 0
and 1.

The recipe uses 20 Hz Python stack sampling. Native CPU sampling, context
switches, and OS-runtime events are enabled on the controller and selected
workers, but only one in every eight native samples gets a DWARF backtrace.
CUDA API backtraces and CUDA memory-usage tracking are disabled to limit
overhead; CUDA API and kernel activity remain in each role-local timeline.
These are ordinary Hydra launch options rather than separate wrapper settings.
When a different sampling density or a deep CUDA API investigation is needed,
append force overrides such as these to the command:

```bash
++global_profiler.global_tool_config.nsys.controller_nsight_options.python-sampling-frequency=10 \
++global_profiler.global_tool_config.nsys.worker_nsight_options.python-sampling-frequency=10 \
++global_profiler.global_tool_config.nsys.controller_nsight_options.samples-per-backtrace=4 \
++global_profiler.global_tool_config.nsys.worker_nsight_options.samples-per-backtrace=4 \
++global_profiler.global_tool_config.nsys.worker_nsight_options.cudabacktrace='"sync:100000,memory:100000"'
```

Role-local CUDA traces use the `cuda-sw` backend so independent sessions can
share a GPU. The V1 recipe also collects device-level GPU metrics for all local
GPUs at 20 Hz in a dedicated process, over the same profiling window. Do not
enable GPU metrics in other profiling processes: multiple collectors on the
same device compete for its counters.

In each role-local report:

- **NVTX** ranges show the duration of each training phase.
- **Python Backtrace** shows individual sampling points after zooming in on the
  timeline. Hover over a sampling point to see its captured Python call stack.
- Expand **CUDA HW** with the triangle on the left, then zoom in to see kernels
  from that role and its traced child processes.
- Zoom in on **CUDA API** to see host-side CUDA API calls grouped by their
  originating processes.

The controller normally has no GPU kernels of its own; actor, rollout, and
reward kernels appear in their corresponding role-local reports. Overall GPU
busy and idle intervals appear in the dedicated `gpu_metrics_*.nsys-rep`; GPU
metrics are device-level and do not attribute activity to an individual role.

> [!WARNING]
> The recipe does not set `discard-environment` because the option is not
> available in every Nsight Systems release. If your version supports it, you
> can set it to `true` for both the controller and workers:
>
> ```bash
> +global_profiler.global_tool_config.nsys.controller_nsight_options.discard-environment='"true"' \
> +global_profiler.global_tool_config.nsys.worker_nsight_options.discard-environment='"true"'
> ```
>
> Without this option, reports may contain environment variables such as
> `HF_TOKEN`, so take care when sharing report files.

After training, the wrappers wait for report generation to finish and copy
reports into `$RUN_DIR/nsight`. Set `PROFILE_FINALIZE_TIMEOUT_S` to change the
120-second timeout. Available reports and raw streams are preserved on training
or collection failure. Keep the generated session names and output filename IDs
when overriding Nsight launch options.

Collection reads from `session_latest`. Avoid starting another Ray cluster
during collection, since it can redirect that link. Reusing a Ray cluster and
capture ID may include reports from an earlier run.

Report names identify the process role and capture ID. Open the controller,
actor, rollout-server, reward-loop, and GPU metrics reports together with
Nsight Systems' multi-report view. For reports collected on one host, prefer
`TSC` alignment and verify the alignment source in **Analysis Summary**. `UTC`
is suitable for coarse phase analysis but not millisecond-scale cross-process
latency comparisons.

<!-- TODO: Add an updated Nsight Systems multi-report timeline screenshot from
     a new V1 run, showing controller, actor, rollout, reward-loop, and GPU metrics. -->

This recipe intentionally requires one node. A multi-node variant needs a
collector on every node and synchronized clocks.

### 5. Rollout servers and reward-loop workers

Generation runs in vLLM-Omni server processes, so it needs its own profiler,
driven by `actor_rollout_ref.rollout.profiler`. For phase-scoped collection,
the trainer starts/stops it around the generation phase of each selected step.
Set `tool_config.torch.discrete=True` to enable vLLM's built-in Torch profiler
through the verl rollout integration. With `False`, the integration skips
engine profiling:

```bash
global_profiler.tool=torch \
global_profiler.steps=[1] \
actor_rollout_ref.rollout.profiler.enable=True \
actor_rollout_ref.rollout.profiler.ranks=[0] \
actor_rollout_ref.rollout.profiler.tool=torch \
actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cpu,cuda] \
actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True
```

`ranks` selects global ranks within the rollout workers; the owning replicas
collect traces. Each profiled replica writes its trace to
`{save_path}/agent_loop_rollout_replica_{rank}`, next to the actor traces.

For diffusion V1, `global_profiler.profile_continuous_steps=False` closes
rollout collection at the mode-specific phase boundary. With `True`, collection
stays open across adjacent selected steps. This trainer setting controls the
capture window independently of the rollout integration's `discrete=True`
requirement. In `separate_async`, select ranks
belonging to the standalone rollout replicas according to the actual worker
layout, rather than using replica indices or physical device IDs.
Combine with recipe 1 to capture the actor train phase and the rollout in the
same step.

#### Nsight Systems support (diffusion V1)

Select rollout and reward-loop ranks through their respective profiler
configurations:

- `actor_rollout_ref.rollout.profiler`: rollout server and its engine process tree;
  `ranks` contains global rollout GPU ranks and is mapped to owning replicas.
- `reward.profiler`: reward-loop worker indices
  in the flattened worker list. For named rewards, the shared group comes
  first, followed by native model groups in reward-term order.

For an Nsight Systems capture of one representative process from each role:

```bash
global_profiler.tool=nsys \
global_profiler.steps=[1] \
actor_rollout_ref.rollout.profiler.enable=True \
actor_rollout_ref.rollout.profiler.ranks=[0] \
actor_rollout_ref.rollout.profiler.tool=nsys \
reward.profiler.tool=nsys \
reward.profiler.enable=True \
reward.profiler.ranks=[0] \
global_profiler.global_tool_config.nsys.discrete=False
```

Nsight profiling for these roles requires `tool_config.nsys.discrete=False`.
Set `enable=False` to disable a role or `all_ranks=True` to select all its
ranks. Rollout reports include the server and its engine subprocesses, rather
than a separate report for each GPU worker.

Reward profiling does not pause scoring requests at the capture boundaries. In
asynchronous training, a request may span a boundary; the report contains only
the work captured within the configured window.

For engine-backed reward servers, use `reward.reward_model.rollout.profiler`
(recipe 6), rather than `reward.profiler`. See recipe 4a for the full-window
scripts and dedicated GPU metrics collection.

### 6. Reward-model servers

When `reward.reward_model.enable=True`, the reward model runs in its own
vLLM server processes — the same server stack as recipe 5, driven by
`reward.reward_model.rollout.profiler`:

```bash
global_profiler.tool=torch \
global_profiler.steps=[1] \
reward.reward_model.rollout.profiler.enable=True \
reward.reward_model.rollout.profiler.ranks=[0] \
reward.reward_model.rollout.profiler.tool=torch \
reward.reward_model.rollout.profiler.tool_config.torch.contents=[cpu,cuda] \
reward.reward_model.rollout.profiler.tool_config.torch.discrete=True
```

The trainer starts/stops it around the phase where the servers actually
score: the generation phase when reward computation streams with the rollout
(`reward.reward_model.enable_resource_pool=True`), or the reward phase in
colocate mode. Each profiled replica writes to
`{save_path}/reward_model/agent_loop_rollout_replica_{rank}`, keeping reward
traces apart from the actor rollout ones.

### 7. Ascend NPU profiling (`npu`)

NPU profiling supports diffusion training through `main_diffusion` or
`main_diffusion_v1`, and Omni training through `main_omni` with PPO V1. You can collect actor training
and rollout inference traces in the same run, or enable either role separately.
Set `global_profiler.tool=npu` and configure each role under
`actor_rollout_ref.{actor,rollout}.profiler`.

#### Before you start

- Set up the [NPU environment](../start/install_npu.md) and the training data.
- Install `torch-npu>=2.10.0.post6` with a compatible PyTorch and CANN environment.
  Start with `contents='[npu,cpu]'` and add `module` or `stack` as needed;
  rollout collection options also depend on the inference backend.

Diffusion V1 (`trainer.use_v1=True`) supports `sync` and `separate_async`.
Use the same per-role options below, with the device-selection constraints
for your training mode.

#### Collect actor and rollout traces

Choose a launcher for your model. The same profiling overrides work with both
of these NPU examples:

| Training workload | Launcher |
| --- | --- |
| Qwen-Image FlowGRPO LoRA | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh` |
| Qwen3-Omni GSPO on AVQA | `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_avqa_v1.sh` |

Set `RECIPE` to your launcher and select training steps 1 and 2:

```bash
RECIPE=examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh
bash "$RECIPE" \
    global_profiler.tool=npu \
    global_profiler.steps='[1,2]' \
    global_profiler.profile_continuous_steps=False \
    global_profiler.save_path=./outputs/profile \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.tool=npu \
    actor_rollout_ref.actor.profiler.ranks='[0]' \
    actor_rollout_ref.actor.profiler.tool_config.npu.contents='[npu,cpu]' \
    actor_rollout_ref.actor.profiler.tool_config.npu.level=level0 \
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True \
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.tool=npu \
    actor_rollout_ref.rollout.profiler.ranks='[0]' \
    actor_rollout_ref.rollout.profiler.tool_config.npu.contents='[npu,cpu]' \
    actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
```

The example selects actor rank 0 and the rollout replica that owns global rank 0.
A tensor-parallel rollout engine collects across its whole replica. Profiling
output is saved under `./outputs/profile`, with rollout output in a
`agent_loop_rollout_replica_{rank}` subdirectory. To collect all ranks, set
`all_ranks=True` separately for each role. Profiling is disabled by default;
select the global steps and explicitly enable each role to collect traces.
These fields already exist, so use plain `key=value` overrides without a `+` prefix.

The verl rollout integration requires
`rollout.profiler.tool_config.npu.discrete=True` for NPU engine profiling,
independently of the trainer's capture-window setting.
The example also sets actor `discrete=True` to collect individual training
stages. Actor-only collection can use either `discrete` value where supported
by the backend.

For diffusion V1 (`sync` and `separate_async`),
`global_profiler.profile_continuous_steps=False` stops rollout collection at
the mode-specific phase boundary. With `True`, rollout collection stays open
across adjacent selected steps and stops at the end of that window. This
setting controls trainer lifecycle; backend-specific `discrete` requirements
still apply. NPU continuous-window capture requires validation on NPU hardware.

Concurrent NPU collectors must use disjoint physical devices. For diffusion
V1 `separate_async`, set `all_ranks=False` and select actor and standalone
rollout ranks accordingly. This also applies with
`trainer.v1.separate_async.hybrid_rollout.enable_switch=True`; lending actor
devices to hybrid rollout does not serialize collectors on the same device.
When switching is disabled, V1 does not profile the unused hybrid manager.

Set either role's `profiler.enable=False` to
collect only the other role. Reward-model profiling is configured separately
under `reward.reward_model.rollout.profiler` (recipe 6).

`actor_rollout_ref.rollout.profiler.ranks` contains global
ranks within the rollout workers, not replica IDs or physical device IDs.
For replica world size `W = TP * DP * PP`, selecting rank `r` selects replica
`r // W`; collection may
include all devices of that replica. In diffusion V1 `separate_async`, with
`H` hybrid replicas preceding the
standalone replicas, standalone ranks begin at `H * W`. For example, eight
hybrid devices with `TP=2, DP=PP=1` occupy ranks 0–7: actor rank `[0]` and
standalone rollout rank `[8]` select disjoint devices, provided the resource
pools are placed accordingly. Check actual replica placement; equal or unequal
rank numbers across roles alone do not prove physical-device overlap.

`all_ranks=true` remains usable when the participating collectors are physically
disjoint. Manual rank selection cannot make overlapping NPU collectors safe.
In asynchronous training with a replay buffer, profile steps denote trainer
time windows, not the policy version of buffered samples. Check trace contents
against the selected ranks and actual engine placement.

#### Collection options

The per-role `tool_config.npu` block also exposes the following options:

| Option | Values / behavior | Default |
| --- | --- | --- |
| `contents` | Collection options: `npu`, `cpu`, `memory`, `shapes`, `module`, `stack` | `[]` |
| `level` | `level_none`, `level0`, `level1`, `level2` | `level0` |
| `analysis` | Automatically parse the collected data | `True` |
| `discrete` | Separate databases per task when `True`; tasks in one training step share a database when `False` | `False` |
| `profile_token_start` / `profile_token_end` | Response-token collection bounds for Omni autoregressive decoding (end exclusive), where supported by the inference engine | `null` / `null` (full collection) |

`profile_token_start` and `profile_token_end` select response-token positions
during Omni autoregressive decoding in compatible rollout engines. They do not
select diffusion denoising steps or control the actor's forward/backward passes.

The table lists defaults for the shared diffusion/reward profiler config.
Omni actor/ref/rollout defaults come from upstream verl and can differ.

#### View the results

To inspect the collected Ascend traces, follow the upstream guide's
[visualization and offline parsing instructions](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/en/dev_guide/performance/ascend_profiling.rst#visualization).
When automatic analysis is disabled, **parse the raw data offline before opening
it in MindStudio Insight**. Run the following Python code in your torch-npu
environment, replacing the example path with your actual `*_ascend_pt` trace
directory:

```python
import torch_npu

torch_npu.profiler.profiler.analyse(
    profiler_path="./outputs/profile/actor_update/worker_123_20260918_ascend_pt"
)
```

To parse multiple traces together, you can also pass their immediate parent
directory (for example, `./outputs/profile/actor_update`). The parser checks
that directory and its immediate children; it does not recursively search
all nested stage directories.

Repeat for other actor stages or rollout trace directories as needed, then open
the parsed results in MindStudio Insight. Rollout collection is performed by the inference
engine, so its parsing behavior and supported options can differ from the actor's;
the actor's `analysis=True` does not configure rollout parsing.

#### Further reading

- [verl NPU profiling examples](https://github.com/verl-project/verl/blob/2b4ed6826a39205838d0ca004bdd942334dbe4b6/examples/profile/README.md#npu-profiling)
  cover continuous and discrete collection. Their `PROFILE_*` environment
  variables apply to the upstream scripts; use the Hydra overrides above with
  VeRL-Omni launchers.
- [Ascend profiling guide](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/en/dev_guide/performance/ascend_profiling.rst)
  explains collection levels, Agent Loop profiling, and trace analysis.

## Lightweight profiling recipe

Profiling a full FlowGRPO step produces a large trace that is slow to open.
The SD3.5-Medium V1 launcher below profiles lightweight FlowGRPO OCR steps —
4 prompts, 2 rollouts per prompt, 4 denoising steps, and 256px images — after
a warmup: it runs 5 training steps, leaves steps 1-2 unprofiled as warmup, and
records only steps 3 and 4. It reuses
the existing V1 sync recipe and collects actor rank 0 and rollout rank 0 using
PyTorch. It needs two actor/rollout GPUs and one OCR reward-model GPU; reward
inference still runs, but this launcher does not collect reward-model traces.

Run from the repository root in an installed CUDA training environment. Prepare
the OCR parquet files with
`examples/flowgrpo_trainer/data_process/sd3_ocr.py` and obtain access to
`stabilityai/stable-diffusion-3.5-medium` (or set `MODEL_PATH` to a local copy).
`OCR_WORKSPACE` points to the directory containing
`data/ocr/sd3/{train,test}.parquet`. It defaults to `WORKSPACE`, then `$HOME`.

```bash
OCR_WORKSPACE=/path/to/workspace \
PROFILE_DIR=./outputs/profile_sd35_v1 \
bash examples/flowgrpo_trainer/sd35/profile_sd35_medium_ocr_lora_v1.sh
```

Traces are written under `PROFILE_DIR` (default `./outputs/profile_sd35_v1`);
each recorded step writes its own trace. Open an actor Chrome trace in Perfetto
or `chrome://tracing`; it should contain
`actor_update` and `mini_batch0`/`mini_batch1` ranges. Rollout traces come from
the inference engine in separate processes. Check that both actor and rollout
artifacts contain CPU/CUDA events. A successful training exit alone does not
establish that a trace was collected.

Hydra overrides passed after the script name take precedence. For a single
cold sample, use `trainer.total_training_steps=1 'global_profiler.steps=[1]'`.
For actor-only collection, use `actor_rollout_ref.rollout.profiler.enable=False`.

### Selecting update mini-batches

By default the actor trace contains the whole selected RL step. To collect only
the second update mini-batch while keeping each log-prob stage in its own trace:

```bash
bash examples/flowgrpo_trainer/sd35/profile_sd35_medium_ocr_lora_v1.sh \
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True \
    '+actor_rollout_ref.actor.profiler.tool_config.torch.schedule={_target_:verl.utils.profiler.config.TorchProfilerScheduleConfig,skip_first:1,wait:0,warmup:0,active:1,repeat:1}'
```

The schedule advances once per update mini-batch, including across PPO epochs.
The lightweight defaults have two mini-batches per update. The `+` appends the
optional schedule block; the other profiler keys are already declared. With
`discrete=False`, skipping early mini-batches also drops earlier log-prob stages
from the shared trace. Schedules never select diffusion denoising timesteps.

The launcher's `trainer.*` overrides are intentional: the run's last step force-triggers
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
  The nested actor/reference training workers receive the typed per-role profiler
  config. Actor updates label and advance each mini-batch following the pinned
  `verl/workers/engine_workers.py`, so PyTorch schedules can sub-sample updates.
* `global_profiler.profile_continuous_steps=True` keeps one profiling window
  open across consecutive steps in `global_profiler.steps`, which is helpful
  for analysing inter-step behaviour. Diffusion V1 closes the window at the
  final training step even if the configured range extends beyond the
  training limit. Rollout profiling supports the same continuous windows;
  non-continuous collection stops at the mode-specific phase boundary.
* Standalone Ray actors inherit the global capture window and worker Nsight
  options through `profiler.ray_actor_nsight_options`.
* In the legacy diffusion trainer, for the rollout servers, the trainer calls
  `llm_server_manager.start_profile()`/`stop_profile()` around the generation
  phase of profiled steps; the servers record through vLLM's built-in torch
  profiler (recipe 5). The reward-model servers are driven the same way
  through `verl_omni.reward_loop.OmniRewardLoopManager` (recipe 6).
* `relocate_results` and `finish_hook_*` are forwarded to upstream workers.
  In diffusion V1, a training-worker finish command runs when closing the
  window containing the last selected step within the training limit. For
  example, `steps=[1,100]` with four training steps runs the command at step 1.
  Rollout-only collection does not run a training-worker finish hook.
* If profiling startup or training fails, diffusion V1 attempts to stop all
  active targets, including a target that raised during startup. Cleanup does
  not run finish commands or replace the original exception. During normal
  window closure, all remaining stop calls are attempted before the first
  stop error is reported.

## Further reading

* Upstream PyTorch profiler guide:
  [`docs/perf/torch_profiling.md` in verl](https://github.com/verl-project/verl/blob/main/docs/perf/torch_profiling.md)
* Upstream Nsight guide:
  [`docs/perf/nsight_profiling.md` in verl](https://github.com/verl-project/verl/blob/main/docs/perf/nsight_profiling.md)
