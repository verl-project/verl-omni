# Multi-turn image agents

Last updated: 10/29/2026

A multi-turn image agent generates an image, reads judge feedback, and either
rewrites the image prompt or finishes the episode with `Done.`. The shared
runtime keeps each episode's generated images and judgments together while
multiple episodes run concurrently. Model recipes supply the training setup,
dataset, reward objective, and model-specific rollout behavior.

## Runtime flow

1. `OmniAgentLoopWorker` derives the trajectory path from the step, sample index,
   and rollout index, then opens an episode scope around the agent invocation.
2. `ImageGenToolAgentLoop` runs VeRL's tool-calling loop. For this loop, the worker
   fills unset function-tool and format settings with `tools/image_gen.py` and
   Hermes format, preserving explicit overrides.
3. `generate_image(prompt)` calls the configured image backend, saves its output,
   and registers the generation with the current episode.
4. `judge_image(user_request, image_prompt)` resolves an episode image and records
   valid judge feedback against its generation call. The agent can rewrite and
   generate again, subject to the generation limit and good-enough latch.
5. On scope exit, the runtime restores the parent context and closes the episode.
   Saved artifacts remain on disk.

The main components are listed below, with paths relative to `verl_omni/`:

| Component | Responsibility |
| --- | --- |
| `agent_loop/omni_agent_loop.py` | Bind worker configuration, scope agent invocations, and export rollout artifacts. |
| `agent_loop/tool_agent_loop.py` | Run the multi-turn tool protocol and optional first-call/reflection aids. |
| `tools/image_gen.py` | Execute the frozen generation and judge tools. |
| `tools/trajectory/episode.py` | Own generation records, judgment records, image lookup, and the stop latch. |
| `utils/agentic/config.py` | Bind process-local tool settings and prepare scorer settings for sample metadata. |
| `utils/agentic/artifacts.py` | Resolve run directories, trajectory paths, and artifact identifiers. |

## Select and configure the loop

Use an agent-loop configuration file to register the implementation with VeRL:

```yaml
# /path/to/agent_loops.yaml
- name: image_gen_tool_agent
  _target_: verl_omni.agent_loop.tool_agent_loop.ImageGenToolAgentLoop
```

Add the following settings to an omni training configuration, alongside the
model and rollout settings from its recipe:

```yaml
actor_rollout_ref:
  rollout:
    agent:
      default_agent_loop: image_gen_tool_agent
      agent_loop_config_path: /path/to/agent_loops.yaml

agentic_image_gen:
  vllm_omni_url: http://image-server:8000
  vllm_url: http://judge-server:8001
  max_generate_image_passes: 3
  good_enough_threshold: 0.80
  block_generate_after_yes: true
  force_first_generate: false
  force_reflection_after_judge: true
  e2e_root: /shared/agentic_runs
```

The omni trainer composes defaults from
`verl_omni/trainer/config/agentic/image_gen_tools.yaml`. `vllm_omni_url` selects
the image-generations service; `vllm_url` selects the judge's chat-completions
service. Other generation backends use `qwen_image_url` or `diffusion_tool_url`.
Configure only the backend needed by the recipe. A successful YES judgment can
block further generation; the pass limit counts calls with saved PNGs, excluding
stubs and the few-shot fallback.

The worker and manager bind configuration in each process. Artifact binding reads
directory settings from the incoming configuration, so binding a new run cannot
reuse a previous run's `run_dir` or `e2e_root`. Binding tool settings requires a
configuration object; `bind_agentic_image_gen(None)` raises `ValueError`.

## Episode ownership for custom adapters

Import the episode entry points from `verl_omni.tools.trajectory`. Open one scope
per independent episode and use its owner from tool callbacks:

```python
from verl_omni.tools.trajectory import EpisodeState, current_episode, episode_scope

state = EpisodeState(episode_id="sample-7", user_prompt="Draw a cafe poster")
with episode_scope(state):
    owner = current_episode()
    generated = owner.record_generation(
        call_id="generate-0",
        prompt="A cafe poster with a ceramic cup",
        paths=["/shared/run/rollout_images/poster.png"],
        artifact_ids=["poster-0"],
    )
    owner.record_judgment(generated.call_id, good_enough=True)
    assert owner.good_enough

assert state.closed
```

The path above represents an image already saved by the backend; recording a
generation stores metadata and does not create the file. One generation record
can hold several image paths and their aligned artifact IDs. Unknown generation
IDs are rejected when recording judgments. Image lookup stays within the episode;
an unknown explicit artifact ID returns no image rather than another sample's image.

Scopes propagate through `await` and `asyncio.to_thread`. Context copies share
the same owner, so independent concurrent episodes must each create their own
`EpisodeState`. Nested scopes restore their parent, and re-entering the same
owner leaves it open until its owning scope exits. Exit also runs on exceptions
and cancellation: records and the latch are cleared, and late writes are rejected.
The owner is process-local; send rollout payloads or immutable records across Ray
boundaries rather than passing the state object and its lock.

`episode_context.py`, `episode_artifacts.py`, and `episode_judge.py` adapt existing
helper calls to this owner. The compatibility latch setter only changes the latch;
it does not add judgment history. Custom judge adapters should call
`record_judgment` for actual feedback, as the frozen `judge_image` tool does.

## Artifact layout and import migration

By default, artifacts live under
`outputs/e2e/<trainer.experiment_name>/`. An explicit `agentic_image_gen.e2e_root`
replaces `outputs/e2e`, retaining the experiment-name subdirectory. An explicit
`agentic_image_gen.run_dir` selects the run directory verbatim; when adding this
field through Hydra on the shared config, use
`agentic_image_gen.run_dir=/shared/my_run`.

The run directory contains `rollout_images/`, `rollout_trajectories/`, and
`hermes_actions/`. Closing an episode releases runtime references without deleting
those files. See {doc}`diffusion_media_artifacts` for the separate contract for
model-native rollout media and training tensors.

The old module aliases have been removed. Update direct imports and test
monkeypatch targets as follows. Package-level ContextVar facades are replaced by
explicit `get_active_*`, `set_active_*`, and `reset_active_*` functions; the cleanup
helper is `clear_tool_artifacts_for_active_rollout`. Artifact experiment metadata
uses `utils.agentic.artifacts.get_run_name()` instead of a cached `run_name`:

| Removed module | Replacement |
| --- | --- |
| `verl_omni.tools.trajectory.context` | `verl_omni.tools.trajectory.episode_context` |
| `verl_omni.tools.trajectory.artifacts` | `verl_omni.tools.trajectory.episode_artifacts` |
| `verl_omni.tools.trajectory.judge_latch` | `verl_omni.tools.trajectory.episode_judge` |
| `verl_omni.tools.trajectory.hydra_env` | `verl_omni.utils.agentic.config` |
| `verl_omni.tools.trajectory.paths` | `verl_omni.utils.agentic.artifacts` |
