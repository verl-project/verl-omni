# Integrating a Native Rollout

Last updated: 09/22/2026.

A native rollout runs model sampling through the actor-side model code instead of a separate rollout service such as vLLM-Omni. Select it with `actor_rollout_ref.rollout.name=native`. The native path is useful when sampling needs model-owned state, variable-length multimodal generation, or a joint update that cannot be expressed through the standard server rollout contract.

This guide uses the BAGEL UniGRPO integration as the example. The names under `verl_omni/pipelines/bagel_unigrpo/` are model-specific; the worker, trainer, and engine hook contracts are shared.

## 1. Decide what belongs to the model adapter

Keep architecture and trajectory details in a model package under `verl_omni/pipelines/<model_name>/`. The package should own:

- model loading and any architecture-specific FSDP sharding units;
- the native sampler and its replay data;
- the policy update that consumes the replay data;
- conversion of model outputs into the generic `responses` tensor used by the reward loop.

BAGEL keeps its thinking-token decode, image SDE trajectory, and joint AR/image backward in `bagel_unigrpo`. The native worker does not inspect those fields.

## 2. Register the training adapter

Subclass `DiffusionModelBase` or the appropriate model base and register the `(architecture, algorithm)` pair. For a model that drives submodules functionally instead of calling the root module, return the exact FSDP2 units that participate in those forwards. Do not wrap a root module that the replay path never invokes: its child parameters may remain un-gathered during a functional call.

Implement `build_engine_hooks(module, model_config, optimizer_config)` when native generation or a custom policy update is required. The hook object is created once per engine and may retain model-owned state, but it must not own the optimizer or checkpoint manager.

BAGEL registers `BagelUniGRPO` for `(OmniBagelForConditionalGeneration, unigrpo)`. Its hook creates the `BagelUniPipeline` and `UniGRPOJointUpdater` around the FSDP module.

## 3. Implement the hook contract

A native hook normally implements two methods:

```python
class MyNativeHooks(DiffusionEngineHooks):
    def generate(self, data: TensorDict) -> TensorDict:
        """Return reward inputs and opaque model-owned replay data."""

    def forward_backward_batch(
        self, data: TensorDict, loss_function, forward_only: bool = False
    ) -> dict:
        """Accumulate gradients and return the shared engine result contract."""
```

`generate` receives a batch containing the model's prompt representation. It may return tensors such as `responses` for reward computation and non-tensor values such as trajectory objects. Those trajectory values are opaque to the generic trainer and must be carried through the batch until `forward_backward_batch` consumes them.

`forward_backward_batch` must return `loss`, `metrics`, and `model_output` in the same shape as the normal diffusion engine. It may perform multiple backward passes, but the shared engine owns gradient clearing, clipping, the single optimizer step, learning-rate scheduling, and checkpoints. Reject `forward_only=True` when the native replay has no inference-only implementation instead of returning a fake loss.

## 4. Use the native worker and trainer

`NativeRolloutWorker` is a thin actor-side worker. It initializes the shared actor worker with the actor role, does not construct a `BaseRollout` server, and dispatches `generate` on the actor mesh. This keeps model initialization, FSDP updates, and checkpoints in the shared worker implementation.

`NativeRayDiffusionTrainer` reuses the common reward, advantage, checkpoint, and metric helpers. Its rollout loop is:

```text
native worker generate -> reward -> advantage -> actor update
```

The trainer must not parse model-specific trajectory fields. If the algorithm needs special data, the adapter hook owns that contract.

## 5. Configure the recipe

A native recipe selects the native worker together with the algorithm that provides its adapter and loss:

```yaml
algorithm:
  trainer_type: unigrpo
actor_rollout_ref:
  model:
    algorithm: unigrpo
  rollout:
    name: native
```

The `native` name is an execution backend selector. It does not imply a particular model architecture, sampler, or sharding layout. The model adapter may still require FSDP2 or another explicitly documented training backend.

BAGEL also configures `prompt_token_ids` as the model-native prompt representation and returns `responses` plus an opaque `unigrpo_samples` trajectory stack. The reward loop only consumes `responses`; the BAGEL hook consumes the trajectory during replay.

## 6. Add focused tests

At minimum, test the following on CPU with small stand-ins:

- `rollout.name=native` selects `NativeRolloutWorker`, while `vllm_omni` keeps the standard worker;
- the native worker preserves actor-role initialization and dispatches generation on the actor mesh;
- the adapter returns the expected FSDP2 units and creates independent hook instances;
- generation returns reward inputs and opaque replay data;
- the hook accumulates gradients without stepping the optimizer;
- the native trainer composes the rollout, reward, advantage, and update stages.

A GPU smoke test should additionally verify replay ratios, finite gradients, checkpoint save/resume, and that no vLLM rollout server is started.
