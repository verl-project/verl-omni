# How to Add a New Omni Model

Last updated: 09/22/2026.

This guide walks through adding a new omni (multimodal autoregressive) model to
the verl-omni training framework. It uses the Qwen3-Omni Thinker adapter as a
**running example**, not as the only valid pattern. Your model's architecture,
decomposition, and required adapter logic may differ. All adapter code lives
under [`verl_omni/pipelines/`](https://github.com/verl-project/verl-omni/tree/main/verl_omni/pipelines).

## 1. Understand the architecture

Decide which **training stage** you want to train and how the model decomposes:

- **Stage-split**: Multi-component omni models (thinker → talker → code2wav)
  train one selected autoregressive stage during RL post-training. Other
  components are stripped before FSDP wrapping to save memory. Qwen3-Omni
  trains the thinker; adapters for other architectures may select a different
  autoregressive stage.
- **Encoder-frozen**: Vision/audio encoders are typically frozen during RL
  training (`freeze_vision_tower=True`). The training adapter's
  `get_strip_modules` excludes them from the trainable set if they are separate
  submodules. If they stay in the module graph and are *conditionally skipped*
  rather than removed, keep them sharded-safe with
  `get_fsdp_ignored_module_names` instead — see §2 and §6.
- **Discrete-token**: Unlike diffusion models, omni models produce discrete
  text tokens. RL algorithms (GSPO, GRPO, RLOO) are selected through standard
  verl config fields (`actor.policy_loss.loss_mode`,
  `algorithm.adv_estimator`) — the adapter is algorithm-agnostic.

The stage-split decomposition described above is specific to Qwen3-Omni. Your
omni model may have a simpler (single-stage) or different multi-stage
architecture.

## 2. Create the training adapter

Subclass `OmniModelBase` (see
[`verl_omni/pipelines/model_base.py`](https://github.com/verl-project/verl-omni/tree/main/verl_omni/pipelines/model_base.py)
and implement these methods. Descriptions below use Qwen3-Omni as an example —
adapt each implementation to your model's architecture:

- **`get_strip_modules(model_config)`**: Return a list of submodule attribute
  names to delete before FSDP wrapping (e.g. `["talker", "code2wav",
  "code_predictor"]`). This removes inference-only stages from the trainable
  module and is called by the base `configure_model` implementation.

- **`configure_processor(model_path, model_config)`**: Load and configure the
  multimodal processor. For Qwen3-Omni, this swaps `processor.config` to
  `thinker_config`, binds `get_rope_index` (cast to `int64` to avoid bf16
  rounding) and `get_llm_pos_ids_for_vision` to the processor, and binds
  `dedup_pad_tokens` to collapse consecutive multimodal pad tokens that
  would otherwise be double-expanded in AR mode.

- **`configure_tokenizer(model_path, model_config)`**: Load the tokenizer.
  Qwen3-Omni loads `chat_template.json` from the model checkpoint if the
  tokenizer config does not contain one — a common pattern for models that
  ship the template separately.

- **`configure_model(module, model_config)`**: Called after the base-class
  stripping. Qwen3-Omni redirects `module.forward` →
  `module.thinker.forward`, swaps the embedding accessors, and sets
  `module._no_split_modules` to the correct decoder layer class for FSDP.
  This method runs before FSDP wrapping and LoRA injection.

- **`get_fsdp_ignored_module_names(model_config)`** (optional): Return
  submodule name components to leave unsharded under FSDP2; default `[]`.
  Declare the frozen encoders when they *stay in the module graph* but their
  forward is skipped for some micro-batches — an unsharded forward emits no
  collectives, so skipping it cannot desync the ranks (e.g. the MiniCPM-o
  adapter in #572 returns `["apm", "vpm", "resampler"]`). Ignored parameters
  must stay frozen: FSDP2 does not synchronize their gradients. FSDP2 only —
  under `strategy=fsdp` the engine raises when the list is non-empty.

- **`register_auto_classes()`** (optional): Register classes supplied by an
  optional model package with the appropriate Transformers Auto APIs. The model
  config resolves one `(architecture, stage)` adapter before calling this hook;
  the base implementation is a no-op. Set the adapter's `auto_model_class` when
  the default `AutoModelForMultimodalLM` loader does not own the architecture;
  the FSDP engine still owns `from_pretrained`.

- **`prepare_model_inputs(model_inputs, micro_batch, model_config)`**
  (optional): Validate model-native trajectory or conditioning data retained by
  rollout and add it to the actor forward inputs. Per-sample rollout data starts
  under a model-defined, namespaced key in `AgentLoopOutput.extra_fields`;
  `AgentLoopWorker` batches that key into the top level of `micro_batch`. For
  example, data stored as `output.extra_fields["your_model_replay"]` is consumed
  as `micro_batch["your_model_replay"]`. This is required when the policy token
  sequence alone cannot reconstruct the exact sampled trajectory. Missing
  required fields or inconsistent shapes should raise an actionable error; the
  adapter must not silently reconstruct a different trajectory.

Reference:
[`verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py`](../../verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py)

## 3. Create the rollout adapter

Subclass `OmniRolloutPipelineBase` (see
[`verl_omni/pipelines/model_base.py`](../../verl_omni/pipelines/model_base.py))
and implement:

- **`build_stage_configs(pipeline_mode)`**: Return a list of per-stage
  pipeline topology objects. Qwen3-Omni delegates to vLLM-Omni's frozen
  `QWEN3_OMNI_THINKER_ONLY_PIPELINE` for thinker-only training and supports
  `thinker_talker` / `full` modes for inference.

- **`rollout_flags(pipeline_mode)`**: Return per-stage flags dict. For
  thinker-only mode this is empty (text output). Multi-stage modes return
  `return_hidden_states` flags so intermediate hidden states flow between
  pipeline stages.

- **`get_pipeline_id(pipeline_mode)`**: Return the vLLM-Omni pipeline
  `model_type` string, used when auto-generating the deploy config YAML.

Optional overrides fall into four groups:

- Pipeline setup: `ensure_pipeline_registered`, `get_engine_hf_overrides`, and
  `get_stage_engine_extras`.
- Policy and resource behavior: `policy_stage_id` identifies the stage whose
  sampling parameters and logprobs define the trained policy;
  `weight_sync_stage_ids` identifies the stages that receive actor weights.
- Request construction: `prepare_engine_prompt`. When this hook returns a
  custom prompt, the adapter must include any non-`None`
  `mm_processor_kwargs`; the shared strategy adds them automatically only to
  its default prompt.
- Multi-stage output assembly: override `combine_engine_outputs` to opt into
  retaining outputs from every stage marked `final_output` in the pipeline
  topology. Adapters that keep the default hook preserve the engine's existing
  single-output behavior. A custom combiner must also handle abort outputs with
  empty token IDs and, pending
  [vllm-omni#6973](https://github.com/vllm-project/vllm-omni/issues/6973), an
  empty output list.

Their defaults preserve the existing single-output AR behavior. Override only
the hooks required by the model. A stage-split adapter may, for example, limit
actor weight synchronization to its trainable stage while retaining outputs
from both the policy and decoder stages.

When training an omni model's autoregressive Talker stage, also override
`postprocess_agent_loop_output`. Put the sampled policy sequence in
`response_ids`, align `response_mask` and optional `response_logprobs`
one-to-one, and retain model-native acoustic trajectory and conditioning data
under a model-defined, namespaced key in `extra_fields`. The corresponding
training adapter consumes the batched top-level key in `prepare_model_inputs`.
The common contract intentionally does not prescribe the key name, its nested
schema, a codebook count, or a conditioning source.

Reference:
[`verl_omni/pipelines/qwen3_omni/omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen3_omni/omni_rollout_adapter.py)

## 4. Register both adapters

Registration uses Python decorators at class-definition time:

```python
@OmniModelBase.register("YourArchitectureName", stage="thinker")
class YourThinkerAdapter(OmniModelBase):
    ...

@OmniRolloutPipelineBase.register("your_pipeline_name")
class YourRolloutAdapter(OmniRolloutPipelineBase):
    ...
```

The `architecture` key for `OmniModelBase` matches the HuggingFace config
`architectures[0]` value. The `model_type` key for
`OmniRolloutPipelineBase` matches the vLLM-Omni pipeline registry name.

To ensure registration fires before the trainer starts, import your adapter
module from [`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py).
The `VERL_USE_EXTERNAL_MODULES=verl_omni` environment variable triggers verl
to import `verl_omni`, which in turn imports the pipeline package and
activates all registrations. No `external_lib` CLI argument is needed.

## 5. Write the run script

The V1 trainer uses pure CLI overrides on `verl_omni.trainer.main_omni` with
no YAML config files or `--config-path/--config-name`:

```bash
export VERL_USE_EXTERNAL_MODULES=verl_omni

python3 -m verl_omni.trainer.main_omni \
    data.train_files="$HOME/data/train.parquet" \
    data.val_files="$HOME/data/test.parquet" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.agent.default_agent_loop=omni_single_turn_agent \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="your_pipeline_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    "$@"
```

Key points:
- No `external_lib` — adapters are auto-registered via the Python import
  triggered by `VERL_USE_EXTERNAL_MODULES=verl_omni`.
- No `stage_configs_path` — the rollout deploy config is auto-generated
  from `pipeline_name` by `vLLMOmniHttpServer`.
- Use `omni_single_turn_agent` only for an omni model's autoregressive Talker
  stage when its rollout adapter must map model-native output to the Talker
  policy sequence. Standard text-token stages such as the Thinker can keep
  verl's `single_turn_agent`.
- No `--config-path/--config-name` — all config comes from CLI overrides
  on `verl_omni`'s `omni_trainer.yaml` defaults.
- The `"$@"` at the end lets callers override any field without editing
  the script (e.g. `bash run.sh trainer.total_epochs=10`).

### Sizing rollout memory in colocated sleep mode

In colocated training the rollout engine sleeps (level 1) while the actor
trains and re-maps its memory (`weights`, then `kv_cache`) every step, on
the same GPUs. Two footprint components matter, and they are controlled by
different knobs:

- **Steady-state KV cache** — pre-allocated at
  `gpu_memory_utilization × total` and self-limiting (a full pool preempts,
  it does not OOM). Raise it for long-response workloads that genuinely
  fill KV; lower it to widen the wake-up remap margin (audio and other
  encoder-heavy workloads see larger unbudgeted transients, so they need
  more margin than image/text-only ones at the same utilization).
- **Unbudgeted generation transients** — CUDA-graph capture pools (largest
  capture defaults to `min(2 × max_num_seqs, 512)`) and the in-flight
  multimodal envelope (encoder outputs retained for all concurrently
  admitted requests) sit on top of every budget. These are bounded by
  `max_num_seqs` and `cudagraph_capture_sizes`, **not** by
  `gpu_memory_utilization` or `max_num_batched_tokens`.

For encoder-heavy workloads (e.g. audio) with short responses, capping
concurrency keeps the transient off the memory ceiling at negligible
throughput cost:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256]
```

For long-response workloads that fill the KV pool, prefer keeping
concurrency high and tuning `gpu_memory_utilization` instead — preempting
KV is cheap relative to starving decode.

Reference:
[`examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh`](../../examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh)

### VeOmni backend (optional)

Install PyPI VeOmni **0.1.12** and its GPU kernels using the
[optional engine backends guide](../start/engine_backends.md).
The [Thinker GSPO recipe](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/README.md#veomni-full-parameter-thinker-training)
provides a complete launch example. Select the backend with:

```bash
python3 -m verl_omni.trainer.main_omni \
    model_engine=veomni \
    actor_rollout_ref.actor._target_=verl_omni.workers.config.omni.OmniVeOmniActorConfig \
    ...
```

Both FSDP and VeOmni resolve the same `OmniModelBase` adapter by
`(architecture, model_stage)`. Extend that adapter to support VeOmni; the
shared `OmniVeOmniEngine` owns engine registration and delegates model loading,
FSDP2/EP, optimization and checkpointing to verl. A new model does not need
another engine class.

#### Extend the training adapter

| Hook | Responsibility |
| --- | --- |
| `setup_veomni(model_config, engine_config)` | Opt in, validate supported settings before model loading, and install backend integrations such as weight-export handlers. |
| `prepare_veomni_inputs(model_inputs, micro_batch, model_config)` | Adapt packed inputs after verl's VeOmni transforms; defaults to passthrough. |
| `configure_veomni_trainable_params(module, model_config)` | Set trainable parameters after parallelization and before optimizer creation; defaults to no-op. |

The existing `prepare_model_inputs` replay hook runs afterwards on both
backends. Keep optional VeOmni imports inside the backend hooks. Adapters
without `setup_veomni` support fail before model loading; selecting VeOmni
does not imply that every registered architecture is supported. Do not
replace modules in `configure_veomni_trainable_params`: VeOmni already owns
their distributed layout. This hook does not run for forward-only reference
engines.

Use [`qwen3_omni/veomni.py`](../../verl_omni/pipelines/qwen3_omni/veomni.py)
as a model-specific reference:

- Packed prompt/response boundaries define modality masks, so placeholder
  tokens sampled into a response cannot consume prompt image features.
- Weight export expands fused gate/up and down tensors to Hugging Face
  per-expert weights, including EP rank offsets, for vLLM-Omni updates.
- Setup accepts policy-gradient Thinker training with packed text/image
  inputs and Ulysses size 1. It rejects direct-preference batches, unsupported
  stages and LoRA, including a nonempty `lora_adapter_path` with rank zero.
  Input preparation rejects audio/video features.
- Before creating the optimizer, the adapter freezes vision/audio encoders
  and rejects a loaded graph containing `talker`, `code2wav`, `code_predictor`
  or `has_talker=True`. VeOmni 0.1.12 constructs only the Thinker even when
  the checkpoint config enables speech; the guard detects changes to that
  behavior. These are the same excluded-module names used by the FSDP adapter.
- The version-sensitive `create_causal_mask` shim drops the obsolete
  `cache_position` keyword from VeOmni 0.1.12's generated GPU model when the
  Transformers signature no longer accepts it, validated with Transformers
  5.14.1. Compatible signatures are unchanged and other unknown arguments
  still raise. Recheck this shim when either dependency is upgraded.

#### Select native operators

Set VeOmni's operator fields under `actor_rollout_ref.actor.veomni`. The
Thinker launcher explicitly selects VeOmni 0.1.12's GPU defaults for the
Qwen3 operators, overriding verl's conservative eager defaults:

| VeOmni selector | Recipe default |
| --- | --- |
| `attn_implementation` | `flash_attention_2` |
| `moe_implementation` | `fused_triton` |
| `cross_entropy_loss_implementation` | `liger_kernel` |
| `rms_norm_implementation` | `liger_kernel` |
| `swiglu_mlp_implementation` | `liger_kernel` |
| `rotary_pos_emb_implementation` | `liger_kernel` |
| `load_balancing_loss_implementation` | `triton` |

The launcher makes reference selectors inherit the actor's values, including
CLI overrides; an explicit `actor_rollout_ref.ref.veomni.<selector>` override
still takes precedence. For example:

```bash
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh \
    actor_rollout_ref.actor.veomni.moe_implementation=fused_quack
```

`MOE_IMPL` and `ATTN_IMPL` are optional launcher conveniences for the same
fields. Use explicit `fused_triton` / `fused_quack` names instead of VeOmni's
deprecated `fused` alias. The field mapping to VeOmni's builder and
`OpsImplementationConfig` lives in verl's VeOmni engine; the omni adapter
adds no operator-selection mapping.

`actor_rollout_ref.model.use_fused_kernels=true` selects verl's RL output
protocol: the engine passes `return_log_probs=True`, temperature and
pre-shifted labels. It does not select MoE, norm or CE operators. VeOmni's
non-eager CE implementation computes chunked log-probabilities without
materializing the full logits tensor; `cross_entropy_loss_implementation=eager`
may still materialize logits.

#### Validate the VeOmni integration

Run both checks from the repository root through normal package initialization
with the installed VeOmni and pinned verl/vLLM-Omni stack. They need two GPUs
and use tiny random checkpoints without downloading the 30B model. Both use
the launcher's default FA2 / fused Triton / Liger operators; the backend check
uses eager operators on CPU only when generating its checkpoint fixture.

```bash
torchrun --standalone --nproc_per_node=2 \
    tests/special_e2e/check_qwen3_omni_veomni_backend.py

bash tests/special_e2e/run_gspo_qwen3_omni_thinker_veomni_smoke.sh
```

The backend check loads a speech-enabled config with extra Talker/codec
checkpoint keys and verifies a Thinker-only optimizer. It compares the actual
`use_fused_kernels` path with logits for log-probabilities and entropy at
temperatures 1.0 and 0.8, with images on one rank and text on the other. It
also checks two optimizer updates with EP=2 and exact agreement of exported
weights across ranks.

The V1 smoke exercises generation, actor/reference scoring, backward and
optimizer execution, full-weight rollout updates, and final validation over
two GSPO steps. It allows 1800 seconds for rollout startup because cold
FlashInfer kernel compilation can exceed the default timeout. Its random
arithmetic task may yield zero rewards; the backend check separately verifies
nonzero gradients. These checks validate training and weight-transfer mechanics,
not 30B-model convergence. Both scripts remain manual validation tools outside
the required `ci-e2e-omni` group.

## 6. Common pitfalls

These pitfalls are drawn from the Qwen3-Omni and MiniCPM-o adapters. Some are
model-specific — verify each against your own model's architecture.

- **`_no_split_modules`**: Must be set to the correct decoder layer class
  name in `configure_model` (e.g. `Qwen3OmniMoeThinkerTextDecoderLayer`).
  FSDP uses this hint for sharding granularity — a wrong name causes the
  entire module to be treated as a single leaf, defeating parameter sharding.

- **mrope**: Qwen-style multimodal models use 3-component position IDs
  (temporal, height, width) for mrope. HuggingFace's `get_rope_index` returns
  float32 position IDs that FSDP would bf16-round. Cast to `int64` in
  `configure_processor` (see `_get_rope_index_long` in
  `thinker_training_adapter.py:98-100`).

- **`dedup_pad_tokens`**: Bind on the processor for multimodal (image/video/
  audio) training to avoid double-expansion in AR mode. Both the HF processor
  and vLLM's `_apply_prompt_updates` expand the pad token, causing a mismatch.
  The `dedup_pad_tokens` helper collapses consecutive identical multimodal
  pad tokens before sending to vLLM-Omni.

- **tokenizer `chat_template.json`**: If the model checkpoint ships
  `chat_template.json` separately (not in `tokenizer_config.json`), load it
  in `configure_tokenizer` and assign it to `tokenizer.chat_template`.
  verl's dataset loader calls `tokenizer.apply_chat_template()` and will
  fail without a template.

- **Conditionally skipped encoders under FSDP2**: FSDP2 shards down to
  individual `nn.Embedding` and `nn.Linear` leaves, so a *sharded* tower that
  some micro-batches skip desyncs NCCL — the ranks that enter its collectives
  and the ranks that skip it disagree on the collective order, and the failure
  surfaces much later as a hang or a garbage gradient. Declare the subtree in
  `get_fsdp_ignored_module_names`. The skip must be genuine too: if media
  presence is not DP-balanced (verl balances token counts only), every rank
  still has to reach the same number of collectives.

- **Actor/rollout probability consistency**: Autoregressive codec policies may
  combine several codebook embeddings before predicting the selected token.
  Match actor, reference, rollout, and weight-sync dtypes, then verify selected
  token log-probabilities before training. Treat numerical comparisons as
  execution-consistency diagnostics, not evidence of output quality or bitwise
  agreement between different precision paths.
