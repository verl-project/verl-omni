# Config Explanation

Last updated: 08/23/2026

VeRL-Omni builds on [verl](https://github.com/verl-project/verl) and reuses the
same Hydra config surface for shared RL trainer fields (`data`, FSDP actor /
optim, generic `trainer` / `ray_kwargs`, and so on).

For the full field-by-field reference of those **shared** options, see the
upstream verl documentation:

- **[Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)**

The rest of this page documents **VeRL-Omni-only** knobs: diffusion and omni
trainer entry points, algorithm / model / loss / rollout blocks that do not
exist (or differ) in upstream verl.

## Trainer entry points

VeRL-Omni composes configs from two primary trainer YAMLs under
[`verl_omni/trainer/config/`](https://github.com/verl-project/verl-omni/tree/main/verl_omni/trainer/config):

| Trainer | Config | Typical use |
|---------|--------|-------------|
| Diffusion | `diffusion_trainer.yaml` | Image / video / audio diffusion RL (FlowGRPO, MixGRPO, Diffusion-DPO, …) |
| Omni | `omni_trainer.yaml` | Omni-modality models (e.g. Qwen3-Omni GSPO / DPO); inherits verl `ppo_trainer` via Hydra `searchpath` |

Recipe scripts under `examples/` pass Hydra overrides on the command line.
Precedence is lowest to highest:

```text
trainer YAML defaults  →  recipe config (if any)  →  CLI overrides
```

Source of truth for defaults:

- Dataclasses under [`verl_omni/workers/config/`](https://github.com/verl-project/verl-omni/tree/main/verl_omni/workers/config) and [`verl_omni/trainer/config/algorithm.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/config/algorithm.py)
- Flattened reference dumps: `_generated_diffusion_trainer.yaml`, `_generated_diffusion_veomni_trainer.yaml`, `_generated_omni_trainer.yaml` (auto-generated; do not edit)

---

## Diffusion trainer (`diffusion_trainer.yaml`)

### `algorithm` — `DiffusionAlgoConfig`

```yaml
algorithm:
  _target_: verl_omni.trainer.config.DiffusionAlgoConfig
  trainer_type: policy_gradient
  sample_source: online
  adv_estimator: ${oc.select:actor_rollout_ref.model.algorithm,flow_grpo}
  norm_adv_by_std_in_grpo: true
  global_std: true
  old_policy_decay_schedule: copy
  old_policy_decay: null
  old_policy_update_interval: 1
  timestep_fraction: 1.0
  adv_mode: continuous
  paired_preference: false
  rollout_correction: { ... }   # mirrors upstream RolloutCorrectionConfig
```

- `algorithm.trainer_type`: Trainer loop. `policy_gradient` (FlowGRPO, MixGRPO, Flow-DPPO, …) or `direct_preference` (DPO, DiffusionNFT, AWM).
- `algorithm.sample_source`: `online` (rollout + reward engines) or `offline` (actor-only, precomputed batches).
- `algorithm.adv_estimator`: Advantage estimator name; defaults to `actor_rollout_ref.model.algorithm` (e.g. `flow_grpo`).
- `algorithm.norm_adv_by_std_in_grpo`: Normalize advantages by within-group std (GRPO-style).
- `algorithm.global_std`: Use a global (cross-group) std for advantage normalization.
- `algorithm.old_policy_decay_schedule`: DiffusionNFT old-policy EMA schedule. One of `copy`, `linear_to_0_5`, `delayed_linear_to_0_999`.
- `algorithm.old_policy_decay`: Fixed old-policy EMA decay in `[0, 1]`. When set, overrides `old_policy_decay_schedule`.
- `algorithm.old_policy_update_interval`: DiffusionNFT optimizer steps between old-policy adapter refreshes (must be `> 0`).
- `algorithm.timestep_fraction`: Fraction of rollout timesteps used for forward-process training, in `(0, 1]`.
- `algorithm.adv_mode`: Advantage mapping before reward-probability scaling. One of `continuous`, `positive_only`, `negative_only`, `one_only`, `binary`.
- `algorithm.paired_preference`: `true` for pair-based algorithms (e.g. offline DPO); doubles actor batch size and disables shuffle.
- `algorithm.rollout_correction.*`: Experimental IS / RS correction. Schema mirrors upstream verl; see {doc}`../algo/rollout_correction` and [verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html).

### `actor_rollout_ref.model` — `DiffusionModelConfig`

```yaml
actor_rollout_ref:
  model:
    _target_: verl_omni.workers.config.diffusion.DiffusionModelConfig
    model_type: diffusion_model
    path: ~/models/Qwen/Qwen-Image
    algorithm: flow_grpo
    tokenizer_path: null
    config_path: null
    transformer_subfolder: transformer
    attn_backend: _flash_3_varlen_hub
    enable_gradient_checkpointing: True
    lora_rank: 0
    lora_alpha: 64
    lora_init_weights: gaussian
    target_modules: all-linear
    target_parameters: null
    exclude_modules: null
    lora_adapter_path: null
    policy_state_adapters: ["default"]
    lora_dtype: null
    fsdp_layer_prefixes: ["transformer_blocks."]
    # pipeline / algo mirror rollout (see below)
```

- `actor_rollout_ref.model.model_type`: Dispatch key; must be `diffusion_model` for the diffusion agent loop.
- `actor_rollout_ref.model.path`: HuggingFace / local diffusion pipeline root.
- `actor_rollout_ref.model.algorithm`: RL algorithm name (also drives default `diffusion_loss.loss_mode` / `adv_estimator`). Examples: `flow_grpo`, `mix_grpo`, `flow_dppo`, `diffusion_nft`, `dpo`, `dance_grpo`.
- `actor_rollout_ref.model.tokenizer_path`: Optional tokenizer path if not under `path` (falls back to `<path>/tokenizer` or `path`).
- `actor_rollout_ref.model.config_path`: Optional transformer config path. If null, backends use `<path>/<transformer_subfolder>`.
- `actor_rollout_ref.model.transformer_subfolder`: Subfolder with diffusion transformer weights/config (default `transformer`).
- `actor_rollout_ref.model.attn_backend`: Diffusers attention backend. One of `native`, `_native_npu`, `_flash_3_varlen_hub`. Must stay consistent with `rollout.rollout_attn_backend`.
- `actor_rollout_ref.model.lora_rank`: LoRA rank; `> 0` enables LoRA.
- `actor_rollout_ref.model.lora_alpha`: LoRA scaling factor.
- `actor_rollout_ref.model.lora_init_weights`: LoRA init method (default `gaussian`).
- `actor_rollout_ref.model.target_modules`: LoRA targets (`all-linear` or a list of module name patterns).
- `actor_rollout_ref.model.target_parameters`: Optional list of `nn.Parameter` names for LoRA.
- `actor_rollout_ref.model.exclude_modules`: Modules excluded from LoRA.
- `actor_rollout_ref.model.lora_adapter_path`: Pre-trained LoRA adapter for continued training.
- `actor_rollout_ref.model.policy_state_adapters`: Named LoRA policy states required by the algorithm. `"reference"` disables adapters.
- `actor_rollout_ref.model.lora_dtype`: Convert LoRA params to a dtype (e.g. `fp32`, `bf16`); `null` = no conversion.
- `actor_rollout_ref.model.fsdp_layer_prefixes`: FSDP layer name prefixes for LoRA layered summon (default `["transformer_blocks."]`).
- `actor_rollout_ref.model.pipeline` / `algo`: Mirrored from `actor_rollout_ref.rollout.pipeline` / `algo` via `oc.select`; prefer overriding the rollout copies.

### `actor_rollout_ref.actor` — diffusion actor / loss

```yaml
actor_rollout_ref:
  actor:
    _target_: verl_omni.workers.config.diffusion.DiffusionActorConfig  # or FSDP / VeOmni subclass
    strategy: fsdp   # fsdp | fsdp2 | veomni
    diffusion_loss:
      _target_: verl_omni.workers.config.diffusion.DiffusionLossConfig
      loss_mode: ${oc.select:actor_rollout_ref.model.algorithm,flow_grpo}
      clip_ratio: 0.0001
      adv_clip_max: 5.0
      mix_beta: 0.5
      ref_kl_coef: 0.0
      adaptive_weight_min: 1e-5
      dpo_beta: 2000.0
      kl_mask_threshold: 1e-5
      add_kl_coefficient: true
    loss_scale_factor: null
    use_kl_loss: false
    kl_loss_coef: 0.001
    use_distill_loss: false
    distill_loss_mode: distill_kl
    distill_loss_coef: 1.0
```

#### `diffusion_loss` — `DiffusionLossConfig`

- `actor_rollout_ref.actor.diffusion_loss.loss_mode`: Loss registry key. One of `flow_grpo`, `flow_dppo`, `grpo_guard`, `diffusion_nft`, `dpo`, `dance_grpo`, `distill_kl`, `distill_fm_mse`.
- `actor_rollout_ref.actor.diffusion_loss.clip_ratio`: PPO-style clip ratio for diffusion policy loss (FlowGRPO default is very small, e.g. `1e-4`).
- `actor_rollout_ref.actor.diffusion_loss.adv_clip_max`: Max absolute advantage before computing the policy loss (must be `> 0`).
- `actor_rollout_ref.actor.diffusion_loss.mix_beta`: DiffusionNFT β for positive / implicit-negative prediction mixing (must be `> 0`).
- `actor_rollout_ref.actor.diffusion_loss.ref_kl_coef`: DiffusionNFT prediction-space reference MSE coefficient.
- `actor_rollout_ref.actor.diffusion_loss.adaptive_weight_min`: DiffusionNFT minimum adaptive denominator for x0 reconstruction losses (must be `> 0`).
- `actor_rollout_ref.actor.diffusion_loss.dpo_beta`: DPO inverse temperature for pairwise flow-matching preference loss.
- `actor_rollout_ref.actor.diffusion_loss.kl_mask_threshold`: Flow-DPPO divergence threshold for masking high-KL updates (must be `> 0`).
- `actor_rollout_ref.actor.diffusion_loss.add_kl_coefficient`: Whether Flow-DPPO normalizes mean drift by the scheduler SDE noise scale.

#### Actor-level extras

- `actor_rollout_ref.actor.loss_scale_factor`: Optional global scale on the diffusion loss; `null` disables.
- `actor_rollout_ref.actor.use_kl_loss` / `kl_loss_coef`: Enable KL against the reference policy (FlowGRPO).
- `actor_rollout_ref.actor.use_distill_loss`: Enable teacher-anchored online policy distillation.
- `actor_rollout_ref.actor.distill_loss_mode`: `distill_kl` or `distill_fm_mse`.
- `actor_rollout_ref.actor.distill_loss_coef`: Distillation loss coefficient.
- `distillation.enabled` / `distillation.teacher_models.<name>.{key,model_path,world_size}` / `distillation.teacher_key` / `distillation.{n_gpus_per_node,nnodes}`: Frozen teachers (routed per sample, colocated or on their own pool) that produce the `teacher_*` batch keys the distillation losses consume — see [Diffusion On-Policy Distillation](../algo/diffusion_opd.md).
- `actor_rollout_ref.actor.rollout_correction.*`: Per-actor mirror of `algorithm.rollout_correction` (used when `bypass_mode=True` for per-step RS inside `diffusion_loss`).

Shared PPO / FSDP / optim fields (`ppo_mini_batch_size`, `ppo_epochs`, `optim.lr`, `fsdp_config`, …) follow upstream verl — see the [verl Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html).

VeOmni engine path (`strategy=veomni`) adds `veomni_config` / VeOmni optimizer fields; see {doc}`../start/install` and the `run_*_veomni.sh` recipes.

### `actor_rollout_ref.rollout` — `DiffusionRolloutConfig`

Diffusion-specific blocks sit under `pipeline`, `algo`, and `val_kwargs`. Several engine knobs are shared with verl vLLM rollout but have diffusion defaults.

#### Pipeline — `DiffusionPipelineConfig`

```yaml
actor_rollout_ref:
  rollout:
    pipeline:
      height: 512
      width: 512
      num_inference_steps: 10
      true_cfg_scale: 1.0
      max_sequence_length: 512
      guidance_scale: null
      num_frames: 1
      task: null
```

- `actor_rollout_ref.rollout.pipeline.height` / `width`: Image / video spatial size for training rollout.
- `actor_rollout_ref.rollout.pipeline.num_inference_steps`: Denoising steps during training generation.
- `actor_rollout_ref.rollout.pipeline.true_cfg_scale`: True classifier-free guidance scale; values `> 1.0` enable CFG with a negative prompt (e.g. Qwen-Image).
- `actor_rollout_ref.rollout.pipeline.max_sequence_length`: Max text-encoder token length for prompt encoding.
- `actor_rollout_ref.rollout.pipeline.guidance_scale`: Distilled guidance scale for models with guidance embeddings; `null` disables.
- `actor_rollout_ref.rollout.pipeline.num_frames`: Wan2.2 (and similar) video frame count (`81` ≈ 3s at 24 fps; image models keep `1`).
- `actor_rollout_ref.rollout.pipeline.task`: Optional task label forwarded to the pipeline's request contract (vLLM-Omni reads it as the request `task`); values are pipeline-specific (e.g. MiniMax-H3: `t2va` / `fl2va` / `ref2va`), `null` lets the engine infer it.
- `actor_rollout_ref.rollout.pipeline.output_type`: Pipeline output modality (dataclass default `image`).

#### Rollout algo — `DiffusionRolloutAlgoConfig`

```yaml
actor_rollout_ref:
  rollout:
    algo:
      noise_level: 1.0
      sde_type: sde
      sde_window_size: null
      sde_window_range: null
      sample_strategy: random      # MixGRPO
      iters_per_group: 1          # MixGRPO progressive
      sde_window_seed: 0          # MixGRPO random
```

- `actor_rollout_ref.rollout.algo.noise_level`: Magnitude of SDE noise inside the active window (larger → more diversity, often lower quality).
- `actor_rollout_ref.rollout.algo.sde_type`: SDE variant: `sde` or `cps`.
- `actor_rollout_ref.rollout.algo.sde_window_size`: Active SDE window length in denoising steps; `null` = all steps.
- `actor_rollout_ref.rollout.algo.sde_window_range`: `[start, end)` envelope for random window-start sampling; `null` = entire trajectory.
- `actor_rollout_ref.rollout.algo.sample_strategy`: MixGRPO sliding-window scheduler: `random` (fresh window each step) or `progressive` (advance by `sde_window_size` every `iters_per_group` iterations).
- `actor_rollout_ref.rollout.algo.iters_per_group`: Training iterations spent at each progressive window position (must be `> 0` when `sample_strategy=progressive`).
- `actor_rollout_ref.rollout.algo.sde_window_seed`: Base seed for random window draws (`sample_strategy=random`).

#### Validation sampling — `val_kwargs`

```yaml
actor_rollout_ref:
  rollout:
    val_kwargs:
      n: 1
      pipeline: { ... }   # same fields as rollout.pipeline; defaults often higher (e.g. num_inference_steps: 50)
      algo:
        noise_level: 0.0  # typically ODE (no SDE noise) at eval
        ...
```

- `actor_rollout_ref.rollout.val_kwargs.n`: Generations per prompt during validation.
- `actor_rollout_ref.rollout.val_kwargs.pipeline.*`: Validation pipeline overrides (often more steps / ODE).
- `actor_rollout_ref.rollout.val_kwargs.algo.*`: Validation SDE settings (`noise_level: 0.0` by default).

#### Diffusion rollout engine knobs

- `actor_rollout_ref.rollout.name`: Rollout backend; diffusion recipes use `vllm_omni`.
- `actor_rollout_ref.rollout.mode`: Must be `async` (`sync` was removed).
- `actor_rollout_ref.rollout.n`: Samples per prompt (FlowGRPO group size; usually `> 1`).
- `actor_rollout_ref.rollout.seed`: Base seed for deterministic training rollout RNG. Per-step base is `seed + global_step - 1`; `null` disables seeding.
- `actor_rollout_ref.rollout.rollout_attn_backend`: vLLM-Omni diffusion attention backend. One of `FLASH_ATTN`, `FLASH_ATTN_HUB`, `FLASH_ATTN_3_HUB`, `TORCH_SDPA`. Must match `model.attn_backend` (default `FLASH_ATTN_3_HUB` ↔ `_flash_3_varlen_hub`).
- `actor_rollout_ref.rollout.step_execution`: When `true`, run the registered pipeline in step-execution (continuous / stepwise batching) mode. See {doc}`../start/rollout_batching`.
- `actor_rollout_ref.rollout.max_num_seqs`: Max concurrent sequences in the engine; also the request-level batching capacity knob.
- `actor_rollout_ref.rollout.gpu_memory_utilization`: Fraction of GPU memory for the vLLM-Omni cache.
- `actor_rollout_ref.rollout.calculate_log_probs`: Log rollout log-probs for debugging.
- `actor_rollout_ref.rollout.rollout_adapter`: Named adapter for generation: `default` or `old`.
- `actor_rollout_ref.rollout.agent.default_agent_loop`: Default `diffusion_single_turn_agent`.
- `actor_rollout_ref.rollout.engine_kwargs.vllm_omni`: Extra vLLM-Omni engine kwargs (dict).

### `trainer` — diffusion-only dump / video knobs

These sit on the diffusion trainer YAML (in addition to shared verl trainer fields):

- `trainer.video_fps`: FPS for videos written to `rollout_data_dir` / `validation_data_dir` and logged to W&B (image runs ignore this).
- `trainer.rollout_data_save_freq`: Dump train rollout every N steps (`1` = every step, `<= 0` = never).
- `trainer.rollout_data_max_samples` / `validation_data_max_samples`: Cap samples dumped per train / val run (`null` = all).
- `trainer.use_v1`: Use the V1 trainer (TransferQueue + ReplayBuffer). When `false`, legacy v0 diffusion trainer.
- `trainer.v1.*`: V1 mode / sampler / async placeholders (`trainer_mode`, `max_off_policy_threshold`, …). See {doc}`../start/diffusion_v1`.

### `reward` — visual reward manager

Diffusion recipes compose `reward@reward: reward` (`verl_omni/trainer/config/reward/reward.yaml`):

- `reward.num_workers`: Parallel reward-manager workers.
- `reward.custom_reward_function.path` / `name`: Single custom score function.
- `reward.reward_functions`: Multi-reward dict (`{name: {path, name, weight}}`); mutually exclusive with `custom_reward_function`.
- `reward.aggregation`: Multi-reward aggregation (`weighted_sum` only).
- `reward.reward_manager`: Defaults to `VisualRewardManager` from `pkg://verl_omni.reward_loop.reward_manager`.
- `reward.reward_model.*`: Optional model-based RM (resource pool, rollout engine knobs). See {doc}`../algo/async_reward` and {doc}`../start/http_scorer`.

---

## Omni trainer (`omni_trainer.yaml`)

`omni_trainer.yaml` inherits verl `ppo_trainer` via Hydra `searchpath`, then overlays omni model / actor / algorithm defaults. Shared PPO / FSDP / rollout fields stay as in [verl Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html); below are the omni deltas.

### `algorithm` — `OmniAlgoConfig`

```yaml
algorithm:
  _target_: verl_omni.trainer.config.OmniAlgoConfig
  trainer_type: direct_preference
  sample_source: offline
  paired_preference: true
  adv_estimator: dpo
  norm_adv_by_std_in_grpo: true
  global_std: true
```

- `algorithm.trainer_type`: `policy_gradient` (online GSPO / GRPO / PPO) or `direct_preference` (offline / online DPO).
- `algorithm.sample_source`: `online` or `offline`.
- `algorithm.paired_preference`: `true` for pair-based algorithms such as offline DPO.
- `algorithm.adv_estimator`: Placeholder for shared trainer utilities (default `dpo` on the omni trainer YAML).
- `algorithm.norm_adv_by_std_in_grpo` / `global_std`: Same meaning as diffusion; mainly relevant for policy-gradient omni runs.

### `actor_rollout_ref.model` — `OmniModelConfig`

```yaml
actor_rollout_ref:
  model:
    _target_: verl_omni.workers.config.omni.OmniModelConfig
    model_type: omni_model
    path: ???
    hf_config_path: null
    tokenizer_path: null
    architecture: null
    model_stage: thinker
    hf_config_name: null
    enable_gradient_checkpointing: True
    enable_activation_offload: False
    use_remove_padding: True
    lora_rank: 0
    lora_alpha: 16
    lora_init_weights: gaussian
    target_modules: all-linear
    policy_state_adapters: ["default"]
    fsdp_layer_prefixes: []
    max_image_tokens: null
    max_audio_tokens: null
    max_video_tokens: null
```

- `actor_rollout_ref.model.model_type`: Dispatch key; must be `omni_model`.
- `actor_rollout_ref.model.path`: HuggingFace / local omni checkpoint root.
- `actor_rollout_ref.model.hf_config_path`: Optional HF config path if different from `path`.
- `actor_rollout_ref.model.tokenizer_path`: Optional tokenizer path (defaults to `<path>/tokenizer` or `path`).
- `actor_rollout_ref.model.architecture`: HF `architectures[0]`; auto-detected from `config.json` if unset.
- `actor_rollout_ref.model.model_stage`: Which stage to train: `thinker`, `talker`, or `all`.
- `actor_rollout_ref.model.hf_config_name`: Sub-config key for the trainable component (e.g. `thinker_config`, `talker_config`).
- `actor_rollout_ref.model.override_config`: Dict merged into HF config load (e.g. `attn_implementation`).
- `actor_rollout_ref.model.enable_activation_offload` / `use_remove_padding`: Memory / packing flags for the FSDP actor.
- `actor_rollout_ref.model.lora_*` / `target_modules` / `policy_state_adapters` / `fsdp_layer_prefixes`: Same LoRA roles as diffusion (defaults differ slightly, e.g. `lora_alpha: 16`).
- `actor_rollout_ref.model.use_liger` / `use_fused_kernels`: Unsupported by omni FSDP/FSDP2 and must remain `false`; enabling either fails before model loading. The adjacent `fused_kernel_options` / `tiled_mlp` fields are backend-specific.
- `actor_rollout_ref.model.max_image_tokens` / `max_audio_tokens` / `max_video_tokens`: Multimodal token budgets (`null` = unset).
- `actor_rollout_ref.model.lora` / `mtp`: Megatron-style LoRA block and multi-token prediction (speculative decoding) configs; see the YAML comments in `omni/model/omni_model.yaml`.

### `actor_rollout_ref.actor` — `OmniActorConfig` / `omni_loss`

```yaml
actor_rollout_ref:
  rollout:
    name: vllm_omni
  actor:
    _target_: verl_omni.workers.config.omni.OmniActorConfig
    trainer_type: ${algorithm.trainer_type}
    omni_loss:
      _target_: verl_omni.workers.config.omni.OmniLossConfig
      loss_mode: dpo
      beta: 0.1
      label_smoothing: 0.0
      loss_type: sigmoid
      average_log_prob: false
      refer_model_precision: bfloat16
    shuffle: false
```

Which loss block is active depends on `algorithm.trainer_type`:

| `trainer_type` | Loss config used |
|----------------|------------------|
| `policy_gradient` | Upstream verl `actor_rollout_ref.actor.policy_loss` (+ clip / KL fields). `omni_loss` is **not** read. |
| `direct_preference` | `actor_rollout_ref.actor.omni_loss` (`OmniLossConfig` → `OmniDPOLoss`). |

#### `omni_loss` — `OmniLossConfig` (direct preference only)

- `actor_rollout_ref.actor.omni_loss.loss_mode`: Preference loss registry key. Currently only `dpo`.
- `actor_rollout_ref.actor.omni_loss.beta`: DPO inverse temperature β. Typical AR DPO values ~`0.01`–`0.5` (default `0.1`). Must be `> 0`.
- `actor_rollout_ref.actor.omni_loss.label_smoothing`: Label smoothing for sigmoid DPO (cDPO). `0.0` disables; ignored when `loss_type=ipo`.
- `actor_rollout_ref.actor.omni_loss.loss_type`: `sigmoid` (standard DPO) or `ipo` (identity preference optimization).
- `actor_rollout_ref.actor.omni_loss.average_log_prob`: If `true`, average token log-probs before the pairwise margin; if `false`, sum (TRL default).
- `actor_rollout_ref.actor.omni_loss.refer_model_precision`: Dtype for the frozen reference policy during ref log-prob (`bfloat16`, `float32`, …). Trainable policy dtype stays under `actor.fsdp_config.model_dtype`.

### `trainer.v1`

- `trainer.v1.trainer_mode`: Omni recipes set `omni_sync` (vs diffusion `sync` / async placeholders).

---

## Where to look next

- Shared PPO / FSDP / rollout knobs — [verl Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
- Diffusion algorithm pages — {doc}`../algo/flowgrpo`, {doc}`../algo/mixgrpo`, {doc}`../algo/flowdppo`, {doc}`../algo/diffusion_dpo`, {doc}`../algo/diffusionnft`, {doc}`../algo/grpo_guard`
- Rollout batching / step execution — {doc}`../start/rollout_batching`
- Rollout correction — {doc}`../algo/rollout_correction`
- Profiler — {doc}`../perf/profiler`
- Model catalogue and example scripts — {doc}`../start/models`
