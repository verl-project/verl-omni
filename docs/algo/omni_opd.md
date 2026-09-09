# Qwen3-Omni On-Policy Distillation

Last updated: 09/09/2026.

## Background

On-policy distillation (OPD) trains a student on states sampled from the
student's own policy, with a teacher providing dense supervision at those
states. Compared with distilling on teacher-generated data, this removes the
train/inference state mismatch: the teacher advises on the trajectories the
student actually visits. Compared with reward-only RL, the supervision is dense
and continuous instead of a sparse outcome score. For a more detailed
description of OPD, see the [verl OPD documentation](https://verl.readthedocs.io/en/latest/algo/opd.html).

For autoregressive omni models (Qwen3-Omni Thinker), a rollout is a token
sequence sampled over image / video / audio / text prompts. The frozen teacher
replays the student's full `prompt + response` sequence in a single forward
pass and returns the teacher's log-probability of every student token. The
`distillation_loss` then moves the student toward the teacher, token by token —
the AR analog of matching transition means in diffusion OPD.

Two ways to consume the teacher signal (mirroring verl's
`distillation.distillation_loss`):

- `use_policy_gradient=true` — the per-token distillation signal enters the
  policy-gradient reward, as in the
  [Thinking Machines on-policy distillation recipe](https://thinkingmachines.ai/blog/on-policy-distillation/).
  Recommended with the sampled estimators `loss_mode=kl` / `k1`.
- `use_policy_gradient=false` — the distillation loss is backpropagated
  directly as a supervised loss, as in [GKD](https://arxiv.org/abs/2306.13649).
  Recommended with `loss_mode=k3` or `forward_kl_topk`.

Unlike diffusion OPD, the teacher is **not** fused into the actor worker. AR
teachers reuse the rollout infrastructure: each teacher is one or more
`vllm_omni` async-server replicas — the same engine class as the student's
rollout, with the full omni processor pipeline (vision / audio towers, chat
template, multimodal preprocessing) — so multimodal student rollouts are
scored exactly as they were generated.

## Teacher Runtime

Teacher lifecycle and scoring are handled by verl's core teacher loop
(`MultiTeacherModelManager`); verl-omni plugs the `vllm_omni` engine into it:

1. **Route.** The batch column named by `teacher_key` (default `data_source`)
   maps every sample to a teacher. A single-teacher setup skips the column; a
   missing column or an unmatched key raises instead of mis-routing.
2. **Place.** The teacher resource pool (`distillation.n_gpus_per_node ×
   distillation.nnodes` GPUs) is split into one sub-pool per teacher according
   to `num_replicas × per_replica_world_size`, where `per_replica_world_size =
   tensor_model_parallel_size × data_parallel_size × pipeline_model_parallel_size`
   of the teacher's `inference` config. Each replica is a `vLLMOmniReplica`
   (registered under the `vllm_omni` name in verl's `RolloutReplicaRegistry`)
   running a full `vllm_omni` HTTP server.
3. **Score, streamed with the rollout.** Teacher computation overlaps the
   student rollout through the agent loop: as each student sequence completes,
   its token ids together with its multimodal inputs (`images` / `videos` /
   `audios` plus processor kwargs) are sent to the teacher server with
   `prompt_logprobs` and `max_tokens=1` — a pure forward pass, so temperature
   is ignored (always 1.0). The AR strategy's `extract_prompt_logprobs`
   collects the teacher's per-token log-probabilities (top-`k` when the loss
   needs them) into the response's extra fields.
4. **Reassemble.** The agent loop pads the returned `teacher_ids` /
   `teacher_logprobs` to the batch's prompt / response widths and merges them
   into the training batch. The distillation loss in the actor update reads
   them alongside the student's own log-probs.

### Omni plumbing

What verl-omni adds on top of verl's teacher loop:

- `verl_omni/trainer/config/omni_trainer.yaml` overrides the teacher
  `_target_` so each teacher entry materializes as
  `OmniDistillationTeacherModelConfig`
  (`verl_omni/workers/config/omni/distillation.py`) instead of verl's
  text-only default. The subclass accepts `inference.name == "vllm_omni"` and,
  for top-k losses, seeds / validates
  `inference.engine_kwargs.vllm_omni.max_logprobs` against
  `distillation.distillation_loss.topk`.
- The teacher replica converts its model config to `OmniModelConfig` inside
  the server process. `engine_kwargs.vllm_omni.pipeline_name` (e.g.
  `"qwen3_omni_moe"`) selects the rollout pipeline adapter — the same deploy
  config, HF overrides and stage layout the student rollout uses — and
  `engine_kwargs.vllm_omni.output_mode="ar"` selects the AR strategy that
  knows how to extract prompt log-probs.

## Configuration Parameters

The `distillation.*` group follows verl's on-policy distillation config; the
omni-specific surface is the teacher's `inference` block pointed at
`vllm_omni`.

### `distillation.enabled` (bool)

Whether on-policy distillation is enabled. Default: `false`. When `true`, the
teacher pool is built and every student rollout is scored by the teachers
during rollout, before the actor update.

### `distillation.nnodes` / `distillation.n_gpus_per_node` (int)

Size of the teacher resource pool: `n_gpus_per_node × nnodes` GPUs. Defaults
`0` / `0`. A single teacher auto-fills the whole pool
(`num_replicas = pool_size // per_replica_world_size`); multiple teachers must
set their parallelism so the per-teacher footprints sum to the pool size, or
`DistillationConfig.__post_init__` raises.

Example: pool = 16 GPUs, `tensor_model_parallel_size=2` → 8 replicas of the
teacher, scored round-robin.

### `distillation.teacher_key` (str)

Column used to route samples to teachers in multi-teacher setups. Default:
`"data_source"`. Ignored for a single teacher.

### `distillation.teacher_models.<name>.model_path` (str)

Path to the frozen teacher checkpoint. **Required.**

**Constraint:** teacher and student must share the same tokenizer (same model
family). Cross-vocabulary distillation is not supported.

### `distillation.teacher_models.<name>.inference.*` (RolloutConfig)

The teacher's rollout-engine config. The omni-relevant keys:

- `name=vllm_omni` — **required**; selects the omni replica / server.
- `tensor_model_parallel_size` / `data_parallel_size` /
  `pipeline_model_parallel_size` — per-replica parallelism.
- `prompt_length` / `response_length` — the **student's** prompt / response
  lengths; the teacher scores the student's prompt and response in one forward
  pass, so `max_model_len` must be at least `prompt_length + response_length +
  1` (validated at startup).
- `gpu_memory_utilization`, `max_model_len`, and other standard rollout keys.
- `+inference.engine_kwargs.vllm_omni.output_mode="ar"` — **required**; the
  teacher scores token sequences, so it always runs the AR strategy.
- `+inference.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe"` —
  selects the omni rollout pipeline adapter matching the teacher model.
- `+inference.engine_kwargs.vllm_omni.max_logprobs=<int>` — only needed for
  top-k losses (`loss_mode=forward_kl_topk`); auto-seeded to
  `distillation_loss.topk` when unset, and validated `>= topk`.

### `distillation.teacher_models` naming pitfall

Same as verl: when adding more named teachers, the default `teacher_model`
entry is silently popped — rename it (e.g. `teacher_model1`) instead of
keeping it alongside others. See the
[diffusion OPD multi-teacher notes](diffusion_opd.md#multi-teacher) for the
routing semantics.

### `distillation.distillation_loss.*`

- `loss_mode` — `kl`, `k1`, `abs`, `mse`, `k2`, `low_var_kl`, `k3` (sampled
  estimators over the student's tokens) or `forward_kl_topk` (full top-k
  forward KL).
- `use_policy_gradient` (bool) — see Background. The validated Qwen3-Omni
  recipe uses `loss_mode=kl` + `use_policy_gradient=true`.
- `use_task_rewards` (bool) / `distillation_loss_coef` — combine the task
  reward with the distillation signal (policy-gradient mode) or weight the
  distillation term against the main loss (supervised mode).
- `topk` (int) — top-k size for `forward_kl_topk`.

## Usage

Supported scope: the omni V1 sync trainer (`verl_omni.trainer.main_omni`) with
`vllm_omni` rollout in AR mode and a `vllm_omni` teacher from the same model
family as the student. Validated on 2× Ascend 910C machines: student
rollout/actor on 16 GPUs of node 1, teacher on 16 GPUs of node 2 — the teacher
runs on its own node with `distillation.nnodes=1`,
`distillation.n_gpus_per_node=16`.

A complete runnable recipe is
[`examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1_opd_npu.sh`](../examples/gspo_trainer/README.md#mmk12-on-policy-distillation-opd):
GSPO + LoRA training of Qwen3-Omni-30B-A3B-Instruct (noise-perturbed student)
distilling from the original model on MMK12, where OPD converges faster than
the self-training baseline. Runtime dependencies on all Ray worker nodes:
`math-verify` (reward) and `qwen-vl-utils` (multimodal processing).

Minimal config:

```yaml
distillation:
  enabled: true
  nnodes: 1
  n_gpus_per_node: 16
  teacher_models:
    teacher_model:
      model_path: /path/to/Qwen3-Omni-30B-A3B-Instruct
      inference:
        name: vllm_omni
        tensor_model_parallel_size: 2
        gpu_memory_utilization: 0.6
        max_model_len: 16640
        prompt_length: 4160        # student prompt length
        response_length: 12288     # student response length
        engine_kwargs:
          vllm_omni:
            output_mode: ar
            pipeline_name: qwen3_omni_moe
  distillation_loss:
    loss_mode: kl
    use_policy_gradient: true
```

Multi-teacher and colocated-teacher setups follow verl's `distillation.*`
semantics (see the [diffusion OPD guide](diffusion_opd.md)); only the
`inference.*` block above is omni-specific.

## Metrics

- `distillation/loss` — the distillation loss over valid response tokens.
  Under pure distillation it should be clearly positive at step one (the
  teacher's weights differ from the student's) and decrease as the student
  matches the teacher.
- `distillation/loss_min` / `distillation/loss_max` — per-batch range of the
  per-token distillation loss over valid response tokens, useful to spot
  outlier sequences.
