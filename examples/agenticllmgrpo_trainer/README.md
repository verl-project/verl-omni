# Agentic LLM RPCO trainer

Last updated: 09/11/2026

This recipe runs RPCO stage-3 GRPO on the UniCoT Self-Reflection and Breakdown
datasets. The trainable agent LLM calls two frozen services:

1. `generate_image` produces a candidate image.
2. `judge_image` returns correctness, aesthetics, findings, and a
   `good_enough` decision.
3. The agent stops or rewrites the image prompt.

RPCO combines reflection, plan, format, tool, and result rewards from
`verl_omni.utils.reward_score.agentic_multidim_reward` (PR #412). Trajectories
come from `image_gen_tool_agent` + `OmniAgentLoopManager` (PR #409). Parquet
rows are built by `build_unicot_agentic_rl` (PR #411).

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
- `RPCO_W_REFLECT`, `RPCO_W_PLAN`, `RPCO_W_FORMAT`, `RPCO_W_TOOL`,
  `RPCO_W_RESULT`: per-dimension weights baked into parquet ground truth
  (`RPCO_W_TOOL_CALL` is accepted as an alias for `RPCO_W_TOOL`).
- `AGENTIC_VLLM_OMNI_URL` / `AGENTIC_VLLM_URL`: sidecar endpoints, forwarded
  to Hydra `agentic_image_gen.vllm_omni_url` / `vllm_url`.
- `AGENTIC_E2E_ROOT`: dump root for traj / images (`agentic_image_gen.e2e_root`).
- `TEST_FREQ` and `VAL_BEFORE_TRAIN`: periodic validation.
- `VAL_ROLLOUT_N`: greedy validation rollout count; defaults to 1.
- `REBUILD_UNICOT=0`: reuse existing train/val parquet.

Reward wiring (required by PR #412):

- `reward.reward_manager.name=naive` (text trajectory; not VisualRewardManager)
- `reward.custom_reward_function` → `agentic_multidim_reward.compute_score`

## Scope note

The RFC also proposed an external per-dimension HTTP protocol. This recipe does
not add that protocol: RPCO dimensions are computed locally by
`agentic_multidim_reward`.
