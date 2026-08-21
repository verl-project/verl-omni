# Agentic LLM RPCO trainer

Last updated: 08/21/2026

This recipe runs RPCO stage-3 GRPO on the UniCoT Self-Reflection and Breakdown
datasets. The trainable agent LLM calls two frozen services:

1. `generate_image` produces a candidate image.
2. `judge_image` returns correctness, aesthetics, findings, and a
   `good_enough` decision.
3. The agent stops or rewrites the image prompt.

RPCO combines reflection, plan, format, tool-call, and result rewards from
`verl_omni.utils.reward_score.agentic_multidim_reward`.

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

The defaults listen on `127.0.0.1:8092` and `127.0.0.1:8093`. Override
`IMAGE_GEN_MODEL`, `JUDGE_IMAGE_MODEL`, or the corresponding host/port
variables when needed.

## Run stage 3

The launcher builds real UniCoT parquet files on first use and then starts
GRPO:

```bash
CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 TOTAL_STEPS=200 \
  bash examples/agenticllmgrpo_trainer/agent_llm/run_agentic_rpco.sh
```

By default the builder uses Hugging Face cache directories for:

- `Fr0zencr4nE/UniCoT-Self-Reflection-6K`
- `Fr0zencr4nE/UniCoT-Breakdown-3K`

Set `UNICOT_REFLECTION_DIR` and `UNICOT_BREAKDOWN_DIR` to explicit snapshot
directories when the datasets live elsewhere. The full parsed dataset is used
unless `UNICOT_TRAIN_SIZE` or `UNICOT_VAL_SIZE` is set for a smoke run.

Useful controls:

- `RPCO_INIT_CKPT`: initialize from a stage-1 agent checkpoint.
- `RPCO_W_REFLECT`, `RPCO_W_PLAN`, `RPCO_W_FORMAT`, `RPCO_W_TOOL`,
  `RPCO_W_RESULT`: per-dimension weights (`RPCO_W_TOOL_CALL` is accepted as
  an alias).
- `TEST_FREQ` and `VAL_BEFORE_TRAIN`: periodic validation.
- `VAL_ROLLOUT_N`: greedy validation rollout count; defaults to 1.
- `AGENTIC_VAL_VIZ`: fixed cafe-poster reflect/plan visualization; defaults
  to 1.
- `REBUILD_UNICOT=1`: rebuild the train/validation parquet.

Validation mirrors reward summaries under `val_agentic_reward/*`. The fixed
holdout is accumulated in the `val/generations` and
`val/generations_plan` W&B tables.

## Scope note

The RFC also proposed an external per-dimension HTTP protocol returning
`{"dimension", "score", "metadata"}`. This recipe does not add that protocol:
the reference implementation has no compatible server or client contract.
RPCO dimensions are computed locally by `agentic_multidim_reward`.
