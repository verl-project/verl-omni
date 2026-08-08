(agentic_rpco)=
# Multi-Turn Agentic Reflection–Plan Co-Optimization (RPCO)

Last updated: 08/05/2026

This note records what landed and was verified for
[Mode (2a) agentic GRPO](https://github.com/verl-project/verl-omni/issues/302)
on Lance-3B understanding (`Lance_3B_hf_und`), with frozen diffusion as an
external tool.

## Goal

Prove the Mode (2a) infra boundary:

1. Train **only** the agent LLM with stock verl GRPO (`main_ppo` + vLLM).
2. Call frozen diffusion through a **function tool** outside the actor optimizer.
3. Keep existing single-turn FlowGRPO paths untouched.

## Current Design

| Area | Location | Role |
| --- | --- | --- |
| GPU merge gate (ST-1, AC1) | `tests/special_e2e/run_agentic_grpo_lance.sh` | 1-step Lance agentic GRPO smoke |
| Toy Hermes data | `tests/special_e2e/create_dummy_agentic_data.py` | Few-shot `<tool_call>` + reflection seed (ST-1 auto-generates) |
| HF und export | `tests/special_e2e/prepare_lance_hf_und.py` | MoT → HF CausalLM und (optional tokenizer bake) |
| Tool chat template | `tests/special_e2e/qwen2_tool_chat_template.jinja2` | Upstream-style Jinja2; baked into und by `prepare_lance_hf_und.py` |
| Frozen tool stub/HTTP | `verl_omni/agent_loop/diffusion_tool.py` | `generate_image` function tool |
| Smoke reward | `verl_omni/utils/reward_score/agentic_reward.py` | Deterministic response-length variance for ST-1 |
| Trajectory types | `verl_omni/agent_loop/agentic_trajectory.py` | Mode (2a) data contract (`AgenticTrajectory`) |
| Trajectory CPU tests | `tests/agent_loop/test_agentic_trajectory.py` | Round-trip + stock tool wiring |
| CPU AC2 / AC3 | `tests/agent_loop/test_agentic_compat.py` | Stock tool wiring + FlowGRPO compat |

Hermes handling follows **upstream verl**:

- Parse: `actor_rollout_ref.rollout.multi_turn.format=hermes` → `HermesToolParser`
- Render: tool-aware Jinja baked into `Lance_3B_hf_und` tokenizer by
  `prepare_lance_hf_und.py` (Instruct-style; no fragile Hydra CLI Jinja override)
- No Hermes-specific preflight helper

This merge deliberately uses the stock `ToolAgentLoop` with no worker monkey patches.
`AgenticTrajectory` is delivered as an in-tree data contract with CPU coverage;
ST-1 does not yet consume it on the live `main_ppo` path. The full Mode (2a)
reward, rollout artifact context, custom metrics, and force/teacher loop remain
follow-up work outside this merge.

## How to run

```bash
# Operator env (CUDA / Ray / MODEL_PATH), then from repo root:
MODEL_PATH=/path/to/Lance_3B_hf_und \
  bash tests/special_e2e/run_agentic_grpo_lance.sh
```

ST-1:

1. Sets `multi_turn.format=hermes` (upstream ToolAgentLoop / HermesToolParser).
2. Uses the prepared und’s baked tool chat template (from
   `qwen2_tool_chat_template.jinja2` via `prepare_lance_hf_und.py`).
3. Regenerates toy Hermes parquet via `create_dummy_agentic_data.py` into
   `$DATA_DIR` (default `$HOME/data/agentic`) unless `ST1_USE_ENV_DATA=1`.

Needs ~2 free GPUs (`CUDA_VISIBLE_DEVICES=1,4` etc.) and
`GPU_MEM_UTIL` default **0.15** (raise if vLLM reports no KV cache room).

```bash
# CPU ACs
pytest tests/agent_loop/test_agentic_compat.py \
       tests/agent_loop/test_agentic_trajectory.py \
       tests/utils/test_agentic_reward.py \
       tests/special_e2e/test_create_dummy_agentic_data.py
```

Prepare und export if needed:

```bash
python3 tests/special_e2e/prepare_lance_hf_und.py \
  --src /path/to/Lance_3B \
  --dst /path/to/Lance_3B_hf_und
```
