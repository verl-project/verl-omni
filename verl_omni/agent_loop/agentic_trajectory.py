# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agentic multi-turn trajectory data structures for Mode (2a) agentic RL.

``AgenticTrajectory`` and its constituent types (``AgenticTurn``, ``ToolCall``,
``ToolOutput``, ``AgenticMetadata``) carry per-turn tokens, logprobs, tool
calls/outputs, and trajectory metadata for online multi-turn GRPO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass
class ToolCall:
    """A tool invocation by the agent, containing a (possibly rewritten) prompt."""

    tool_name: str
    params: dict[str, Any]  # key "prompt" holds the prompt sent to diffusion


@dataclass
class ToolOutput:
    """Observation returned by a frozen diffusion tool call."""

    output_type: Literal["image"]
    output_data: torch.Tensor  # image tensor [C, H, W]
    is_stub: bool = False  # True when und-only / missing diffusion_output synthesized a stub


@dataclass
class AgenticTurn:
    """One turn of the multi-turn agentic interaction.

    Decision vocabulary: ``"continue"`` or ``"stop"``.

    Prompt rewriting is captured via:
      - turn[i].tool_call.params["prompt"]  — prompt at turn i
      - turn[i+1].tool_call.params["prompt"] — rewritten prompt at turn i+1
    """

    turn_idx: int
    agent_tokens: list[int]  # full agent text tokens (train mask applies to agent turns only)
    agent_logprobs: list[float]  # per-token logprobs from rollout
    agent_text: str  # decoded text: reasoning + prompt + decision
    tool_call: ToolCall | None = None  # None on stop turn
    tool_output: ToolOutput | None = None  # None on stop turn
    decision: Literal["continue", "stop"] = "stop"


@dataclass
class AgenticMetadata:
    """Trajectory-level metadata."""

    num_turns: int
    terminated: bool
    termination_reason: str  # "agent_stop" | "max_turns" | "response_truncated"
    tool_stubbed: bool = False  # True if any tool turn used a stub image


@dataclass
class AgenticTrajectory:
    """Full multi-turn agentic trajectory for Mode (2a) RL."""

    prompt: str
    turns: list[AgenticTurn]
    reward_score: float | None = None
    metadata: AgenticMetadata = field(default_factory=lambda: AgenticMetadata(0, False, ""))
    trajectory_id: str = ""
    source_dataset: str = ""


def agentic_trajectory_to_dict(traj: AgenticTrajectory) -> dict[str, Any]:
    """Serialize an AgenticTrajectory to a JSON-serializable dict for non_tensor_batch."""
    return {
        "prompt": traj.prompt,
        "trajectory_id": traj.trajectory_id,
        "source_dataset": traj.source_dataset,
        "turns": [
            {
                "turn_idx": t.turn_idx,
                "agent_tokens": t.agent_tokens,
                "agent_logprobs": t.agent_logprobs,
                "agent_text": t.agent_text,
                "tool_call": (
                    {"tool_name": t.tool_call.tool_name, "params": t.tool_call.params} if t.tool_call else None
                ),
                "tool_output": (
                    {
                        "output_type": t.tool_output.output_type,
                        "output_data_shape": list(t.tool_output.output_data.shape),
                        "is_stub": bool(getattr(t.tool_output, "is_stub", False)),
                    }
                    if t.tool_output
                    else None
                ),
                "decision": t.decision,
            }
            for t in traj.turns
        ],
        "reward_score": traj.reward_score,
        "metadata": {
            "num_turns": traj.metadata.num_turns,
            "terminated": traj.metadata.terminated,
            "termination_reason": traj.metadata.termination_reason,
            "tool_stubbed": bool(getattr(traj.metadata, "tool_stubbed", False)),
        },
    }


def agentic_trajectory_from_dict(d: dict[str, Any]) -> AgenticTrajectory:
    """Deserialize an AgenticTrajectory from a dict (round-trip with ``to_dict``)."""
    turns = []
    for t in d.get("turns", []):
        tc = t.get("tool_call")
        to = t.get("tool_output")
        tool_call = ToolCall(tc["tool_name"], tc["params"]) if tc else None
        # Round-trip stores shape only; restore an empty tensor of that shape.
        if to is not None:
            shape = to.get("output_data_shape") or []
            tool_output = ToolOutput(
                to["output_type"],
                torch.empty(shape) if shape else torch.empty(0),
                is_stub=bool(to.get("is_stub", False)),
            )
        else:
            tool_output = None
        turns.append(
            AgenticTurn(
                turn_idx=t["turn_idx"],
                agent_tokens=list(t.get("agent_tokens", [])),
                agent_logprobs=list(t.get("agent_logprobs", [])),
                agent_text=t.get("agent_text", ""),
                tool_call=tool_call,
                tool_output=tool_output,
                decision=t.get("decision", "stop"),
            )
        )
    meta = d.get("metadata") or {}
    return AgenticTrajectory(
        prompt=d.get("prompt", ""),
        turns=turns,
        reward_score=d.get("reward_score"),
        metadata=AgenticMetadata(
            num_turns=meta.get("num_turns", len(turns)),
            terminated=meta.get("terminated", False),
            termination_reason=meta.get("termination_reason", ""),
            tool_stubbed=bool(meta.get("tool_stubbed", False)),
        ),
        trajectory_id=d.get("trajectory_id", ""),
        source_dataset=d.get("source_dataset", ""),
    )
