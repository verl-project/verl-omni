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

"""Frozen ``generate_image`` / ``judge_image`` tools, loaded by file path.

``OmniAgentLoopWorker`` binds ``image_gen.py`` as ``function_tool_path``.
Importing this package does not register the tools.

Tools are registered as verl ``@function_tool`` callables so frozen HTTP sidecars
register via ``function_tool_path`` with no create/execute/release lifecycle.

Upstream ``FunctionTool.call`` runs **sync** tool bodies with ``asyncio.to_thread``.
That call never receives ``agent_data`` (by design — see ``ToolAgentLoop._call_tool``).
``asyncio.to_thread`` **copies the calling task's context**, so ContextVars bound on
the agent-loop task (trajectory relpath, rollout id, user prompt) are visible inside
the tool thread without changing verl's FunctionTool contract.

Thread-locals alone are **not** enough: the default executor reuses workers across
concurrent rollouts, so a TLS YES latch leaked between samples (blocked the next
sample's first ``generate_image``). ContextVars + a rollout_id-keyed latch avoid that.

``BaseTool.execute(..., agent_data=...)`` remains the upstream path for truly stateful
tools; we deliberately did not take it here so sidecar tools stay file-path registered
and HTTP-stateless. See ``trajectory/`` for ContextVar / registry / latch used by
the tool bodies. See ``agent_helper/`` for *tool-agent* readers used by
``ImageGenToolAgentLoop`` (curriculum / Hermes / forced Reflection) — not by
``generate_image`` / ``judge_image``.
"""
