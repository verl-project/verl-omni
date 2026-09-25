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

"""Omni agent-loop runtime plus the generic loops shared by every pipeline.

Layout rule: a module here that *registers* an ``AgentLoopBase`` subclass is named
``*_agent_loop.py`` and holds a loop every pipeline may reuse
(``single_turn_agent_loop.py``). A concrete loop that belongs to one training recipe
lives in that pipeline instead (``verl_omni/pipelines/<recipe>/agent_loop.py``),
which is where ``image_gen_tool_agent`` and the MiniMax H3 loop are. The remaining
``*_agent_loop*.py`` modules here are the Ray workers and managers — the Omni
counterpart of upstream's ``experimental/agent_loop/agent_loop.py``, which likewise
keeps ``AgentLoopBase``, ``AgentLoopWorker`` and ``AgentLoopManager`` in one file.
"""

# The MiniMax H3 agent loop lives in its pipeline package; import it here so the
# @register decorator fires when the agent_loop package is imported. Do not
# re-export the class from the pipeline package __init__ (import cycle).
from verl_omni.pipelines.minimax_h3_diffusion_nft.agent_loop import MiniMaxH3DiffusionSingleTurnAgentLoop

from .composite_agent_loop import CompositeAgentLoopWorker
from .diffusion_agent_loop import DiffusionAgentLoopOutput, DiffusionAgentLoopWorker
from .diffusion_agent_loop_tq import (
    DiffusionAgentLoopWorkerTQ,
    create_diffusion_agent_loop_manager,
)
from .single_turn_agent_loop import DiffusionSingleTurnAgentLoop, OmniSingleTurnAgentLoop

__all__ = [
    "CompositeAgentLoopWorker",
    "DiffusionAgentLoopOutput",
    "DiffusionAgentLoopWorker",
    "DiffusionAgentLoopWorkerTQ",
    "create_diffusion_agent_loop_manager",
    "DiffusionSingleTurnAgentLoop",
    "OmniSingleTurnAgentLoop",
    "MiniMaxH3DiffusionSingleTurnAgentLoop",
]
