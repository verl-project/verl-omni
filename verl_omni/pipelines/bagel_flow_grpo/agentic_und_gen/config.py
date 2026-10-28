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
"""Omni / Bagel Co-RL (Joint-Training) rollout agent config extensions (do not patch upstream verl)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from verl.workers.config.rollout import AgentLoopConfig

from verl_omni.utils.agentic.config import agentic_scorer_knobs_from_config, yaml_agentic_image_gen_defaults

__all__ = ["BagelCorlAgentLoopConfig", "bagel_scorer_knobs_from_config"]


def bagel_scorer_knobs_from_config(config: Any) -> dict[str, Any]:
    """Extend the shared scorer payload with the BAGEL recipe's reward settings."""
    knobs = agentic_scorer_knobs_from_config(config)
    node = (
        config.get("agentic_image_gen") if isinstance(config, Mapping) else getattr(config, "agentic_image_gen", None)
    )
    defaults = yaml_agentic_image_gen_defaults()
    for key in (
        "score_backend",
        "corl_reward_version",
        "unified_good_enough_threshold",
        "pickscore_good_enough_threshold",
    ):
        knobs[key] = node.get(key, defaults[key]) if isinstance(node, Mapping) else getattr(node, key, defaults[key])
    return knobs


@dataclass
class BagelCorlAgentLoopConfig(AgentLoopConfig):
    """``AgentLoopConfig`` plus Bagel Co-RL (Joint-Training) knobs used by ``bagel_multiturn_agent``.

    Kept in verl-omni so upstream ``verl.workers.config.AgentLoopConfig`` stays untouched.
    """

    # S = FlowGRPO seeds per generate_image turn (not episode GEN-turn count K).
    gen_samples_per_call: Optional[int] = None
    # Empty preserves policy conditioning; sentinel uses the original task request.
    corl_fixed_gen_prompt: str = ""
    # PR1 fail-closed: must be 1 (bounds in-episode K = number of GEN turns).
    max_generate_passes: Optional[int] = None
    # UND turn budget before force-stop (bounds in-episode J).
    max_und_turns: Optional[int] = None
    # Reduction over the S per-seed ``good_enough`` flags when deciding the
    # episode-level stop cue: ``"any"`` (best-of-S, default), ``"all"``, or
    # ``"mean"`` (fraction of YES flags >= ``agentic_image_gen.good_enough_threshold``).
    # Declared as a real field — ``agentic_und_gen/agent_loop.py`` reads it and fails loud on an
    # unknown value, so leaving it undeclared would instead fail at ``instantiate``
    # with "unexpected keyword" for any recipe that sets it.
    good_enough_reduction: str = "any"
    # Set True only after Bagel UND AR (Hermes tool-call) is proven on the live
    # replica — ``bagel_single_stage`` is DiffusionStrategy / GEN-only today.
    und_ar_serving_ready: bool = False
    # Deploy yaml for the standalone UND AR replica (``bagel_think`` topology).
    # Required when ``und_ar_serving_ready=True``; sibling of GEN ``deploy_config``.
    und_deploy_config: Optional[str] = None
    # GPUs for the standalone UND AR ``LLMServerManager`` (not colocated with GEN).
    und_n_gpus: Optional[int] = None
    # HF config used to build the UND AR (Thinker) ``OmniModelConfig``.
    # ``OmniModelConfig`` runs ``AutoConfig.from_pretrained`` on this path, falling
    # back to the model path. Bagel checkpoints are weights-only (``model_type:
    # bagel``, no ``auto_map``/modeling code), so ``transformers`` cannot resolve
    # them; set this to the checkpoint's ``llm_config.json`` instead.
    und_hf_config_path: Optional[str] = None
    und_gpu_memory_utilization: Optional[float] = None
