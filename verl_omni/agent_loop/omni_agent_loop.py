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

"""Agent-loop worker wiring, rollout monitoring, and invalid-rollout masking."""

from __future__ import annotations

import logging
from pathlib import Path

import ray
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils import hf_tokenizer

from verl_omni.tools.trajectory import (
    bind_run_artifact_env,
    build_trajectory_relpath,
    clear_good_enough_yes_reached,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)
from verl_omni.utils.agentic.image_gen_rollout_dump import discard_invalid_rollouts, dump_raw_rollouts
from verl_omni.utils.agentic.image_gen_rollout_parse import (
    last_user_prompt,
    split_assistant_rollouts,
    split_rollout_turns,
)
from verl_omni.utils.metrics_utils import AgenticRewardMetrics

# Register ``image_gen_tool_agent`` when this module is loaded.
from . import tool_agent_loop as image_gen_tool_agent_loop  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "OmniAgentLoopWorker",
    "OmniAgentLoopManager",
    "split_assistant_rollouts",
    "split_rollout_turns",
]


class OmniAgentLoopWorker(AgentLoopWorker):
    """Worker-side hooks: trajectory bind + step kwargs for force-first curriculum.

    ``AgentLoopManager.generate_sequences`` dispatches to Ray ``AgentLoopWorker``s.
    Overrides on the Manager class never run per-rollout — they must live here.

    Also hard-binds agentic multi-turn defaults (Hermes + ``verl_omni/tools``)
    so launch recipes need not pass ``function_tool_path`` / ``format`` Hydra overrides.
    """

    _AGENTIC_TOOL_FORMAT = "hermes"
    _AGENTIC_FUNCTION_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "image_gen.py"

    def __init__(self, config, *args, **kwargs):
        from omegaconf import open_dict

        # Bind by path string only — importing image_gen.py would double-register tools.
        bind_run_artifact_env(config)
        tool_path = self._AGENTIC_FUNCTION_TOOLS
        if not tool_path.is_file():
            raise FileNotFoundError(
                f"agentic function tools not found at {tool_path}. Expected verl_omni/tools/image_gen.py"
            )
        with open_dict(config.actor_rollout_ref.rollout.multi_turn):
            config.actor_rollout_ref.rollout.multi_turn.function_tool_path = str(tool_path)
            config.actor_rollout_ref.rollout.multi_turn.format = self._AGENTIC_TOOL_FORMAT
        super().__init__(config, *args, **kwargs)

    async def _run_agent_loop(
        self,
        sampling_params,
        trajectory,
        *,
        agent_name,
        trace=True,
        **kwargs,
    ):
        relpath = build_trajectory_relpath(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
        )
        raw_prompt = kwargs.get("raw_prompt")
        user_prompt = last_user_prompt(raw_prompt) if raw_prompt is not None else ""
        path_token = set_active_trajectory_relpath(relpath)
        prompt_token = set_active_user_prompt(user_prompt)
        clear_good_enough_yes_reached()
        kwargs["_agentic_step"] = trajectory["step"]
        kwargs["_agentic_validate"] = trajectory["validate"]
        kwargs["_agentic_trajectory_relpath"] = relpath
        try:
            return await super()._run_agent_loop(
                sampling_params,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                **kwargs,
            )
        finally:
            reset_active_user_prompt(prompt_token)
            reset_active_trajectory_relpath(path_token)


class OmniAgentLoopManager(AgentLoopManager):
    """Use stock rollout management, dump outputs, and mask invalid rollouts."""

    def __init__(self, *args, **kwargs):
        # Must set before AgentLoopManager.__init__ creates Ray workers.
        self.agent_loop_workers_class = ray.remote(OmniAgentLoopWorker)
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is not None:
            bind_run_artifact_env(config)
        super().__init__(*args, **kwargs)
        model_path = self.model_config.get("tokenizer_path") or self.model_config.get("path")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self._monitor_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    def generate_sequences(self, prompts):
        step = prompts.meta_info.get("global_steps")
        output = super().generate_sequences(prompts)
        # Dump before discard: discard zeros response_mask and hides tool-less prose.
        dump_raw_rollouts(tokenizer=self._monitor_tokenizer, output=output, step=step)
        discard_invalid_rollouts(output)
        metrics = AgenticRewardMetrics.aggregate(output.non_tensor_batch)
        if metrics:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=int(step) if step is not None else None, commit=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log agentic reward metrics to W&B: %s", exc)
        return output
