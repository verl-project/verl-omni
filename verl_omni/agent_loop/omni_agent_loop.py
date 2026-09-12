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
    active_user_prompt,
    bind_agentic_image_gen,
    bind_run_artifacts,
    build_trajectory_relpath,
    clear_good_enough_yes_reached,
    reset_active_trajectory_relpath,
    set_active_trajectory_relpath,
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


def _stamp_reward_context(batch, config) -> None:
    """Stamp reward-side inputs onto each sample ``extra_info``.

    Copies composed ``SCORER_KNOB_KEYS`` (judge knobs) plus the resolved
    ``rollout_images_root``. Reward actors run in Ray processes that never bind
    Hydra ``config`` or ``tools.trajectory``, so both must travel on the row:
    the knobs let ``compute_score`` avoid yaml-filling, and the images root lets
    its ``call_reflect_vlm`` fallback resolve the last generated PNG when the
    trajectory has no parseable ``judge_image`` observation.

    Args:
        batch: ``DataProto`` (or test stub) with ``non_tensor_batch``. Used for
            inbound ``prompts`` (before worker dispatch) and concatenated
            ``output`` (streaming ``_postprocess`` drops input ``extra_info``).
        config: Composed Hydra config from the agent-loop manager.

    Returns:
        None. Mutates ``batch.non_tensor_batch["extra_info"]`` in place.
        Existing row keys win (same precedence as ``w_*``).
    """
    import numpy as np

    from verl_omni.tools.trajectory import resolve_rollout_images_root
    from verl_omni.tools.trajectory.hydra_env import agentic_scorer_knobs_from_config

    knobs = agentic_scorer_knobs_from_config(config)
    # Same root the frozen tool writes under; reward fallback only accepts PNGs
    # confined to it. Harmless when the reward process cannot read the path.
    knobs["rollout_images_root"] = str(resolve_rollout_images_root())
    ntb = getattr(batch, "non_tensor_batch", None)
    if ntb is None:
        batch.non_tensor_batch = {}
        ntb = batch.non_tensor_batch
    extras = ntb.get("extra_info")
    if extras is None:
        n = 0
        for value in ntb.values():
            n = len(value)
            break
        if n == 0:
            tensors = getattr(batch, "batch", None) or {}
            for value in tensors.values():
                shape = getattr(value, "shape", None)
                if shape:
                    n = int(shape[0])
                    break
        extras = np.empty(n, dtype=object)
        extras[:] = [{} for _ in range(n)]
        ntb["extra_info"] = extras
    stamped = np.empty(len(extras), dtype=object)
    for index, info in enumerate(extras):
        row = dict(info) if isinstance(info, dict) else {}
        for key, value in knobs.items():
            row.setdefault(key, value)
        stamped[index] = row
    ntb["extra_info"] = stamped


class OmniAgentLoopWorker(AgentLoopWorker):
    """Bind trajectory Hydra knobs and pass step kwargs into the agent loop.

    Overrides must live here: ``AgentLoopManager.generate_sequences`` dispatches
    to Ray workers. Hermes / ``image_gen.py`` bind is gated on
    ``default_agent_loop == image_gen_tool_agent`` and only fills unset keys.
    """

    _AGENTIC_TOOL_FORMAT = "hermes"
    _AGENTIC_FUNCTION_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "image_gen.py"

    def __init__(self, config, *args, **kwargs):
        from omegaconf import open_dict

        # Bind by path string only — importing image_gen.py would double-register tools.
        bind_run_artifacts(config)
        bind_agentic_image_gen(config)
        default_loop = None
        try:
            default_loop = config.actor_rollout_ref.rollout.agent.get("default_agent_loop")
        except Exception:  # noqa: BLE001
            default_loop = None
        if default_loop == "image_gen_tool_agent":
            tool_path = self._AGENTIC_FUNCTION_TOOLS
            if not tool_path.is_file():
                raise FileNotFoundError(
                    f"agentic function tools not found at {tool_path}. Expected verl_omni/tools/image_gen.py"
                )
            with open_dict(config.actor_rollout_ref.rollout.multi_turn):
                mt = config.actor_rollout_ref.rollout.multi_turn
                # Only fill unset keys so explicit Hydra overrides still win.
                if not mt.get("function_tool_path"):
                    mt.function_tool_path = str(tool_path)
                if not mt.get("format"):
                    mt.format = self._AGENTIC_TOOL_FORMAT
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
        prompt_token = active_user_prompt.set(user_prompt)
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
            active_user_prompt.reset(prompt_token)
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
            bind_run_artifacts(config)
            bind_agentic_image_gen(config)
        super().__init__(*args, **kwargs)
        model_path = self.model_config.get("tokenizer_path") or self.model_config.get("path")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self._monitor_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    def generate_sequences(self, prompts):
        """Run stock generate, dump, discard invalid rows, and emit rollout metrics.

        Args:
            prompts: ``DataProto`` batch from the trainer.

        Returns:
            ``DataProto`` with invalid rollouts masked and metrics on
            ``meta_info["agentic_metrics"]`` and ``meta_info["timing"]``.
            Each row ``extra_info`` also carries ``SCORER_KNOB_KEYS`` and
            ``rollout_images_root``.
        """
        step = prompts.meta_info.get("global_steps")
        # Stamp inbound rows first: default RayPPOTrainer enables agent_reward_loop
        # (no RM), so AgentLoopWorker._compute_score.remote runs during generate
        # on kwargs built from this extra_info. Post-hoc output stamp is too late.
        _stamp_reward_context(prompts, self.config)
        output = super().generate_sequences(prompts)
        # Streaming _postprocess does not copy input extra_info; stamp output too
        # so NaiveRewardManager / dumps still see composed knobs.
        _stamp_reward_context(output, self.config)
        # Dump before discard: discard zeros response_mask and hides tool-less prose.
        dump_raw_rollouts(tokenizer=self._monitor_tokenizer, output=output, step=step)
        discard_invalid_rollouts(output)
        metrics = AgenticRewardMetrics.aggregate(output.non_tensor_batch)
        if metrics:
            meta = getattr(output, "meta_info", None)
            if not isinstance(meta, dict):
                output.meta_info = {}
                meta = output.meta_info
            # Keep a dedicated stash for L2 smoke / trainers that read agentic_metrics.
            existing = meta.get("agentic_metrics")
            if isinstance(existing, dict):
                existing.update(metrics)
            else:
                meta["agentic_metrics"] = dict(metrics)
            # Fold into timing so stock PPO's timing_raw.update(meta_info["timing"])
            # carries rollout counters into compute_timing_metrics for every backend.
            timing = meta.get("timing")
            if not isinstance(timing, dict):
                timing = {}
                meta["timing"] = timing
            timing.update(metrics)
            step_i = int(step) if step is not None else None
            logger.info("agentic_metrics step=%s %s", step_i, metrics)
        return output
