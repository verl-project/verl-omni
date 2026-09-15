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
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
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
    "OmniAgentLoopWorkerTQ",
    "OmniAgentLoopManager",
    "split_assistant_rollouts",
    "split_rollout_turns",
]


def _transfer_queue_enabled(config) -> bool:
    """Return whether V1 TaskRunner has flipped TransferQueue on for this run."""
    if config is None:
        return False
    try:
        return bool(config.transfer_queue.enable)
    except Exception:  # noqa: BLE001
        return False


def _unwrap_ray_actor(cls):
    """Return the implementation class behind a ``@ray.remote`` actor wrapper."""
    meta = getattr(cls, "__ray_metadata__", None)
    inner = getattr(meta, "modified_class", None) if meta is not None else None
    if inner is None:
        raise TypeError(f"cannot unwrap Ray actor class: {cls!r}")
    return inner


def _stamp_scorer_knobs(batch, config) -> None:
    """Copy composed ``SCORER_KNOB_KEYS`` onto each sample ``extra_info``.

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

    from verl_omni.tools.trajectory.hydra_env import agentic_scorer_knobs_from_config

    knobs = agentic_scorer_knobs_from_config(config)
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


class OmniAgentLoopMixin:
    """Bind Hydra knobs and stamp trajectory kwargs shared by DataProto and TQ workers.

    Overrides must live on the Ray worker: ``generate_sequences`` dispatches
    remotely. Hermes / ``image_gen.py`` bind is gated on
    ``default_agent_loop == image_gen_tool_agent`` and only fills unset keys.
    """

    _AGENTIC_TOOL_FORMAT = "hermes"
    _AGENTIC_FUNCTION_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "image_gen.py"

    def _bind_agentic_rollout_config(self, config) -> None:
        from omegaconf import open_dict

        from verl_omni.tools.trajectory.hydra_env import merge_agentic_scorer_knobs

        # Bind by path string only — importing image_gen.py would double-register tools.
        bind_run_artifacts(config)
        bind_agentic_image_gen(config)
        self._agentic_scorer_bind = merge_agentic_scorer_knobs
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
        extra = kwargs.get("extra_info")
        stamp = getattr(self, "_agentic_scorer_bind", None)
        if stamp is not None:
            kwargs["extra_info"] = stamp(extra if isinstance(extra, dict) else {}, self.config)
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


class OmniAgentLoopWorker(OmniAgentLoopMixin, AgentLoopWorker):
    """DataProto / legacy worker used by L2 GPU smoke and ``trainer.use_v1=false``."""

    def __init__(self, config, *args, **kwargs):
        self._bind_agentic_rollout_config(config)
        super().__init__(config, *args, **kwargs)


_AgentLoopWorkerTQImpl = _unwrap_ray_actor(AgentLoopWorkerTQ)


class OmniAgentLoopWorkerTQImpl(OmniAgentLoopMixin, _AgentLoopWorkerTQImpl):
    """V1 TransferQueue worker: bind tools, then put outputs into the queue."""

    def __init__(self, config, *args, **kwargs):
        self._bind_agentic_rollout_config(config)
        super().__init__(config, *args, **kwargs)


OmniAgentLoopWorkerTQ = ray.remote(OmniAgentLoopWorkerTQImpl)


class OmniAgentLoopManager(AgentLoopManager):
    """Use stock rollout management, dump outputs, and mask invalid rollouts."""

    def __init__(self, *args, **kwargs):
        # Must set before AgentLoopManager.__init__ creates Ray workers.
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        # V1 TaskRunner sets transfer_queue.enable=True and passes TensorDict.
        # DataProto GPU smoke / v0 keep OmniAgentLoopWorker.
        if _transfer_queue_enabled(config):
            self.agent_loop_workers_class = OmniAgentLoopWorkerTQ
        else:
            self.agent_loop_workers_class = ray.remote(OmniAgentLoopWorker)
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
            prompts: ``DataProto`` (legacy / GPU smoke) or V1 ``TensorDict``.
                V1 workers put results into TransferQueue and this method
                returns ``None``.

        Returns:
            ``DataProto`` with invalid rollouts masked and metrics on
            ``meta_info["agentic_metrics"]`` and ``meta_info["timing"]``, or
            ``None`` on the V1 TensorDict path.
            Each DataProto row ``extra_info`` also carries ``SCORER_KNOB_KEYS``.
        """
        if not hasattr(prompts, "meta_info"):
            chunks = prompts.chunk(len(self.agent_loop_workers))
            ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
                ]
            )
            return None
        step = prompts.meta_info.get("global_steps")
        # Stamp inbound rows first: default RayPPOTrainer enables agent_reward_loop
        # (no RM), so AgentLoopWorker._compute_score.remote runs during generate
        # on kwargs built from this extra_info. Post-hoc output stamp is too late.
        _stamp_scorer_knobs(prompts, self.config)
        output = super().generate_sequences(prompts)
        # Streaming _postprocess does not copy input extra_info; stamp output too
        # so NaiveRewardManager / dumps still see composed knobs.
        _stamp_scorer_knobs(output, self.config)
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
