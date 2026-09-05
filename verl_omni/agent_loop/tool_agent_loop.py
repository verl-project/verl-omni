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

"""Image-gen ToolAgentLoop: force-first curriculum + optional forced Reflection.

Teacher-forced Hermes tool tokens use ``response_mask=1``; injected Reflection
after successful ``judge_image`` uses ``response_mask=0``. Terminal ``Done.`` is
always policy-sampled.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall

from verl_omni.agent_loop.utils import (
    build_forced_reflection,
    count_successful_generates,
    count_successful_judges,
    fits_response_budget,
    force_enabled,
    force_first_generate_probability,
    hermes_tool_call,
    last_user_text,
    max_generate_passes,
    rewrite_judge_before_generate,
    tool_calls_are_premature_judge,
    tool_message_text,
)
from verl_omni.tools.trajectory import (
    clear_good_enough_yes_reached,
    clear_latest_tool_image_for_active_rollout,
    get_active_trajectory_relpath,
    reset_active_trajectory_relpath,
    set_active_trajectory_relpath,
)

logger = logging.getLogger(__name__)


@register("image_gen_tool_agent")
class ImageGenToolAgentLoop(ToolAgentLoop):
    """Stock tool agent + forced Reflection after successful ``judge_image``."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Per-rollout latch reset: YES from sample N must not block sample N+1.
        self._agentic_step = kwargs.pop("_agentic_step", 0)
        self._agentic_validate = bool(kwargs.pop("_agentic_validate", False))
        self._agentic_trajectory_relpath = (
            kwargs.pop("_agentic_trajectory_relpath", None) or get_active_trajectory_relpath()
        )
        path_tokens = None
        if self._agentic_trajectory_relpath:
            path_tokens = set_active_trajectory_relpath(self._agentic_trajectory_relpath)
        clear_good_enough_yes_reached()
        clear_latest_tool_image_for_active_rollout()
        try:
            # Dense extra_fields so DataProto.concat across workers keeps a shared key set.
            output = await super().run(sampling_params, **kwargs)
            output.extra_fields.pop("_forced_generate_prompt", None)
            output.extra_fields.setdefault("forced_reflection", False)
            output.extra_fields.setdefault("force_stop_max_passes", False)
            output.extra_fields.setdefault("stop_decision_required", False)
            output.extra_fields.setdefault("forced_first_generate", False)
            output.extra_fields.setdefault("forced_first_judge", False)
            output.extra_fields.setdefault("rewrote_judge_before_generate", False)
            output.extra_fields.setdefault("force_first_probability", 0.0)
            output.extra_fields["trajectory_relpath"] = self._agentic_trajectory_relpath or ""
            return output
        finally:
            clear_good_enough_yes_reached()
            clear_latest_tool_image_for_active_rollout()
            if path_tokens is not None:
                reset_active_trajectory_relpath(path_tokens)

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        # Re-bind trajectory path before each tool (tool threads may not see run()'s bind).
        relpath = getattr(self, "_agentic_trajectory_relpath", None) or get_active_trajectory_relpath()
        path_tokens = None
        if relpath:
            path_tokens = set_active_trajectory_relpath(relpath)
            agent_data.extra_fields["trajectory_relpath"] = relpath
        try:
            return await super()._call_tool(tool_call, tools_kwargs, agent_data)
        finally:
            if path_tokens is not None:
                reset_active_trajectory_relpath(path_tokens)

    async def _rewrite_premature_judge_to_generate(self, agent_data: AgentData) -> AgentState | None:
        """Replace a first-turn ``judge_image`` call with ``generate_image`` (mask=1)."""
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        if "generate_image" not in active_tools:
            return None
        prompt = last_user_text(agent_data.messages)
        if not prompt:
            return None
        hermes = hermes_tool_call("generate_image", prompt=prompt)
        tool_call = FunctionCall(
            name="generate_image",
            arguments=json.dumps({"prompt": prompt}, ensure_ascii=False),
        )
        new_state = await self._replace_last_assistant_with_tool_call(agent_data, hermes, tool_call)
        if new_state is None:
            return None
        agent_data.extra_fields["rewrote_judge_before_generate"] = True
        agent_data.extra_fields["_forced_generate_prompt"] = prompt
        logger.info(
            "Rewrote premature judge_image → generate_image at global_step=%s (no live PNG yet)",
            getattr(self, "_agentic_step", 0),
        )
        return new_state

    async def _encode_assistant_completion(self, text: str) -> list[int]:
        """Encode as a generation delta (content + EOS), matching server-sampled tokens.

        ``apply_chat_template([assistant])`` would re-emit ``<|im_start|>assistant``
        (and with ``add_generation_prompt=True`` a second assistant header). The
        rollout prompt already ends at the assistant generation prefix, so teacher
        tokens must be content-only like vLLM completions.
        """
        eos = self.tokenizer.eos_token or "<|im_end|>"
        payload = text if text.endswith(eos) else f"{text}{eos}\n"
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(payload, add_special_tokens=False),
        )

    async def _replace_last_assistant_with_tool_call(
        self,
        agent_data: AgentData,
        hermes_text: str,
        tool_call: FunctionCall,
    ) -> AgentState | None:
        """Replace the last sampled assistant span with teacher-forced Hermes tokens.

        Returns ``None`` if the forced span would exceed ``response_length`` (caller
        keeps the original TERMINATED state; nothing is mutated).
        """
        response_ids = await self._encode_assistant_completion(hermes_text)
        n_last = len(agent_data.response_ids)
        new_mask_len = len(agent_data.response_mask) - n_last + len(response_ids)
        if new_mask_len >= self.response_length or not response_ids:
            return None

        if n_last:
            agent_data.prompt_ids = agent_data.prompt_ids[:-n_last]
            agent_data.response_mask = agent_data.response_mask[:-n_last]
            if agent_data.response_logprobs:
                agent_data.response_logprobs = agent_data.response_logprobs[:-n_last]

        assistant_msg = {"role": "assistant", "content": hermes_text}
        # Stock ToolAgentLoop does not append assistant turns; keep messages coherent for dumps.
        if agent_data.messages and agent_data.messages[-1].get("role") == "assistant":
            agent_data.messages[-1] = assistant_msg
        else:
            agent_data.messages.append(assistant_msg)

        agent_data.response_ids = list(response_ids)
        agent_data.prompt_ids += response_ids
        # mask=1: train on teacher-forced Hermes tool tokens.
        agent_data.response_mask += [1] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.tool_calls = [tool_call]
        return AgentState.PROCESSING_TOOLS

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Teacher-force missing generate/judge tool calls during early curriculum."""
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        probability = force_first_generate_probability(
            getattr(self, "_agentic_step", 0),
            validate=getattr(self, "_agentic_validate", False),
        )
        agent_data.extra_fields.setdefault("forced_first_generate", False)
        agent_data.extra_fields.setdefault("forced_first_judge", False)
        agent_data.extra_fields.setdefault("rewrote_judge_before_generate", False)
        agent_data.extra_fields["force_first_probability"] = float(probability)

        # Premature judge with no live generate → rewrite to generate (independent of anneal).
        n_gen = count_successful_generates(agent_data.messages)
        if (
            rewrite_judge_before_generate()
            and state == AgentState.PROCESSING_TOOLS
            and n_gen == 0
            and tool_calls_are_premature_judge(agent_data.tool_calls)
            and len(agent_data.response_mask) < self.response_length
        ):
            rewritten = await self._rewrite_premature_judge_to_generate(agent_data)
            if rewritten is not None:
                return rewritten

        active_tools = getattr(agent_data, "_active_tools", self.tools)
        if (
            state != AgentState.TERMINATED
            or agent_data.tool_calls
            or probability <= 0.0
            or random.random() >= probability
            or len(agent_data.response_mask) >= self.response_length
        ):
            return state

        n_judge = count_successful_judges(agent_data.messages)

        # First turn with no tools → teacher-force generate_image.
        if agent_data.assistant_turns == 1 and n_gen == 0 and "generate_image" in active_tools:
            prompt = last_user_text(agent_data.messages)
            if not prompt:
                return state
            hermes = hermes_tool_call("generate_image", prompt=prompt)
            tool_call = FunctionCall(
                name="generate_image",
                arguments=json.dumps({"prompt": prompt}, ensure_ascii=False),
            )
            new_state = await self._replace_last_assistant_with_tool_call(agent_data, hermes, tool_call)
            if new_state is None:
                return state
            agent_data.extra_fields["forced_first_generate"] = True
            agent_data.extra_fields["_forced_generate_prompt"] = prompt
            logger.info(
                "Teacher-forced generate_image at global_step=%s (p=%.3f); Hermes tokens mask=1",
                getattr(self, "_agentic_step", 0),
                probability,
            )
            return new_state

        # After generate(s) without judge → teacher-force judge_image (compact placeholder args).
        if n_gen >= 1 and n_judge < n_gen and "judge_image" in active_tools:
            user_request = "same as user message"
            image_prompt = "last"
            hermes = hermes_tool_call(
                "judge_image",
                user_request=user_request,
                image_prompt=image_prompt,
            )
            tool_call = FunctionCall(
                name="judge_image",
                arguments=json.dumps(
                    {"user_request": user_request, "image_prompt": image_prompt},
                    ensure_ascii=False,
                ),
            )
            new_state = await self._replace_last_assistant_with_tool_call(agent_data, hermes, tool_call)
            if new_state is None:
                return state
            agent_data.extra_fields["forced_first_judge"] = True
            logger.info(
                "Teacher-forced judge_image at global_step=%s (p=%.3f, gen=%d judge=%d); Hermes tokens mask=1",
                getattr(self, "_agentic_step", 0),
                probability,
                n_gen,
                n_judge,
            )
            return new_state

        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        agent_data.extra_fields.setdefault("forced_reflection", False)
        agent_data.extra_fields.setdefault("force_stop_max_passes", False)
        agent_data.extra_fields.setdefault("stop_decision_required", False)
        state = await super()._handle_processing_tools_state(agent_data)
        if state == AgentState.TERMINATED:
            return state

        forced: tuple[str, bool] | None = None
        gen_passes = 0
        max_passes = max_generate_passes()
        for message in reversed(agent_data.messages):
            if message.get("role") != "tool":
                break
            gen_passes = count_successful_generates(agent_data.messages)
            force_done = gen_passes >= max_passes
            # With force off, only inject at the generate-pass cap (budget guard).
            if not force_enabled() and not force_done:
                return state
            forced = build_forced_reflection(
                tool_message_text(message),
                force_done=force_done,
                generate_pass=gen_passes,
                max_passes=max_passes,
            )
            if forced is not None:
                break
        if forced is None:
            return state

        reflection_text, stop_required = forced
        if not force_enabled():
            if not stop_required:
                return state
            force_done = gen_passes >= max_passes
            if not force_done:
                return state
        reflection_text = f"{reflection_text} agentic_forced_reflection=1"
        assistant_msg = {"role": "assistant", "content": reflection_text}
        response_ids = await self.apply_chat_template(
            [assistant_msg],
            remove_system_prompt=True,
        )
        if not fits_response_budget(
            len(agent_data.response_mask),
            len(response_ids),
            self.response_length,
        ):
            return AgentState.TERMINATED

        agent_data.messages.append(assistant_msg)
        agent_data.prompt_ids += response_ids
        # mask=0: Reflection is context only (not policy-sampled).
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.assistant_turns += 1
        agent_data.extra_fields["forced_reflection"] = True
        agent_data.extra_fields["force_stop_max_passes"] = bool(
            stop_required and "agentic_force_stop_max_passes=1" in reflection_text
        )
        agent_data.extra_fields["stop_decision_required"] = bool(stop_required)
        logger.info(
            "Forced Reflection after judge_image (stop_required=%s, chars=%d, force_full=%s)",
            stop_required,
            len(reflection_text),
            force_enabled(),
        )
        return AgentState.GENERATING
