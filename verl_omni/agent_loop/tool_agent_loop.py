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

"""Image-gen ``ToolAgentLoop`` with force-first curriculum and forced Reflection.

Teacher-forced Hermes tool tokens use ``response_mask=1``; injected Reflection
uses ``response_mask=0``. Terminal ``Done.`` is policy-sampled.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall

from verl_omni.tools.agent_helper.image_gen_utils import (
    build_forced_reflection,
    count_executed_generates,
    count_successful_generates,
    count_successful_judges,
    fits_response_budget,
    force_first_generate_probability,
    hermes_tool_call,
    last_user_text,
    max_generate_passes,
    tool_calls_are_premature_judge,
    tool_message_text,
)
from verl_omni.tools.trajectory import (
    active_trajectory_relpath,
    clear_good_enough_yes_reached,
    clear_latest_tool_image_for_active_rollout,
    reset_active_trajectory_relpath,
    set_active_trajectory_relpath,
)
from verl_omni.tools.trajectory.hydra_env import agentic_get_bool
from verl_omni.utils.agentic.plan_protocol import plan_lines_from_prose

logger = logging.getLogger(__name__)


def _assistant_content_text(message: dict[str, Any]) -> str:
    """Return the plain text of an assistant message, tolerating multimodal content.

    Args:
        message: One chat message.

    Returns:
        Joined text parts, or ``""`` when the message carries no text.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return ""


@register("image_gen_tool_agent")
class ImageGenToolAgentLoop(ToolAgentLoop):
    """Stock tool agent plus force-first curriculum and forced Reflection."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Per-rollout latch reset: YES from sample N must not block sample N+1.
        self._agentic_step = kwargs.pop("_agentic_step", 0)
        self._agentic_validate = bool(kwargs.pop("_agentic_validate", False))
        # Monitoring label only. Reflect and plan rollouts run the *same* loop; the only
        # difference between the two data sources is the system prompt, so nothing here
        # may branch on this value.
        self._agentic_task_type = str(kwargs.pop("_agentic_task_type", "") or "").strip().lower()
        self._agentic_trajectory_relpath = (
            kwargs.pop("_agentic_trajectory_relpath", None) or active_trajectory_relpath.get()
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
            output.extra_fields.setdefault("refused_premature_judge", False)
            output.extra_fields.setdefault("force_first_probability", 0.0)
            output.extra_fields.setdefault("force_first_swap_rejected", False)
            output.extra_fields.setdefault("plan_turn_seen", False)
            # Monitoring label, mirrored from the row's ``task_type``.
            output.extra_fields.setdefault("agentic_task_type", self._agentic_task_type)
            output.extra_fields.setdefault("num_generate_image_prompts", 0)
            output.extra_fields.setdefault("rollout_has_generate", 0)
            output.extra_fields.setdefault("rollout_valid", 0)
            output.extra_fields["trajectory_relpath"] = self._agentic_trajectory_relpath or ""
            return output
        finally:
            clear_good_enough_yes_reached()
            clear_latest_tool_image_for_active_rollout()
            if path_tokens is not None:
                reset_active_trajectory_relpath(path_tokens)

    async def _agentic_refuse_tool_calls(
        self,
        agent_data: AgentData,
        *,
        reason: str,
        guidance: str,
        flag: str,
    ) -> AgentState:
        """Answer every call in this turn with one observation and execute none of them.

        A refusal is *environment* feedback, not harness-authored model text. The policy's
        call stays in the transcript, the harness substitutes nothing, and the notice is
        merged as ``mask=0``, so it never enters the advantage. That is what separates a
        refusal from the substitutions this loop used to perform, and why a refusal is
        deliberately not gated on ``validate``: a validation rollout gets it too, and
        recovers from an invalid action the way a real environment lets it.

        Args:
            agent_data: Live per-rollout agent state, holding the refused calls.
            reason: Marker for the notice (``agentic_tool_refused reason=…``).
            guidance: Sentences describing what an acceptable turn looks like.
            flag: ``extra_fields`` key set True so the refusal rate is monitorable.

        Returns:
            ``AgentState.GENERATING`` when the notice fits, else ``TERMINATED``.
        """
        refused = list(agent_data.tool_calls)
        names = [str(getattr(call, "name", "") or "unknown") for call in refused]
        joined = ", ".join(names)
        text = (
            f"Refused: {guidance} "
            f"{joined} in this turn was not executed. "
            "Emit one tool call per turn and wait for its <tool_response> before the next call. "
            f"agentic_tool_refused reason={reason} n={len(names)} names={joined}"
        )
        previous_messages = list(agent_data.messages)
        for call in refused:
            message: dict[str, Any] = {"role": "tool", "content": text}
            tool_call_id = getattr(call, "tool_call_id", None)
            if tool_call_id is not None:
                message["tool_call_id"] = tool_call_id
            agent_data.messages.append(message)
        # Nothing ran, so the calls must not reach the processing state: there is no
        # response to merge and no pass to count.
        agent_data.tool_calls = []

        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
            previous_messages,
            agent_data.messages,
            agent_data.prompt_ids,
            agent_data.response_mask,
            agent_data.response_logprobs if agent_data.response_logprobs else None,
            tools=schemas,
        )
        if len(response_mask) >= self.response_length:
            return AgentState.TERMINATED
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = response_logprobs or []
        agent_data.extra_fields[flag] = True
        logger.info(
            "Refused %s at global_step=%s: %s",
            reason,
            getattr(self, "_agentic_step", 0),
            joined,
        )
        return AgentState.GENERATING

    async def _agentic_refuse_premature_judge(self, agent_data: AgentData) -> AgentState:
        """Answer a ``judge_image`` call that has no image to inspect.

        ``REFLECT_SYSTEM_PROMPT`` says "Always generate before judging", but a policy
        still sometimes judges first, and the tool cannot satisfy that: there is no live
        PNG. The honest response is an observation saying so.

        The harness used to answer this by *replacing* the call with
        ``generate_image(prompt=<raw user request>)`` and marking the replacement
        ``mask=1``. That deleted the policy's own action, trained the bare restatement
        both system prompts forbid, and put harness text inside the advantage. A tool
        response keeps the call visible, leaves the prompt to the policy, and costs one
        generation round-trip on a slip that should be rare.

        Args:
            agent_data: Live per-rollout agent state, holding the refused calls.

        Returns:
            ``AgentState.GENERATING`` when the notice fits, else ``TERMINATED``.
        """
        return await self._agentic_refuse_tool_calls(
            agent_data,
            reason="no_image",
            guidance=(
                "there is no image to judge yet. Call generate_image for the user's request "
                "and wait for its <tool_response>, then judge the image it returns."
            ),
            flag="refused_premature_judge",
        )

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

    @staticmethod
    def _agentic_last_tool_message(agent_data: AgentData) -> dict[str, Any] | None:
        """Return the trailing tool message, or ``None`` when there is none.

        Args:
            agent_data: Live per-rollout agent state.

        Returns:
            Last ``role="tool"`` message when it is the most recent message.
        """
        for message in reversed(agent_data.messages):
            if message.get("role") != "tool":
                return None
            return message
        return None

    def _agentic_dropped_tool_calls(self, agent_data: AgentData) -> list[Any]:
        """Return the tool calls the parent will not execute this turn.

        Args:
            agent_data: Live per-rollout agent state.

        Returns:
            Calls beyond ``multi_turn.max_parallel_calls``, in emission order.
        """
        limit = int(getattr(self, "max_parallel_calls", 1) or 1)
        return list(agent_data.tool_calls[limit:])

    async def _agentic_append_dropped_tool_notice(self, agent_data: AgentData, dropped: list[Any]) -> AgentState:
        """Report dropped same-turn tool calls back to the actor as tool responses.

        ``ToolAgentLoop`` executes only ``tool_calls[:max_parallel_calls]`` and says
        nothing about the rest. A model that emits ``generate_image`` and
        ``judge_image`` in one message therefore lost its judge with no observation
        and no error, while the scorer still saw the textual call and docked
        ``judge_parse_ok``. Appending one tool response per dropped call keeps the
        message list well-formed and turns a silent harness truncation into a
        visible, learnable signal.

        Args:
            agent_data: Live per-rollout agent state, already holding the executed
                tool responses.
            dropped: Calls skipped by the parent this turn.

        Returns:
            ``AgentState.GENERATING`` when the notice fits, else ``TERMINATED``.
        """
        names = [str(getattr(call, "name", "") or "unknown") for call in dropped]
        limit = max(1, int(getattr(self, "max_parallel_calls", 1) or 1))
        joined = ", ".join(names)
        text = (
            f"Ignored: at most {limit} tool call(s) run per turn. "
            f"{joined} in this turn was not executed. "
            "Emit one tool call per turn and wait for its <tool_response> before the next call. "
            f"agentic_tool_dropped n={len(names)} names={joined}"
        )
        previous_messages = list(agent_data.messages)
        for call in dropped:
            message: dict[str, Any] = {"role": "tool", "content": text}
            tool_call_id = getattr(call, "tool_call_id", None)
            if tool_call_id is not None:
                message["tool_call_id"] = tool_call_id
            agent_data.messages.append(message)

        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
            previous_messages,
            agent_data.messages,
            agent_data.prompt_ids,
            agent_data.response_mask,
            agent_data.response_logprobs if agent_data.response_logprobs else None,
            tools=schemas,
        )
        if len(response_mask) >= self.response_length:
            return AgentState.TERMINATED
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = response_logprobs or []
        agent_data.extra_fields["num_dropped_tool_calls"] = len(names)
        agent_data.extra_fields["dropped_tool_names"] = ",".join(names)
        logger.info(
            "Ignored %d extra tool call(s) beyond max_parallel_calls=%s at global_step=%s: %s",
            len(names),
            getattr(self, "max_parallel_calls", 1),
            getattr(self, "_agentic_step", 0),
            ", ".join(names),
        )
        return AgentState.GENERATING

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

    def _agentic_exempt_from_forced_actions(self) -> bool:
        """Whether the harness must not inject cues or substitute actions.

        Only validation is exempt. ``force_first_generate_probability`` already
        short-circuits on ``validate``, but the Reflection cue and the action
        substitutions did not, so val transcripts — and val advantages — were partly
        curriculum. Exempting all of them makes a val rollout measure what the policy
        chooses.

        Checked before the ``force_reflection_after_judge`` knob so a validation exemption
        cannot be re-enabled by config, while that knob keeps its full meaning for
        training. A refusal that only reports a fact back to the actor — see
        :meth:`_agentic_refuse_premature_judge` — is not harness-authored model text and
        is deliberately not gated here.

        Reflect and plan rollouts are *not* distinguished here. Both data sources run the
        same loop; the difference between them lives in the system prompt, which is what
        makes the two validation reward curves comparable.

        Returns:
            ``True`` when the harness must not inject cues or substitute actions.
        """
        return bool(getattr(self, "_agentic_validate", False))

    def _agentic_continue_after_plan_turn(
        self,
        agent_data: AgentData,
        state: AgentState,
        *,
        messages_before: int,
    ) -> AgentState | None:
        """Reopen one tool-call-free *plan* turn as ``GENERATING``.

        The stock loop ends a rollout the moment an assistant turn carries no tool
        call, because a finished policy writes prose and stops. The plan data is built on
        a two-step shape: the agent writes the numbered plan, and only calls
        ``generate_image`` on the next turn. Without this override every plan rollout
        would end at the plan with zero images.

        This is the one place the loop looks at what a turn *contains*. It is deliberately
        not a ``task_type`` branch: the gate is the numbered list itself, so reflect and
        plan rollouts run the same code and it is the system prompt that decides which
        one writes prose first. A reflect rollout calls the tool on turn one, so this
        never fires for it.

        Only the no-tool-call exit appends an assistant message — the length and turn
        budget exits return before that — so growth in ``agent_data.messages``
        identifies this termination as "the model stopped talking" rather than "the
        harness stopped it". The continuation is bounded to one turn
        (``plan_turn_seen``) and to the window before the first image, so a policy that
        rambles after generating, or writes prose twice, still terminates on the stock
        rules.

        Args:
            agent_data: Live per-rollout agent state, already post-generation.
            state: State returned by the stock generating handler.
            messages_before: ``len(agent_data.messages)`` recorded before that call.

        Returns:
            ``AgentState.GENERATING`` to consume the plan turn, else ``None`` to keep
            the caller's state.
        """
        if state != AgentState.TERMINATED:
            return None
        if agent_data.tool_calls or agent_data.extra_fields.get("plan_turn_seen"):
            return None
        if len(agent_data.messages) <= messages_before:
            return None
        if agent_data.messages[-1].get("role") != "assistant":
            return None
        if count_successful_generates(agent_data.messages) > 0:
            return None
        if not plan_lines_from_prose(_assistant_content_text(agent_data.messages[-1])):
            return None
        agent_data.extra_fields["plan_turn_seen"] = True
        logger.info(
            "Plan turn written at global_step=%s; continuing to the first generate_image",
            getattr(self, "_agentic_step", 0),
        )
        return AgentState.GENERATING

    @staticmethod
    def _agentic_last_assistant_prose(agent_data: AgentData) -> str:
        """Return the text of the most recent assistant turn, or ``""``.

        Args:
            agent_data: Live per-rollout agent state.

        Returns:
            Assistant text with multimodal parts joined; empty when the last turn is not
            an assistant turn.
        """
        for message in reversed(agent_data.messages):
            if message.get("role") != "assistant":
                continue
            return _assistant_content_text(message)
        return ""

    def _record_force_first_swap_rejected(self, agent_data: AgentData, *, reason: str) -> None:
        agent_data.extra_fields["force_first_swap_rejected"] = True
        logger.info(
            "Force-first Hermes swap rejected (%s) at global_step=%s; keeping TERMINATED state",
            reason,
            getattr(self, "_agentic_step", 0),
        )

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Teacher-force missing generate/judge tool calls during early curriculum."""
        messages_before = len(agent_data.messages)
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        plan_state = self._agentic_continue_after_plan_turn(agent_data, state, messages_before=messages_before)
        if plan_state is not None:
            return plan_state
        probability = force_first_generate_probability(
            getattr(self, "_agentic_step", 0),
            validate=getattr(self, "_agentic_validate", False),
        )
        agent_data.extra_fields.setdefault("forced_first_generate", False)
        agent_data.extra_fields.setdefault("forced_first_judge", False)
        agent_data.extra_fields.setdefault("refused_premature_judge", False)
        agent_data.extra_fields["force_first_probability"] = float(probability)

        # A judge with no image to inspect is a protocol slip the tool cannot satisfy.
        # Answer it with an observation and let the policy author the generate call —
        # never with a substituted action, which would train harness text as the policy's
        # own. Not gated on protocol/validate: a refusal is environment feedback, so plan
        # and validation rollouts recover from the slip exactly as a real env lets them.
        n_gen = count_successful_generates(agent_data.messages)
        if (
            agentic_get_bool("refuse_premature_judge")
            and state == AgentState.PROCESSING_TOOLS
            and n_gen == 0
            and tool_calls_are_premature_judge(agent_data.tool_calls)
            and len(agent_data.response_mask) < self.response_length
        ):
            return await self._agentic_refuse_premature_judge(agent_data)

        active_tools = getattr(agent_data, "_active_tools", self.tools)
        # Do not gate on ``len(response_mask) >= response_length`` here: stock
        # ToolAgentLoop length-terminates *before* tool extraction, and random
        # / tiny models often fill the budget with prose. Force-first *replaces*
        # that last assistant span, so budget is checked inside
        # ``_replace_last_assistant_with_tool_call`` after the swap.
        #
        # Every rollout gets the same substitutions; nothing here branches on the data
        # source. Reflect and plan differ only in their system prompt. A reflect rollout
        # calls the tool on turn one, so there is nothing to replace. A plan rollout
        # writes its plan first, and that turn was already reopened as ``GENERATING`` by
        # :meth:`_agentic_continue_after_plan_turn` above, so this block never sees it.
        if (
            state != AgentState.TERMINATED
            or agent_data.tool_calls
            or probability <= 0.0
            or random.random() >= probability
        ):
            return state

        # Parent may have terminated on max_assistant_turns before tool extract —
        # do not reopen PROCESSING_TOOLS past that budget.
        max_turns = getattr(self, "max_assistant_turns", None)
        if max_turns and int(getattr(agent_data, "assistant_turns", 0)) >= int(max_turns):
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
                self._record_force_first_swap_rejected(agent_data, reason="generate_image_over_budget")
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
                self._record_force_first_swap_rejected(agent_data, reason="judge_image_over_budget")
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
        # Capture what the harness is about to drop: the parent slices
        # ``tool_calls[:max_parallel_calls]`` and never reports the rest.
        dropped = self._agentic_dropped_tool_calls(agent_data)
        state = await super()._handle_processing_tools_state(agent_data)
        # Stamp generate-count after every tool turn so discard_invalid_rollouts
        # can run in generate_sequences before the reward manager writes keys.
        # ``num_generate_image_prompts`` is the *executed* count (any live
        # ``agentic_tool ok=`` response), matching ``num_judge_image_calls`` in the
        # scorer. ``rollout_has_generate``/``rollout_valid`` stay on the *successful*
        # count: a call refused by the pass cap produced no image and must not keep
        # an otherwise-empty rollout alive. ``discard_invalid_rollouts`` reads those
        # two stamps before falling back to this counter, so masking is unchanged.
        n_gen = count_successful_generates(agent_data.messages)
        agent_data.extra_fields["num_generate_image_prompts"] = int(count_executed_generates(agent_data.messages))
        agent_data.extra_fields["rollout_has_generate"] = int(n_gen >= 1)
        agent_data.extra_fields["rollout_valid"] = int(n_gen >= 1)
        if state == AgentState.TERMINATED:
            return state

        # Resolve this from the *executed* responses, before the drop notice lands,
        # so forced Reflection still quotes the real judge/generate feedback.
        last_tool = self._agentic_last_tool_message(agent_data)
        if last_tool is None:
            return state

        if dropped:
            state = await self._agentic_append_dropped_tool_notice(agent_data, dropped)
            if state == AgentState.TERMINATED:
                return state

        gen_passes = n_gen
        max_passes = max_generate_passes()
        force_done = gen_passes >= max_passes
        # Both cues are gated by one predicate, and it exempts validation only: a val
        # rollout must measure the policy rather than the curriculum. Reflect and plan
        # training both keep them, so the two corpora see the same loop.
        force_reflection = agentic_get_bool("force_reflection_after_judge")
        if self._agentic_exempt_from_forced_actions():
            force_reflection = False
            force_done = False
        if not force_reflection and not force_done:
            return state

        forced = build_forced_reflection(
            tool_message_text(last_tool),
            force_done=force_done,
            generate_pass=gen_passes,
            max_passes=max_passes,
        )
        if forced is None:
            return state

        reflection_text, stop_required = forced
        if not force_reflection:
            # Stop cue only: ``force_reflection_after_judge`` is off, so a continue cue
            # must not be authored, and the stop cue is ours to author only when the cap
            # is what ended the run.
            if not stop_required:
                return state
            if not force_done:
                return state
        reflection_text = f"{reflection_text} agentic_forced_reflection=1"
        assistant_msg = {"role": "assistant", "content": reflection_text}
        response_ids = await self._encode_assistant_completion(reflection_text)
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
            force_reflection,
        )
        return AgentState.GENERATING
