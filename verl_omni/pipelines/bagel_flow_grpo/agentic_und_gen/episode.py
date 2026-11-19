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

"""The serial UND/GEN episode loop for Bagel Co-RL (Joint-Training).

J UND turns drive up to K ``generate_image`` calls of S FlowGRPO seeds each; the RM verdict
on every call is answered on the policy's ``judge_image`` call and followed by the Mode-2a
reflection. This module owns the loop only; records (``samples``), parsing (``tool_calls``), the GEN
tool (``gen_tool``), rewards (``und_reward``) and metrics (``episode_metrics``) are siblings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode_state import (
    EpisodeSpec,
    EpisodeTranscript,
    ServingLedger,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.gen_tool import (
    BagelGenerateImageTool,
    build_generate_call_meta,
    compact_image_observation,
    resolve_gen_prompt,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.judge_text import judge_text_from_gen_samples
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.samples import EpisodeRollout, GenSample, bind_episode_ids
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import _force_info_logging
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.tool_calls import (
    is_terminal_only,
    parse_und_tool_call,
    strip_reflection_tail,
    und_turn_kind,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.turn_protocol import build_forced_reflection
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.und_reward import (
    image_credit_reward,
    trusted_image_reward_fields,
)
from verl_omni.tools.trajectory.episode import EpisodeState, current_episode, episode_scope
from verl_omni.tools.trajectory.episode_artifacts import register_tool_artifact
from verl_omni.tools.trajectory.episode_context import get_active_user_prompt
from verl_omni.tools.trajectory.episode_judge import (
    clear_good_enough_yes_reached,
    get_good_enough_yes_reached,
    set_good_enough_yes_reached,
)

logger = logging.getLogger(__name__)

#: ``BAGEL_CORL_DEBUG=1`` re-enables INFO diagnostics (vLLM raises the root level to ERROR) and
#: the per-turn trace below.
if os.getenv("BAGEL_CORL_DEBUG") == "1":
    _force_info_logging()
_BAGEL_CORL_TURN_DEBUG = os.getenv("BAGEL_CORL_DEBUG") == "1"


class _SerialEpisode:
    """One serial UND episode in flight: turn handlers over an ``EpisodeTranscript``."""

    def __init__(self, ids: dict[str, Any], spec: EpisodeSpec, transcript: EpisodeTranscript):
        self.ids = ids
        self.spec = spec
        self.t = transcript
        self.serving = ServingLedger()
        self.gen_samples: list[GenSample] = []
        self.turns = 0
        self.num_gen_calls = 0
        self.images_rendered = 0
        self.used_image_credit = False
        self.forced = False
        self.autonomous_done = False
        self.stop_required = False
        self.early_judge_declined = False
        self.judge_text: str | None = None
        self.pending_judge: tuple[str, str, list[GenSample]] | None = None

    @property
    def _tag(self) -> str:
        return str(self.ids["episode_uid"])[:8]

    @property
    def _v2(self) -> bool:
        return self.spec.corl_reward_version == "v2"

    async def _decode(self) -> dict[str, Any]:
        decode = self.spec.und_decode(prompt_ids=self.t.prompt_ids, response_ids=self.t.response_ids)
        step = await decode if asyncio.iscoroutine(decode) else decode
        self.serving.observe_und(step, response_start=len(self.t.response_ids))
        return step

    def _env_turn(self, body: str, step: dict[str, Any], text: str, default_ids: list[int] | None = None) -> None:
        self.t.append_env_turn(body, text, list(step.get("obs_token_ids") or default_ids or []), turn=self.turns)

    def _record_und_turn(self, step: dict[str, Any], kind: str, req_ctx: int, started: float | None) -> None:
        text = str(step.get("text") or "")
        record = {
            "record": "und_turn",
            "turn": self.turns,
            "kind": kind,
            "req_ctx": req_ctx,
            "out_tokens": len(step.get("token_ids") or []),
            "text": text,
        }
        if started is not None:
            record["decode_s"] = round(time.perf_counter() - started, 3)
        self.t.trace.append(record)
        if _BAGEL_CORL_TURN_DEBUG and started is not None:
            logger.info(
                "bagel_corl_turn episode=%s turn=%d kind=%s out_tokens=%d decode_s=%.1f text=%r",
                self._tag,
                self.turns,
                kind,
                record["out_tokens"],
                time.perf_counter() - started,
                text[:240],
            )

    async def run(self) -> EpisodeRollout:
        """Drive the turn loop to its end and assemble the ``EpisodeRollout``."""
        for _ in range(self.spec.max_und_turns):
            if self.t.budget_spent():
                break
            if await self._turn():
                break
        else:
            self.forced = True
        return self._finish()

    async def _turn(self) -> bool:
        """One UND decode plus its environment response. Returns True when the episode ends."""
        self.turns += 1
        if _BAGEL_CORL_TURN_DEBUG:
            logger.info(
                "bagel_corl_turn episode=%s turn=%d req_ctx=%d resp_tokens=%d",
                self._tag,
                self.turns,
                self.t.context_len,
                len(self.t.response_ids),
            )
        started = time.perf_counter()
        req_ctx = self.t.context_len
        step = await self._decode()
        text = str(step["text"])
        kind = und_turn_kind(text)
        self._record_und_turn(step, kind, req_ctx, started)
        self.t.append_response(list(step["token_ids"]), 1, step.get("log_probs"))
        if kind == "done":
            return self._on_done(text, step)
        if kind == "judge_image":
            return await self._on_judge(text, step)
        if kind == "generate_image":
            return await self._on_generate(text, step)
        return False

    def _on_done(self, text: str, step: dict[str, Any]) -> bool:
        """``Done.`` is accepted after good_enough=YES or the final GEN call; earlier it is declined."""
        tool = self.spec.generate_tool
        remaining = tool.remaining_passes()
        if remaining > 0 and not get_good_enough_yes_reached() and not is_terminal_only(text):
            next_step = (
                "call judge_image on the new image"
                if self.pending_judge is not None
                else "rewrite the diffusion prompt from the judge's findings and call generate_image"
            )
            self._env_turn(
                "agentic_stop ok=0\n"
                f"good_enough=YES has not been reached and {remaining} of "
                f"{tool.max_generate_passes} generate_image calls remain: {next_step}. "
                "Done. is accepted after good_enough=YES or the final generate_image call.",
                step,
                text,
            )
            return False
        self.autonomous_done = not self.forced
        return True

    async def _on_judge(self, text: str, step: dict[str, Any]) -> bool:
        """Answer ``judge_image`` with the held RM verdict (declined once before the plan is rendered)."""
        if self.pending_judge is None:
            self._env_turn(
                "agentic_judge ok=0 parse_ok=0 stub=0 backend=reward_loop\n"
                "No new image to judge: call generate_image first.",
                step,
                text,
            )
            return False
        if (
            not self.early_judge_declined
            and self.images_rendered < self.spec.planned_images
            and self.spec.generate_tool.remaining_passes() > 0
        ):
            self.early_judge_declined = True
            self._env_turn(
                "agentic_judge ok=0 parse_ok=1 stub=0 backend=reward_loop\n"
                f"Plan step {self.images_rendered}/{self.spec.planned_images}: call generate_image for subtask "
                f"{self.images_rendered + 1} first. judge_image runs once, after the final subtask image.",
                step,
                text,
            )
            return False
        judged_text, judged_path, judged_samples = self.pending_judge
        self.pending_judge = None
        self._env_turn(f"VL judge on the last generated image:\n  path={judged_path}\n{judged_text}", step, text)
        return await self._reflect_on_judge(judged_text, judged_samples, step)

    async def _on_generate(self, text: str, step: dict[str, Any]) -> bool:
        """Run one ``generate_image`` call: render S seeds, score, and hold the judge verdict."""
        tool = self.spec.generate_tool
        if get_good_enough_yes_reached():
            return True
        if tool.remaining_passes() <= 0:
            self._env_turn(
                f"agentic_tool ok=0 images=0 backend=vllm_omni\n"
                f"generate_image pass cap ({tool.max_generate_passes}) reached; reply Done.",
                step,
                text,
            )
            return False
        call = parse_und_tool_call(text) or {}
        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        prompt = resolve_gen_prompt(
            strip_reflection_tail(str(arguments.get("prompt", ""))),
            fixed_gen_prompt=self.spec.fixed_gen_prompt,
            active_user_prompt=self.spec.active_user_prompt,
        )
        meta = build_generate_call_meta(prompt=prompt, user_prompt=self.spec.active_user_prompt)
        gen_call_id = (
            str(self.ids["gen_call_id"]) if tool.passes == 0 else f"{self.ids['episode_uid']}:call{self.num_gen_calls}"
        )
        self.num_gen_calls += 1
        call_samples = await tool(
            prompt=prompt, prompt_token_ids=self.t.prompt_ids + self.t.response_ids, gen_call_id=gen_call_id
        )
        call_samples = sorted(call_samples, key=lambda sample: sample.seed_index)
        self.serving.check_gen(call_samples)
        call_role = str(meta.get("call_role") or "initial")
        for sample in call_samples:
            sample.call_role = call_role
            sample.gen_call_index = self.num_gen_calls - 1
        valid_paths = [str(s.image_path) for s in call_samples if s.valid and s.image_path]
        if valid_paths:
            register_tool_artifact(prompt=prompt, paths=valid_paths, call_id=gen_call_id)
        self.t.trace.append(
            {
                "record": "gen_call",
                "turn": self.turns,
                "gen_call_id": gen_call_id,
                "call_role": call_role,
                "prompt": prompt,
                "num_samples": len(call_samples),
                "num_valid": sum(1 for s in call_samples if s.valid),
                "image_paths": valid_paths,
            }
        )
        if self.spec.score_fn is not None:
            scored = self.spec.score_fn(call_samples)
            call_samples = await scored if asyncio.iscoroutine(scored) else scored
            self.serving.check_gen(call_samples)
        self.gen_samples.extend(call_samples)
        observation = self._gen_observation(valid_paths)
        call_judge_text = await self._judge_text(call_samples)
        if valid_paths or call_judge_text:
            self._env_turn(observation, step, text, default_ids=[0])
        if call_judge_text:
            self.judge_text = call_judge_text
            self.pending_judge = (call_judge_text, valid_paths[0] if valid_paths else "", call_samples)
            return False
        forced_ids = list(step.get("forced_token_ids") or [])
        if forced_ids:
            self.forced = True
            self.t.append_response(forced_ids, 0)
        elif step.get("done_token_ids"):
            self.t.append_response(list(step["done_token_ids"]), 1)
        return True

    def _gen_observation(self, valid_paths: list[str]) -> str:
        if not valid_paths:
            return (
                "agentic_tool ok=0 images=0 backend=vllm_omni\n"
                "Image generation failed; rewrite the prompt and call generate_image again."
            )
        self.used_image_credit = True
        self.images_rendered += 1
        observation = (
            f"vLLM-Omni generated the requested image. {compact_image_observation(valid_paths[0])} "
            f"agentic_tool ok=1 images={len(valid_paths)} backend=vllm_omni"
        )
        if self.images_rendered < self.spec.planned_images and self.spec.generate_tool.remaining_passes() > 0:
            observation += (
                f"\nPlan step {self.images_rendered}/{self.spec.planned_images} rendered. Next: call generate_image "
                f"with the complete prompt for subtask {self.images_rendered + 1} of your plan."
            )
        return observation

    async def _judge_text(self, call_samples: list[GenSample]) -> str | None:
        text = None
        if self.spec.build_judge_text_fn is not None:
            maybe = self.spec.build_judge_text_fn(call_samples)
            text = await maybe if asyncio.iscoroutine(maybe) else maybe
        if text:
            return text
        return judge_text_from_gen_samples(
            call_samples,
            reduction="all" if self._v2 else self.spec.good_enough_reduction,
            threshold=self.spec.good_enough_threshold,
        )

    async def _reflect_on_judge(self, verdict: str, judged: list[GenSample], judged_step: dict[str, Any]) -> bool:
        """Inject the Mode-2a reflection after a judge response; True ends the episode."""
        tool = self.spec.generate_tool
        remaining = tool.remaining_passes()
        reflection = build_forced_reflection(
            verdict,
            force_done=remaining == 0,
            generate_pass=tool.max_generate_passes - remaining,
            max_passes=tool.max_generate_passes,
        )
        if reflection is None:
            return False
        reflection_text, self.stop_required = reflection
        forced_ids = self.t.encode(reflection_text + "\n", fallback_ids=list(judged_step.get("forced_token_ids") or []))
        if forced_ids:
            self.forced = True
            self.t.append_response(forced_ids, 0)
        self.t.trace.append({"record": "reflection", "turn": self.turns, "text": reflection_text})
        reached = ("good_enough=YES" in verdict) or (
            not self._v2 and any(s.good_enough for s in judged if s.good_enough is not None)
        )
        state = current_episode()
        judged_call = judged[-1].gen_group_uid if judged else None
        if state is not None and any(record.call_id == judged_call for record in state.generations):
            state.record_judgment(judged_call, good_enough=reached)
        elif reached:
            set_good_enough_yes_reached(True)
        if not self.stop_required:
            return False
        await self._closing_done(judged_step)
        return True

    async def _closing_done(self, judged_step: dict[str, Any]) -> None:
        """Let the policy write its own ``Done.`` after a stop reflection, else append the literal."""
        done_step: dict[str, Any] = {} if self.t.budget_spent() else await self._decode()
        done_text = str(done_step.get("text") or "")
        try:
            done_kind = und_turn_kind(done_text)
        except ValueError as exc:
            logger.warning("bagel_corl_und_done_decode_unparsed err=%s; using Done. fallback", exc)
            done_kind = "continue"
        if done_step:
            self._record_und_turn(done_step, done_kind, self.t.context_len, None)
        done_ids = list(done_step.get("token_ids") or [])
        if done_ids and done_kind == "done":
            self.t.append_response(done_ids, 1, done_step.get("log_probs"))
            return
        fallback = self.t.encode(
            "Done.", fallback_ids=list(done_step.get("done_token_ids") or judged_step.get("done_token_ids") or [])
        )
        if fallback:
            self.t.append_response(fallback, 0 if self._v2 else 1)

    def _finish(self) -> EpisodeRollout:
        trusted_fields = (
            trusted_image_reward_fields(
                self.gen_samples,
                num_gen_calls=self.num_gen_calls,
                autonomous_done=self.autonomous_done,
                forced_reflection=self.forced,
            )
            if self._v2
            else {}
        )
        image_credit = (
            trusted_fields.get("image_reward_objective") if trusted_fields else image_credit_reward(self.gen_samples)
        )
        if image_credit is not None:
            und_reward = image_credit
        elif self.spec.non_image_reward is not None:
            und_reward = float(self.spec.non_image_reward)
        else:
            und_reward = 0.0
        if _BAGEL_CORL_TURN_DEBUG:
            logger.info(
                "bagel_corl_episode episode=%s turns=%d K=%d resp_tokens=%d forced=%s stop_required=%s reward=%.3f",
                self._tag,
                self.turns,
                self.num_gen_calls,
                len(self.t.response_ids),
                self.forced,
                self.stop_required,
                und_reward,
            )
        self.t.truncate_to_budget(self._tag)
        return EpisodeRollout(
            und_group_uid=str(self.ids["und_group_uid"]),
            episode_uid=str(self.ids["episode_uid"]),
            policy_version=self.serving.und_version,
            requested_policy_version=int(self.ids["policy_version"]),
            rollout_replica_id=self.serving.replica_id,
            serving_provenance_verified=self.serving.verified,
            und_serving_provenance=self.serving.provenance,
            prompt_ids=list(self.t.prompt_ids),
            response_ids=self.t.response_ids,
            response_mask=self.t.response_mask,
            rollout_log_probs=self.t.log_probs,
            turns=self.turns,
            gen_samples=self.gen_samples,
            used_image_credit=self.used_image_credit,
            forced_reflection=self.forced,
            autonomous_done=self.autonomous_done,
            corl_reward_version=self.spec.corl_reward_version,
            bagel_corl_reward=trusted_fields,
            und_reward=und_reward,
            judge_text=self.judge_text,
            stop_required=self.stop_required,
            num_gen_calls=self.num_gen_calls,
            turn_trace=self.t.trace,
        )


async def run_serial_episode(
    *,
    dataset_task_uid: str,
    policy_version: int,
    prompt_ids: list[int],
    und_decode: Callable[..., Any],
    generate_tool: BagelGenerateImageTool,
    score_fn: Callable[[list[GenSample]], Any] | None = None,
    build_judge_text_fn: Callable[[list[GenSample]], str | None] | None = None,
    tokenizer: Any | None = None,
    user_prompt: str = "",
    max_und_turns: int = 8,
    max_context_tokens: int | None = None,
    max_response_tokens: int | None = None,
    forced_reflection_text: str = "Done.",
    episode_uid: str | None = None,
    non_image_reward: float | None = None,
    good_enough_threshold: float | None = None,
    good_enough_reduction: str = "any",
    corl_reward_version: str = "v1",
    fixed_gen_prompt: str = "",
    task_type: str | None = None,
    expected_num_images: int = 1,
) -> EpisodeRollout:
    """Serial UND turn(s) -> optional GEN turns (S FlowGRPO seeds each) -> RM -> reflection / Done."""
    if corl_reward_version not in ("v1", "v2"):
        raise ValueError("corl_reward_version must be v1 or v2")
    ids = bind_episode_ids(dataset_task_uid=dataset_task_uid, episode_uid=episode_uid, policy_version=policy_version)
    with episode_scope(current_episode() or EpisodeState(episode_id=ids["episode_uid"], user_prompt=user_prompt)):
        clear_good_enough_yes_reached()
        if _BAGEL_CORL_TURN_DEBUG:
            _force_info_logging()
        spec = EpisodeSpec(
            und_decode=und_decode,
            generate_tool=generate_tool,
            score_fn=score_fn,
            build_judge_text_fn=build_judge_text_fn,
            max_und_turns=max_und_turns,
            non_image_reward=non_image_reward,
            good_enough_threshold=good_enough_threshold,
            good_enough_reduction=good_enough_reduction,
            corl_reward_version=corl_reward_version,
            fixed_gen_prompt=fixed_gen_prompt,
            planned_images=max(1, int(expected_num_images)) if str(task_type or "").strip().lower() == "plan" else 1,
            active_user_prompt=user_prompt or get_active_user_prompt() or "",
        )
        transcript = EpisodeTranscript(
            prompt_ids=prompt_ids,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_response_tokens=max_response_tokens,
        )
        return await _SerialEpisode(ids, spec, transcript).run()
