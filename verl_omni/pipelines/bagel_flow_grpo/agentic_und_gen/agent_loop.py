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

"""Serial Bagel UND→GEN Co-RL (Joint-Training) agent loop (RFC phase A / PR1)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
import zlib
from functools import partial
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.utils import derive_rollout_seed
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.config import bagel_scorer_knobs_from_config
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode import (  # noqa: F401
    GENERATE_IMAGE_TOOL_SCHEMA,
    JUDGE_IMAGE_TOOL_SCHEMA,
    UND_STOP_SEQUENCES,
    BagelGenerateImageTool,
    EpisodeRollout,
    GenSample,
    conditioning_uid,
    generate_retrying_engine_wake,
    resolve_task_type,
    run_serial_episode,
    serving_rollout_provenance,
    strip_reflection_tail,
    trusted_image_reward_fields,
    und_turn_max_tokens,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode import (
    with_reflection_system_prompt as _with_reflection_system_prompt,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.prompts import prepare_bagel_prompt
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.reward import get_bagel_rm_gen_handle, make_rm_score_fn
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import (
    build_gen_sampling_params,
    stash_gen_row_from_diffusion_output,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.turn_protocol import derive_good_enough_from_scores
from verl_omni.tools.trajectory import EpisodeState, episode_scope
from verl_omni.tools.trajectory.episode_artifacts import get_latest_generate_prompt_for_active_rollout
from verl_omni.tools.trajectory.episode_context import (
    get_active_user_prompt,
)
from verl_omni.utils.agentic.artifacts import (
    build_trajectory_relpath,
    rollout_id_from_relpath,
)
from verl_omni.utils.agentic.config import agentic_get
from verl_omni.utils.agentic.image_gen_rollout_parse import last_user_prompt
from verl_omni.utils.agentic_val_viz import VAL_VIZ_PROMPT_KEY, save_val_viz_images

logger = logging.getLogger(__name__)


def _log_engine_asleep_retry(
    _logger, attempt: int, delay: float, exc: Exception, *, lane: str, step: Any = None
) -> None:
    """Report a re-issue of a decode the engine rejected mid-wake."""
    _logger.warning(
        "bagel_corl %s lane: engine rejected a decode mid-wake (step=%s attempt=%d, retrying in %.1fs) -- "
        "the wake call returns when the request is dispatched, not when the memory is mapped: %s",
        lane,
        step,
        attempt + 1,
        delay,
        exc,
    )


def _messages_are_text_only(messages: Any) -> bool:
    """True when every message carries plain text (no image/video/audio parts).

    The UND prompt is rendered from ``raw_prompt`` only when it is text: a multimodal
    message would need the dataset's own tokenizer run (image placeholder expansion),
    so those prompts keep the ids they came with.
    """
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        content = msg.get("content")
        if content is None or isinstance(content, str):
            continue
        if isinstance(content, list):
            if any(isinstance(part, dict) and part.get("type") not in (None, "text") for part in content):
                return False
            continue
        return False
    return True


def _template_renders_hermes_tools(tokenizer: Any) -> bool:
    """True when the checkpoint's chat template knows the Hermes tool-call grammar."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return "<tools>" in template and "<tool_call>" in template


def _inject_tools_schema_enabled() -> bool:
    """``BAGEL_CORL_INJECT_TOOLS=0`` opts out (diagnostics / non-Hermes UND models)."""
    return os.getenv("BAGEL_CORL_INJECT_TOOLS", "1") not in ("0", "false", "False", "")


# Episode-level reduction over the S per-seed ``good_enough`` flags. SoT is
# ``actor_rollout_ref.rollout.agent.good_enough_reduction``; unknown values fail
# loud rather than silently degrading to best-of-S (see ``agentic_und_gen.episode``).
_GOOD_ENOUGH_REDUCTIONS = frozenset({"any", "all", "mean"})

# RFC §5: knobs the agent loop must see but must NOT invent a value for.
_BAGEL_AGENT_REQUIRED_KNOBS = ("gen_samples_per_call", "max_generate_passes", "max_und_turns")


def resolve_bagel_agent_knobs(agent_cfg: Any) -> dict[str, Any]:
    """Resolve the ``bagel_multiturn_agent`` knobs, failing loud on missing ones."""
    missing = [key for key in _BAGEL_AGENT_REQUIRED_KNOBS if agent_cfg.get(key) is None]
    if missing:
        raise ValueError(
            "bagel_multiturn_agent requires actor_rollout_ref.rollout.agent."
            f"{', '.join(missing)} (RFC §5 knob SoT, no code default)"
        )
    reduction = str(agent_cfg.get("good_enough_reduction", "any"))
    if reduction not in _GOOD_ENOUGH_REDUCTIONS:
        raise ValueError(
            "bagel_multiturn_agent actor_rollout_ref.rollout.agent.good_enough_reduction "
            f"must be one of {sorted(_GOOD_ENOUGH_REDUCTIONS)}, got {reduction!r}"
        )
    return {
        # S = FlowGRPO seeds under one generate_image turn. Not episode GEN-turn count K.
        "gen_samples_per_call": int(agent_cfg.get("gen_samples_per_call")),
        "max_generate_passes": int(agent_cfg.get("max_generate_passes")),
        "max_und_turns": int(agent_cfg.get("max_und_turns")),
        "good_enough_reduction": reduction,
    }


def episode_to_agent_output(
    episode: EpisodeRollout,
    *,
    timing: dict[str, Any] | None = None,
    relpath: str | None = None,
) -> AgentLoopOutput:
    """Project a finished episode onto the ``AgentLoopOutput`` the v1 TQ path expects."""
    extra = {
        "text_encoder_responses": "",
        "prompt_ids": episode.prompt_ids,
        "response_ids": episode.response_ids,
        "response_mask": episode.response_mask,
        "gen_samples": episode.gen_samples,
        "und_group_uid": episode.und_group_uid,
        "episode_uid": episode.episode_uid,
        "used_image_credit": episode.used_image_credit,
        "bagel_corl_reward": dict(episode.bagel_corl_reward),
        "corl_reward_version": episode.corl_reward_version,
        "autonomous_done": episode.autonomous_done,
        "forced_reflection": episode.forced_reflection,
        **{k: v for k, v in episode.bagel_corl_reward.items() if k.startswith("image_reward_")},
        "turns": episode.turns,
        "num_gen_calls": episode.num_gen_calls,
        "und_reward": episode.und_reward,
        "policy_version": episode.policy_version,
        "rollout_policy_version": episode.policy_version,
        "requested_policy_version": episode.requested_policy_version,
        "rollout_replica_id": episode.rollout_replica_id,
        "serving_provenance_verified": episode.serving_provenance_verified,
        "und_serving_provenance": list(episode.und_serving_provenance),
        "min_global_steps": episode.policy_version,
        "max_global_steps": episode.policy_version,
        "bagel_role_timing": dict(timing or {}),
        "llm_all_log_probs": None,
        "trajectory_relpath": relpath or "",
    }
    return AgentLoopOutput(
        prompt_ids=episode.prompt_ids,
        response_ids=episode.response_ids,
        response_mask=episode.response_mask,
        # ``None`` when the episode sampled nothing (replay/stub): the trainer then gets no
        # rollout distribution for that row instead of an all-zero one it would trust.
        response_logprobs=list(episode.rollout_log_probs) or None,
        num_turns=episode.turns,
        metrics=AgentLoopMetrics(),
        reward_score=None,
        extra_fields=extra,
    )


def episode_ground_truth_fields(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Read reward metadata with the same task precedence as the episode scorer."""
    extra = kwargs.get("extra_info")
    extra = extra if isinstance(extra, dict) else {}
    reward_model = kwargs.get("reward_model")
    reward_model = reward_model if isinstance(reward_model, dict) else {}
    raw = kwargs.get("ground_truth", reward_model.get("ground_truth", extra.get("ground_truth", {})))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    keys = {
        "task_type",
        "reference_subtasks",
        "reference_steps",
        "expected_num_images",
        "plan_expected",
        "user_request",
    }
    keys.update("w_" + dim for dim in ("reflect", "plan", "format", "protocol", "tool", "result", "image", "done"))
    ground_truth = {key: value for key, value in extra.items() if key in keys}
    ground_truth.update({key: value for key, value in raw.items() if key in keys})
    task_type = resolve_task_type(extra_info=extra, ground_truth=raw)
    if task_type is not None:
        ground_truth["task_type"] = task_type
    return ground_truth


def episode_seed_index(episode_key: str) -> int:
    """Deterministic per-episode integer for the GEN seed base."""
    return int(zlib.crc32(str(episode_key).encode("utf-8")))


def episode_artifact_keys(
    *,
    dataset_task_uid: str,
    session_id: Any,
    global_steps: Any,
    seed: Any = None,
) -> tuple[str, int]:
    """Return ``(relpath, seed_base)`` for one episode."""
    episode_key = f"{dataset_task_uid}:{session_id}"
    relpath = build_trajectory_relpath(
        step=global_steps,
        sample_index=dataset_task_uid,
        rollout_n=int(session_id or 0) + 1,
    )
    seed_base = derive_rollout_seed(
        int(seed or global_steps or 0),
        episode_seed_index(episode_key),
    )
    return relpath, seed_base


@register("bagel_multiturn_agent")
class BagelMultiturnAgentLoop(AgentLoopBase):
    """Per-episode serial UND→GEN→RM loop. Outer gather stays on the worker."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Stash for the GEN tool (``_generate_image``) so it can build diffusion
        # sampling params (seed / logprobs / num_inference_steps) per FlowGRPO seed.
        self._sampling_params = sampling_params
        dataset_task_uid = str(
            kwargs.get("dataset_task_uid")
            or kwargs.get("uid")
            or (kwargs.get("extra_info") or {}).get("dataset_task_uid")
            or "missing_task"
        )
        # This is the requested trainer step for episode IDs, not evidence of
        # which weights served the tokens. The decode outputs supply that version.
        requested_policy_version = int(
            sampling_params.get("global_steps", kwargs.get("global_steps", kwargs.get("policy_version", 0))) or 0
        )
        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("bagel_multiturn_agent requires raw_prompt")
        # Transformers 4/5: tokenize=True may return BatchEncoding; list(enc) would iterate keys ('input_ids', ...)
        # and crash normalize_token_ids on the server.
        prompt_ids = kwargs.get("prompt_ids")
        task_type = resolve_task_type(extra_info=kwargs.get("extra_info"), ground_truth=kwargs.get("ground_truth"))
        prepared_prompt = prepare_bagel_prompt(raw_prompt, task_type=task_type, extra_info=kwargs.get("extra_info"))
        if prepared_prompt is not raw_prompt:
            raw_prompt = prepared_prompt
            prompt_ids = None
        injected_tools = False
        if (
            _inject_tools_schema_enabled()
            and _messages_are_text_only(raw_prompt)
            and _template_renders_hermes_tools(self.tokenizer)
        ):
            # Re-render from the messages rather than patching the dataset's ids: the ``<tools>``
            # block belongs inside the system turn, ahead of its text.
            #
            # ``task_type`` decides whether the dataset's own system turn survives: plan rows must
            # keep ``PLAN_SYSTEM_PROMPT`` (see ``with_reflection_system_prompt``), whose
            # numbered-plan sentence is what makes ``R_plan`` / ``R_result`` reachable.
            encoded = self.tokenizer.apply_chat_template(
                _with_reflection_system_prompt(raw_prompt, task_type=task_type),
                tools=[GENERATE_IMAGE_TOOL_SCHEMA, JUDGE_IMAGE_TOOL_SCHEMA],
                add_generation_prompt=True,
                tokenize=True,
            )
            prompt_ids = normalize_token_ids(encoded)
            injected_tools = True
        elif prompt_ids is None:
            encoded = self.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=True)
            prompt_ids = normalize_token_ids(encoded)
        else:
            prompt_ids = normalize_token_ids(prompt_ids)
        if os.getenv("BAGEL_CORL_DEBUG") == "1":
            logger.info(
                "bagel_corl_prompt uid=%s tools_injected=%s prompt_tokens=%d tail=%r",
                dataset_task_uid,
                injected_tools,
                len(prompt_ids),
                # The tail is the generation prompt (``<|im_start|>assistant\n``) -- cheap proof
                # that the rendered prompt ends where the engine expects it to.
                self.tokenizer.decode(prompt_ids[-24:], skip_special_tokens=False),
            )

        agent_cfg = self.config.actor_rollout_ref.rollout.agent
        # Response kind the RM worker's reward manager will dtype-check the in-loop
        # payload row against. ``output_type`` is a real ``DiffusionPipelineConfig``
        # field, so ``get`` + a literal fallback is only defensive; the fallback
        # matches the field's own default.
        rollout_pipeline = self.config.actor_rollout_ref.rollout.get("pipeline") or {}
        knobs = resolve_bagel_agent_knobs(agent_cfg)
        s = knobs["gen_samples_per_call"]
        max_passes = knobs["max_generate_passes"]
        max_und_turns = knobs["max_und_turns"]
        # good_enough SoT: agentic_image_gen.good_enough_threshold (yaml, bound at
        # worker init). Reduction over per-seed flags defaults to best-of-S ("any").
        good_enough_threshold = float(agentic_get("good_enough_threshold"))
        good_enough_reduction = knobs["good_enough_reduction"]
        # RFC §4.4.2b (R2): conditioning reuse only pays off when the whole S-group
        # is pinned to one replica, because each replica owns its own cache. Read the
        # affinity knob once per episode rather than per seed.
        rollout_cfg = self.config.actor_rollout_ref.rollout
        # Episode context budget: ``_und_decode`` feeds the engine ``prompt_ids +
        # response_ids``, the trainer pads every trajectory to
        # ``rollout.prompt_length + rollout.response_length``, and the AR engine's
        # ``max_model_len`` is sized against that same sum. The turn loop therefore has
        # to stop there -- see ``run_serial_episode``'s ``max_context_tokens``.
        max_context_tokens = int(rollout_cfg.prompt_length) + int(rollout_cfg.response_length)
        max_response_tokens = int(rollout_cfg.response_length)
        self._bagel_cond_affinity = bool(
            getattr(rollout_cfg, "enable_prompt_embed_cache", False)
            and getattr(rollout_cfg, "enable_prompt_embed_cache_routing_affinity", False)
        )
        # Role timing (RFC §8 KPIs): per-episode wall-clock split by lane.
        timing = {"und_decode_s": 0.0, "gen_s": 0.0, "rm_s": 0.0}
        self._bagel_timing = timing
        # In-loop RM (RFC §4.2): score each generate_image call mid-episode via the
        # GEN-side reward handle bound by the worker. UniCoT reference paths ride on
        # the row's extra_info (stamped by the dataset builder / driver).
        rm_handle = get_bagel_rm_gen_handle()
        if rm_handle is not None:
            extra_info = kwargs.get("extra_info") if isinstance(kwargs.get("extra_info"), dict) else {}
            ref_paths = extra_info.get("reference_image_path")
            self._rm_reference_paths = (
                [str(p) for p in ref_paths]
                if isinstance(ref_paths, list | tuple)
                else ([str(ref_paths)] if ref_paths else [])
            )
            self._rm_extra_info = dict(extra_info)
            self._rm_score_fn = make_rm_score_fn(
                rm_handle,
                get_reference_paths=lambda: list(self._rm_reference_paths),
                get_extra_info=lambda: {**self._rm_extra_info, "user_prompt": get_active_user_prompt() or ""},
                get_scorer_knobs=lambda: dict(bagel_scorer_knobs_from_config(self.config)),
                get_image_prompt=lambda: get_latest_generate_prompt_for_active_rollout() or "",
                # The RM worker's reward manager validates the row's ``responses``
                # against the rollout pipeline's response kind, so the un-used
                # placeholder must match it (image → uint8, latent → float).
                output_type=str(rollout_pipeline.get("output_type", "image")),
            )
        else:
            self._rm_score_fn = None

        async def _und_decode(**_decode_kwargs):
            t0 = time.perf_counter()
            try:
                und_params = dict(sampling_params)
                und_params["bagel_role"] = "und"
                # π_rollout for the UND lane: the recipe runs with ``rollout.calculate_log_probs=True``, so the v1
                # trainer reads ``rollout_log_probs`` off the TQ next to our recomputed ``old_log_probs``
                # (``trainer_base.py:1506-1509``); without it the step dies with ``KeyError: 'rollout_log_probs'``.
                und_params["logprobs"] = 0
                # Bound this turn, and cut it at the call boundary. See
                # ``und_turn_max_tokens`` / ``UND_STOP_SEQUENCES``: the AR strategy's default cap
                # is the whole episode response budget, so without an explicit ``max_tokens`` a
                # single degenerate turn (role-label repetition, or the invented ``<output>``
                # spiral after a call) consumes the episode and nothing is left to train on.
                _context_used = len(_decode_kwargs["prompt_ids"]) + len(_decode_kwargs["response_ids"])
                _turn_max = und_turn_max_tokens(
                    context_used=_context_used,
                    max_context_tokens=min(max_context_tokens, len(_decode_kwargs["prompt_ids"]) + max_response_tokens),
                )
                if _turn_max > 0:
                    und_params["max_tokens"] = _turn_max
                und_params.setdefault("stop", list(UND_STOP_SEQUENCES))
                und_params.setdefault("include_stop_str_in_output", True)
                # UND must hit the AR replica (BagelDualRoleLLMServerClient routes on role /
                # absence of diffusion keys). Never send num_inference_steps here.
                for _k in (
                    "num_inference_steps",
                    "noise_level",
                    "sde_window_size",
                    "sde_window_range",
                    "sde_type",
                    "height",
                    "width",
                ):
                    und_params.pop(_k, None)
                try:
                    output = await generate_retrying_engine_wake(
                        partial(
                            self.server_manager.generate,
                            request_id=str(uuid.uuid4()),
                            prompt_ids=list(_decode_kwargs["prompt_ids"]) + list(_decode_kwargs["response_ids"]),
                            sampling_params=und_params,
                        ),
                        on_retry=partial(
                            _log_engine_asleep_retry,
                            logger,
                            lane="und",
                            step=getattr(self, "global_steps", None),
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 — map diffusion-replica errors to UND dual-role failure
                    err = str(exc)
                    if "num_inference_steps" in err or "Diffusion" in type(exc).__name__:
                        raise RuntimeError(
                            "Bagel Co-RL (Joint-Training) UND decode hit the GEN diffusion replica "
                            "(bagel_single_stage / DiffusionStrategy). UND needs AR TokenOutput "
                            "(Hermes tool-call); refuse soft-empty TQ. "
                            "Prove dual-role serving, then set agent.und_ar_serving_ready=True. "
                            f"Underlying error: {err}"
                        ) from exc
                    raise
                if hasattr(output, "diffusion_output"):
                    raise RuntimeError(
                        "Bagel Co-RL (Joint-Training) UND decode received DiffusionOutput: "
                        "rollout is on DiffusionStrategy (bagel_single_stage / output_mode≠ar). "
                        "UND needs AR token generation; "
                        "refuse empty token_ids that would leave TQ with no materializable trajectories."
                    )
                if not hasattr(output, "token_ids"):
                    raise RuntimeError(
                        "Bagel Co-RL (Joint-Training) UND decode expected TokenOutput with token_ids, "
                        f"got {type(output)!r}"
                    )
                token_ids = list(output.token_ids)
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                step_log_probs = getattr(output, "log_probs", None)
                if step_log_probs is None or len(step_log_probs) != len(token_ids):
                    # Fail loud rather than publish a fabricated ratio: the trainer pairs these
                    # with ``old_log_probs`` for the π_rollout/π_θ correction, so a silent
                    # zero-fill would corrupt the objective instead of erroring.
                    raise RuntimeError(
                        "Bagel Co-RL (Joint-Training) UND decode got no per-token log-probs "
                        f"(log_probs={None if step_log_probs is None else len(step_log_probs)} for "
                        f"{len(token_ids)} tokens). The AR replica must be asked with "
                        "sampling_params['logprobs']=0 and its engine started with a logprobs_mode."
                    )
                provenance = serving_rollout_provenance(
                    dict(getattr(output, "extra_fields", None) or {}), required=True
                )
                return {
                    "token_ids": token_ids,
                    "text": text,
                    "log_probs": [float(v) for v in step_log_probs],
                    **provenance,
                }
            finally:
                timing["und_decode_s"] += time.perf_counter() - t0

        # Rollout-scoped artifact context: registrations inside the episode land under
        # this trajectory in the unified tools.trajectory registry (audit T1.6).
        global_steps = sampling_params.get("global_steps", kwargs.get("global_steps"))
        session_id = kwargs.get("session_id", kwargs.get("index", 0))
        # GEN group seed base: episode-scoped, so the S FlowGRPO seeds are not the constant ``range(S)`` they used to
        # be.
        relpath, seed_base = episode_artifact_keys(
            dataset_task_uid=dataset_task_uid,
            session_id=session_id,
            global_steps=global_steps,
            seed=sampling_params.get("seed"),
        )
        tool = BagelGenerateImageTool(
            gen_samples_per_call=s,
            max_generate_passes=max_passes,
            generate_fn=self._generate_image,
            seed_base=seed_base,
        )
        # Non-image UND scalar for pattern-3 (K=0) episodes. The reward model computes
        # the authoritative token-GRPO reward post-hoc (via the worker's ``_compute_score``);
        # this is only a fallback used when no RM is attached.
        non_image_reward = kwargs.get("non_image_reward")
        # ``relpath`` is precomputed beside ``seed_base`` so the folder and the noise
        # draw share one episode identity (see ``episode_artifact_keys``).
        user_prompt_text = last_user_prompt(raw_prompt)
        episode_ground_truth = episode_ground_truth_fields(kwargs)
        try:
            expected_num_images = max(1, int(episode_ground_truth.get("expected_num_images") or 1))
        except (TypeError, ValueError):
            expected_num_images = 1
        with episode_scope(
            EpisodeState(
                episode_id=rollout_id_from_relpath(relpath), artifact_relpath=relpath, user_prompt=user_prompt_text
            )
        ):
            episode = await run_serial_episode(
                dataset_task_uid=dataset_task_uid,
                policy_version=requested_policy_version,
                prompt_ids=list(prompt_ids),
                und_decode=_und_decode,
                generate_tool=tool,
                score_fn=self._score_gen_samples,
                tokenizer=self.tokenizer,
                max_und_turns=max_und_turns,
                max_context_tokens=max_context_tokens,
                max_response_tokens=max_response_tokens,
                non_image_reward=float(non_image_reward) if non_image_reward is not None else None,
                good_enough_threshold=good_enough_threshold,
                good_enough_reduction=good_enough_reduction,
                corl_reward_version=str(bagel_scorer_knobs_from_config(self.config).get("corl_reward_version", "v1")),
                fixed_gen_prompt=str(self.config.actor_rollout_ref.rollout.agent.get("corl_fixed_gen_prompt", "")),
                task_type=episode_ground_truth.get("task_type"),
                expected_num_images=expected_num_images,
            )
        if str(kwargs.get(VAL_VIZ_PROMPT_KEY, "")).lower() in {"1", "true", "yes", "on"}:
            save_val_viz_images(
                [sample.image_path for sample in episode.gen_samples if sample.valid and sample.image_path], relpath
            )
        return episode_to_agent_output(episode, timing=self._bagel_timing, relpath=relpath)

    async def _generate_image(self, **kwargs) -> list[dict[str, Any]]:
        """Run **one** GEN turn: S same-conditioning FlowGRPO seeds, then return."""
        prompt = strip_reflection_tail(str(kwargs.get("prompt", "")))
        seeds = list(kwargs.get("seeds") or [])
        if not prompt:
            raise ValueError("_generate_image requires a non-empty diffusion prompt")

        rollout = self.config.actor_rollout_ref.rollout
        base = dict(getattr(self, "_sampling_params", None) or {})
        # The validation overlay disables trajectory log-probabilities.
        is_validate = base.get("logprobs") is False
        rows: list[dict[str, Any]] = []
        timing = getattr(self, "_bagel_timing", None)
        # BagelPipeline reads text from the request prompt; encode the diffusion prompt directly (not the whole UND
        # conversation) so the GEN replica denoises the right conditioning.
        gen_prompt_ids = normalize_token_ids(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not gen_prompt_ids:
            raise ValueError("_generate_image requires a non-empty tokenized diffusion prompt")
        cond_uid = conditioning_uid(gen_prompt_ids)
        cond_len = len(gen_prompt_ids)
        # RFC §4.4.2b: one sticky key for the whole S-group. ``LLMServerClient.generate``
        # acquires its replica from the pool's load balancer with this id, so passing
        # the *same* key to every seed pins the group to the replica that then holds
        # the conditioning entry and lets seeds 2..S hit it. A per-seed ``uuid4()``
        # scattered the group across replicas and made a cold cache guaranteed.
        affinity = bool(getattr(self, "_bagel_cond_affinity", False))
        routing_key = self._cond_routing_key(kwargs.get("gen_call_id") or cond_uid) if affinity else None
        for seed in seeds:
            t0 = time.perf_counter()
            try:
                request_params = build_gen_sampling_params(rollout, base=base, seed=int(seed))
                request_params["bagel_role"] = "gen"
                # Same wake-window rejection as the UND lane, and the same fix: the engine
                # rejects before executing, so re-issuing is safe. The request id is the S-group's
                # sticky ``routing_key``, so reusing it is required -- a fresh uuid would scatter
                # the group across replicas and cold the conditioning cache (RFC §4.4.2b).
                output = await generate_retrying_engine_wake(
                    partial(
                        self.server_manager.generate,
                        request_id=routing_key or str(uuid.uuid4()),
                        prompt_ids=gen_prompt_ids,
                        sampling_params=request_params,
                    ),
                    on_retry=partial(
                        _log_engine_asleep_retry,
                        logger,
                        lane="gen",
                        step=getattr(self, "global_steps", None),
                    ),
                )
                # Training packs the FlowGRPO trajectory and fails closed without it;
                # validation decodes ODE (``noise_level=0.0``) so it has no SDE
                # log-probs and must not be required to carry one. See
                # ``stash_gen_row_from_diffusion_output`` for the split.
                row = stash_gen_row_from_diffusion_output(output, seed=int(seed), require_traj=not is_validate)
                # Live serving must identify its own GEN version/replica; a
                # parent UND or requested trainer step cannot supply these.
                row.update(serving_rollout_provenance(row, required=True))
                row["cond_uid"] = cond_uid
                row["cond_len"] = cond_len
                rows.append(row)
            finally:
                if timing is not None:
                    timing["gen_s"] += time.perf_counter() - t0
        return rows

    def _cond_routing_key(self, group_key: str) -> str:
        """Sticky replica key for one GEN S-group (RFC §4.4.2b).

        Namespaced so it cannot collide with a routing id another pool minted.
        """
        return f"bagel_cond::{group_key}"

    async def _score_gen_samples(self, samples: list[GenSample]) -> list[GenSample]:
        """Score GEN samples via an injected RM hook (C/A, UniCoT similarity, good_enough)."""
        rm_fn = getattr(self, "_rm_score_fn", None)
        if rm_fn is None:
            for sample in samples:
                if sample.rm_score is not None and sample.good_enough is None:
                    sample.good_enough = derive_good_enough_from_scores(
                        correctness=float(sample.rm_score),
                        aesthetics=float(sample.rm_score),
                    )
            return samples
        t0 = time.perf_counter()
        try:
            result = rm_fn(samples)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            timing = getattr(self, "_bagel_timing", None)
            if timing is not None:
                timing["rm_s"] += time.perf_counter() - t0
