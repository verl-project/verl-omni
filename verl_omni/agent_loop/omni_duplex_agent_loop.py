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
"""The duplex agent loop: one real duplex session per episode.

Registered as ``omni_duplex_agent`` and dispatched through the same rollout
adapter registry as the other omni loops (the adapter must declare
``duplex_rollout_supported``). The loop itself stays model-agnostic: it turns
the dataset's episode spec into a server-side run spec (PCM chunks at the
native 1 s cadence with speech hints), makes one Ray call per episode batch
against the rollout server (the server drives the sessions next to the
engine — see ``DuplexStrategy``), assembles the returned capture scopes into
a :class:`DuplexTrajectory`, and hands it to the adapter for the
model-specific replay build and the standard ``AgentLoopOutput`` mapping
(the validated qwen3_tts transport pattern: one namespaced ``extra_fields``
replay key).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.utils.dataset.duplex.replay import (
    DEFAULT_SESSION_INSTRUCTIONS,
    DUPLEX_REPLAY_KEY,
    DuplexReplay,
)
from verl_omni.utils.dataset.duplex.trajectory import DuplexTrajectory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: When set, the agent loop appends one JSON line per episode's replay payload
#: (PCM and mel features stripped — the FDB input) to
#: ``<dir>/duplex_replay.jsonl``; unset writes nothing.
REPLAY_DUMP_DIR_ENV = "VERL_OMNI_DUPLEX_REPLAY_DUMP_DIR"
_REPLAY_DUMP_FILENAME = "duplex_replay.jsonl"

__all__ = ["OmniDuplexAgentLoop", "REPLAY_DUMP_DIR_ENV", "dump_duplex_replay"]


@register("omni_duplex_agent")
class OmniDuplexAgentLoop(AgentLoopBase):
    """Session-driven duplex rollout loop, generic over the rollout adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_adapter = self._resolve_rollout_adapter(self.rollout_config)

    @staticmethod
    def _resolve_rollout_adapter(rollout_config):
        try:
            pipeline_name = rollout_config.engine_kwargs["vllm_omni"]["pipeline_name"]
        except (KeyError, TypeError) as error:
            raise ValueError("omni_duplex_agent requires engine_kwargs.vllm_omni.pipeline_name.") from error
        adapter = OmniRolloutPipelineBase.get_class(pipeline_name)
        if adapter is None or not getattr(adapter, "duplex_rollout_supported", False):
            raise ValueError(
                f"omni_duplex_agent requires a duplex-capable rollout adapter, got {pipeline_name!r}. "
                "Duplex adapters declare duplex_rollout_supported (first: minicpmo_4_5_duplex)."
            )
        return adapter

    async def _episode_server(self, request_id: str) -> tuple[str, Any, Any]:
        """Pin one rollout server through the shared load balancer."""
        load_balancer = getattr(self.server_manager, "_load_balancer", None)
        if load_balancer is None:
            raise RuntimeError(
                "omni_duplex_agent needs the rollout load balancer; the configured rollout "
                "backend does not provide one."
            )
        server_id, server = await load_balancer.acquire_server.remote(request_id=request_id)
        return server_id, server, load_balancer

    def _run_spec(self, episode: dict[str, Any], *, seed: int, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Lift one dataset episode into the server-side run spec.

        No ``instructions`` key: an explicit value at ``open_session``
        silences the engine's native duplex pipeline, so the driver
        opens without it and the engine's duplex policy substitutes the same
        default system text. The text stays this loop's replay context only.
        """
        chunks = episode.get("chunks") or []
        if not chunks:
            raise ValueError(f"duplex episode {episode.get('episode_id')!r} carries no audio chunks")
        for chunk in chunks:
            if not isinstance(chunk, dict) or "pcm" not in chunk:
                raise ValueError(
                    f"duplex episode {episode.get('episode_id')!r} chunks must be "
                    "{'pcm': float array, 'is_speech': bool} mappings"
                )
        run_spec: dict[str, Any] = {
            "episode_id": str(episode["episode_id"]),
            "axis": str(episode["axis"]),
            "seed": int(seed),
            "chunks": chunks,
            "event": dict(episode.get("event") or {}),
            "temperature": float(sampling_params.get("temperature", 0.7)),
            "tail_horizon_s": float(episode.get("tail_horizon_s", 8.0)),
        }
        if episode.get("axis") == "interruption":
            run_spec["inject"] = {
                "delay_s": float(episode.get("inject_delay_s", 0.6)),
                "tail_silence_chunks": int(episode.get("tail_silence_chunks", 2)),
            }
            run_spec["speak_wait_horizon_s"] = float(episode.get("speak_wait_horizon_s", 30.0))
        if episode.get("commit"):
            run_spec["commit"] = True
        return run_spec

    def _actor_model_path(self) -> str | None:
        """The actor checkpoint path, as the rollout config knows it.

        The engine resolves the bundled reference audio
        (``assets/HT_ref_audio.wav``) from this snapshot, so the replay must
        too — ``None`` (path unset) builds a reference-less replay.
        """
        model_config = getattr(self.config, "actor_rollout_ref", None)
        path = getattr(getattr(model_config, "model", None), "path", None)
        return str(path) if path else None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Drive one episode end to end and map it onto the policy sequence."""
        episode = kwargs.get("duplex_episode")
        if not isinstance(episode, dict):
            raise TypeError(
                "omni_duplex_agent requires a duplex_episode mapping on every sample "
                "(produced by the duplex episode dataset)."
            )
        # Group members (rollout.n copies of one sample) share the dataset
        # row, so the dataset seed alone would make their sessions draw
        # identically whenever batch composition coincides — a degenerate
        # GRPO group. Each episode mixes in a fresh uuid; the effective seed
        # rides every sealed record set, so per-episode replay determinism
        # (same seed => same tokens) is preserved and auditable.
        seed = (int(kwargs.get("duplex_seed", 0) or 0) + uuid4().int) % (2**63)
        instructions = episode.get("instructions") or DEFAULT_SESSION_INSTRUCTIONS
        run_spec = self._run_spec(episode, seed=seed, sampling_params=sampling_params)
        step = _episode_global_steps(kwargs)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"duplex-{run_spec['episode_id']}-{uuid4().hex[:8]}"
            server_id, server, load_balancer = await self._episode_server(request_id)
            try:
                try:
                    result = await server.run_duplex_episode.remote(run_spec)
                except Exception as first_error:
                    # Failure tolerance: one re-drive with a fresh seed. A
                    # transient engine failure must not poison the GRPO
                    # group; the second failure degrades to a mask-0 episode
                    # below instead of failing the rollout batch.
                    logger.warning(
                        "duplex episode %s failed (%s: %s); re-driving with a fresh seed",
                        run_spec["episode_id"],
                        type(first_error).__name__,
                        first_error,
                    )
                    seed = (int(kwargs.get("duplex_seed", 0) or 0) + uuid4().int) % (2**63)
                    run_spec = self._run_spec(episode, seed=seed, sampling_params=sampling_params)
                    result = await server.run_duplex_episode.remote(run_spec)
            except Exception as error:
                return self._degenerate_output(
                    episode,
                    seed=seed,
                    instructions=instructions,
                    step=step,
                    metrics=metrics,
                    error=error,
                )
            finally:
                await load_balancer.release_server.remote(server_id=server_id, request_id=request_id)

        # The server reports every chunk it actually fed (interruption episodes
        # add driver-generated silence around the injected speech); prefer it
        # over the spec's chunks whenever the server provides the list.
        client_chunks = [list(chunk) for chunk in result.get("client_chunks") or []]
        if not client_chunks:
            client_chunks = [chunk["pcm"] for chunk in run_spec["chunks"]]
        trajectory = DuplexTrajectory.assemble(
            session_id=result["session_id"],
            seed=seed,
            event=_episode_event(result["event"] or episode.get("event") or {}),
            record_set_wires=result["record_sets"],
            client_audio_chunks=client_chunks,
        )
        trajectory = _reanchor_interruption(trajectory, result.get("speech_chunk_end_index"), seed)
        replay = self.rollout_adapter.build_duplex_replay(
            trajectory,
            processor=self.processor,
            instructions=instructions,
            model_path=self._actor_model_path(),
        )
        output = AgentLoopOutput(
            prompt_ids=[0],
            response_ids=[],
            response_mask=[],
            num_turns=2,
            metrics=metrics,
            # verl's _compute_metrics reads these tags off every trajectory
            # (off-policy staleness); None would crash the int cast.
            extra_fields={"duplex_replay_payload": replay, "min_global_steps": step, "max_global_steps": step},
        )
        output = self._postprocess(output)
        policy_length = len(output.response_ids)
        if policy_length > self.rollout_config.response_length:
            raise ValueError(
                f"Omni policy output has {policy_length} tokens, exceeding "
                f"response_length={self.rollout_config.response_length}."
            )
        if len(output.response_mask) != policy_length:
            raise ValueError(
                "Omni policy response_mask must align one-to-one with response_ids, "
                f"got {len(output.response_mask)} and {policy_length}."
            )
        if output.response_logprobs is not None and len(output.response_logprobs) != policy_length:
            raise ValueError(
                "Omni policy response_logprobs must align one-to-one with response_ids, "
                f"got {len(output.response_logprobs)} and {policy_length}."
            )
        dump_duplex_replay(output.extra_fields, episode_id=run_spec["episode_id"])
        return output

    def _degenerate_output(
        self,
        episode: dict[str, Any],
        *,
        seed: int,
        instructions: str,
        step: Any,
        metrics: dict[str, Any],
        error: Exception,
    ) -> AgentLoopOutput:
        """The mask-0 episode for a twice-failed drive.

        Bounded episodes must not poison the whole rollout group: the output
        carries an empty (zero-unit) replay payload, a zero loss mask, a zero
        reward score — which also keeps the scorer off the broken payload —
        and a ``duplex_episode_failed`` reward extra so the trainer's
        per-step aggregation surfaces the failure share.
        """
        logger.error(
            "duplex episode %s failed twice (%s: %s); emitting a degenerate mask-0 episode",
            episode.get("episode_id"),
            type(error).__name__,
            error,
        )
        replay = DuplexReplay(
            session_id=f"failed-{episode.get('episode_id', 'unknown')}",
            seed=seed,
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            audio_bounds=[],
            unit_layouts=[],
            unit_pcm=[],
            response_logprobs=[],
            event=dict(episode.get("event") or {}),
            status="failed",
            model_extension={},
            sampling_params=(),
            instructions=instructions,
        )
        output = AgentLoopOutput(
            prompt_ids=[0],
            response_ids=[],
            response_mask=[],
            num_turns=2,
            metrics=metrics,
            reward_score=0.0,
            extra_fields={
                "duplex_replay_payload": replay,
                "min_global_steps": step,
                "max_global_steps": step,
                "reward_extra_info": {"duplex_episode_failed": 1.0},
            },
        )
        return self._postprocess(output)

    def _postprocess(self, output: AgentLoopOutput) -> AgentLoopOutput:
        """Run the rollout adapter's boundary step and require an ``AgentLoopOutput`` back."""
        output = self.rollout_adapter.postprocess_agent_loop_output(
            output,
            tokenizer=self.tokenizer,
            response_length=self.rollout_config.response_length,
        )
        if not isinstance(output, AgentLoopOutput):
            raise TypeError(f"duplex rollout adapter must return an AgentLoopOutput, got {type(output).__name__}.")
        return output


def _episode_global_steps(kwargs: Mapping[str, Any]) -> int | None:
    """The trainer's global step for this episode, or ``None`` when absent.

    The TransferQueue agent-loop worker forwards the batch's
    ``global_steps`` column as a kwarg; it may arrive as a plain int, a 0-dim
    tensor, or a ``NonTensorData``.
    """
    raw = kwargs.get("global_steps")
    if raw is None:
        return None
    value = getattr(raw, "data", raw)
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


def dump_duplex_replay(extra_fields: Mapping[str, Any], *, episode_id: str) -> bool:
    """Append one episode's replay payload to the env-gated JSONL dump.

    With ``VERL_OMNI_DUPLEX_REPLAY_DUMP_DIR`` set, the payload under
    ``duplex_replay`` is appended as one line-buffered JSON line to
    ``<dir>/duplex_replay.jsonl``, PCM- and mel-free (the FDB input). Unset,
    nothing is written.

    Returns:
        Whether a line was written.
    """
    dump_dir = os.environ.get(REPLAY_DUMP_DIR_ENV)
    if not dump_dir:
        return False
    payload = extra_fields.get(DUPLEX_REPLAY_KEY)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"duplex replay dump needs the payload under {DUPLEX_REPLAY_KEY!r}, got {type(payload).__name__}"
        )
    line = {key: value for key, value in payload.items() if key not in ("unit_mels", "unit_mel_lens")}
    dump_path = Path(dump_dir) / _REPLAY_DUMP_FILENAME
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "a", buffering=1, encoding="utf-8") as handle:
        handle.write(json.dumps({"episode_id": episode_id, **line}, default=_json_dump_value) + "\n")
    return True


def _json_dump_value(value: Any) -> Any:
    """JSON encoding for the dump's non-native payload values (arrays, tensors)."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"duplex replay dump: {type(value).__name__} is not JSON-serializable")


def _reanchor_interruption(trajectory: DuplexTrajectory, speech_chunk_end_index: Any, seed: int):
    """Rewrite an interruption episode's yield deadline onto the unit clock.

    Mining writes the corpus-relative end of the interrupting speech, but the
    driver injects that speech relative to the policy's own speak onset — the
    driver reports which fed chunk closed the speech, and the i-th fed chunk
    maps one-to-one onto the i-th client-origin unit decision, so the anchor
    becomes that decision's engine emission time. Non-interruption episodes
    and missing indices pass through unchanged.
    """
    if trajectory.event.axis != "interruption" or speech_chunk_end_index is None:
        return trajectory
    index = int(speech_chunk_end_index)
    client_decisions = [
        decision for decision in trajectory.unit_decisions(listen_token_id=-1) if decision.origin == "client"
    ]
    if not 1 <= index <= len(client_decisions):
        raise ValueError(
            f"duplex episode {trajectory.session_id}: speech-end chunk index {index} outside the "
            f"{len(client_decisions)} fed chunks; the driver and the records disagree"
        )
    anchored = replace(trajectory.event, interrupt_speech_end_s=client_decisions[index - 1].time_s)
    return DuplexTrajectory.assemble(
        session_id=trajectory.session_id,
        seed=seed,
        event=anchored,
        record_set_wires=[record_set.to_wire() for record_set in trajectory.record_sets],
        client_audio_chunks=list(trajectory.client_audio_chunks),
    )


def _episode_event(event: dict[str, Any]) -> Any:
    """Rebuild the episode annotation from the server-echoed mapping."""
    from verl_omni.utils.dataset.duplex.trajectory import DuplexEpisodeEvent

    pause_window = event.get("pause_window_s")
    return DuplexEpisodeEvent(
        axis=str(event["axis"]),
        pause_window_s=tuple(pause_window) if pause_window else None,
        user_turn_end_s=event.get("user_turn_end_s"),
        interrupt_speech_end_s=event.get("interrupt_speech_end_s"),
    )
