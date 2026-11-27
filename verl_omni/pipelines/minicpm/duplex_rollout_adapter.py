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
"""MiniCPM-o 4.5 duplex rollout bridge (``pipeline_mode="duplex"``).

Clones vLLM-Omni's native MiniCPM-o 4.5 stage 0 with its duplex runtime intact
(the data plane, the MRv2 duplex sampler and the duplex plugin are engine
properties of that stage) and registers the clone for RL training. Duplex
rollout is session-driven, not ``generate``-driven: the rollout server builds
``DuplexOmni`` through the same engine-args path as ``AsyncOmni`` and the
duplex agent loop drives one real session per episode
(RFC verl-omni #905 §3.4). The engine-side surface this bridge consumes —
action capture, policy log-probabilities, seeded replay — is the contract of
vllm-project/vllm-omni#10627; at the current verl-omni pin those symbols do
not exist, so every engine import here is lazy and fails loudly with the
required upstream revision.
"""

from __future__ import annotations

import os
from typing import Any

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

from verl_omni.pipelines.minicpm.omni_rollout_adapter import MiniCPMORolloutAdapter
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.utils.dataset.duplex.replay import (
    DUPLEX_REPLAY_KEY,
    DuplexReplay,
    build_replay,
    extract_unit_mels,
    resolve_duplex_vocabulary,
)

#: The recipe exports this to match ``rollout.agent.num_workers``. The adapter
#: has no rollout config at the deploy-override call.
DUPLEX_MAX_SESSIONS_ENV = "VERL_OMNI_DUPLEX_MAX_SESSIONS"
DUPLEX_MAX_SESSIONS_DEFAULT = 16

__all__ = [
    "MINICPMO_4_5_DUPLEX_PIPELINE",
    "MiniCPMODuplexRolloutAdapter",
    "require_duplex_vllm_omni",
]

#: The vLLM-Omni revision window this bridge consumes (vllm-omni#10627).
REQUIRED_DUPLEX_UPSTREAM = "vllm-project/vllm-omni#10627 (duplex action capture)"

MINICPMO_4_5_DUPLEX_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5_duplex",
    # Same routing contract and duplex plugin as the native 3-stage pipeline;
    # only the topology is cut to the duplex Thinker stage.
    model_arch=MINICPMO_4_5_PIPELINE.model_arch,
    duplex_plugin=MINICPMO_4_5_PIPELINE.duplex_plugin,
    hf_architectures=MINICPMO_4_5_PIPELINE.hf_architectures,
    hf_config_predicate=MINICPMO_4_5_PIPELINE.hf_config_predicate,
    stages=(MINICPMO_4_5_PIPELINE.stages[0],),
)


def require_duplex_vllm_omni() -> Any:
    """Import the duplex capture surface, failing loudly at the pin.

    The verl-omni pin ships the duplex runtime (``DuplexOmni``) but predates
    the action-capture surface (``DuplexOmni.collect_action_records`` /
    ``vllm_omni.protocol.duplex.action_records``); a duplex rollout configured
    against the pin must fail with the requirement, not with an opaque error
    deep inside the engine. The probe checks the post-close surface the
    strategy drains (``_drain_terminal``).

    Returns:
        The ``DuplexOmni`` engine-client class.

    Raises:
        RuntimeError: When the installed vLLM-Omni lacks either half.
    """
    try:
        from vllm_omni.entrypoints.duplex_omni import DuplexOmni
    except ImportError as error:
        raise RuntimeError(
            f"duplex rollout requires a vLLM-Omni build with the duplex runtime "
            f"({REQUIRED_DUPLEX_UPSTREAM}); the installed revision does not provide "
            f"it ({error})."
        ) from error
    if not callable(getattr(DuplexOmni, "collect_action_records", None)):
        raise RuntimeError(
            "duplex rollout requires a vLLM-Omni build with the duplex action-capture "
            f"surface ({REQUIRED_DUPLEX_UPSTREAM}); the installed revision's duplex "
            "runtime has no collect_action_records. verl-omni's pinned revision stays "
            "turn-based — see the duplex GRPO recipe for the pin-override instructions."
        )
    return DuplexOmni


@OmniRolloutPipelineBase.register("minicpmo_4_5_duplex")
class MiniCPMODuplexRolloutAdapter(MiniCPMORolloutAdapter):
    """Rollout adapter for MiniCPM-o 4.5 streaming duplex sessions.

    Registered under ``model_type="minicpmo_4_5_duplex"`` (select with
    ``engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5_duplex`` and
    ``output_mode=duplex``). Stage topology is the native duplex Thinker stage;
    ``thinker_only`` stays implemented by the parent.
    """

    #: Declares the session-driven duplex rollout path to the agent loop.
    duplex_rollout_supported = True

    @classmethod
    def build_stage_configs(cls, pipeline_mode="duplex"):
        if pipeline_mode != "duplex":
            raise ValueError(
                f"MiniCPMODuplexRolloutAdapter implements duplex only, got {pipeline_mode!r}. "
                "Text-output thinker training uses pipeline_name=minicpmo_4_5."
            )
        stages = list(MINICPMO_4_5_DUPLEX_PIPELINE.stages)
        assert len(stages) == 1, (
            f"Expected the single duplex Thinker stage, got {len(stages)}. "
            "The runtime clone of MINICPMO_4_5_PIPELINE.stages[0] is wrong."
        )
        return stages

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "duplex") -> str:
        cls.build_stage_configs(pipeline_mode)
        return MINICPMO_4_5_DUPLEX_PIPELINE.model_type

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode: str = "duplex") -> None:
        cls.build_stage_configs(pipeline_mode)
        register_pipeline(MINICPMO_4_5_DUPLEX_PIPELINE)

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "duplex") -> dict:
        """The duplex runtime owns the stage's model and scheduling; no AR extras."""
        cls.build_stage_configs(pipeline_mode)
        if stage_id != 0:
            raise ValueError(f"MiniCPM-o duplex rollout has no stage {stage_id}.")
        return {}

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode: str = "duplex") -> dict:
        return {}

    @classmethod
    def deploy_config_overrides(cls, pipeline_mode: str = "duplex") -> dict:
        """Serve duplex on MRv2 and set the standalone session cap.

        Capture publishes only on model runner v2. ``DUPLEX_MAX_SESSIONS_ENV``
        is the admission cap when no rollout config is in scope. The AR
        strategy overwrites ``duplex_session.max_sessions`` from
        ``rollout.agent.num_workers`` when it writes the training deploy.
        """
        cls.build_stage_configs(pipeline_mode)
        raw = os.environ.get(DUPLEX_MAX_SESSIONS_ENV, str(DUPLEX_MAX_SESSIONS_DEFAULT))
        try:
            max_sessions = int(raw)
        except ValueError as exc:
            raise ValueError(f"{DUPLEX_MAX_SESSIONS_ENV} must be an integer, got {raw!r}") from exc
        if max_sessions < 1:
            raise ValueError(f"{DUPLEX_MAX_SESSIONS_ENV} must be >= 1, got {max_sessions}")
        return {
            "session_mode": "duplex",
            "model_runner": "v2",
            "duplex_session": {"max_sessions": max_sessions},
        }

    @classmethod
    def build_duplex_replay(
        cls,
        trajectory,
        *,
        processor: Any,
        instructions: str | None = None,
        model_path: str | None = None,
    ) -> DuplexReplay:
        """Build one episode's replay and re-derive its per-unit mel features.

        Args:
            trajectory: The assembled :class:`DuplexTrajectory`.
            processor: The checkpoint's processor (streaming audio API).
            instructions: The session's system prompt; must equal the value
                the rollout driver opened the session with.
            model_path: The actor checkpoint path — the snapshot the engine
                resolved the bundled reference audio (``assets/HT_ref_audio.wav``)
                from, so live rollouts embed the same reference block; unset
                defers to ``VERL_OMNI_DUPLEX_REF_AUDIO_MODEL_PATH``.

        Returns:
            The replay with ``unit_mels`` attached for the actor's media inputs.
        """
        vocabulary = resolve_duplex_vocabulary(processor)
        replay = build_replay(trajectory, vocabulary, instructions=instructions, model_path=model_path)
        replay.unit_mels, replay.unit_mel_lens = extract_unit_mels(processor, replay)
        return replay

    @classmethod
    def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length: int):
        """Map one duplex episode onto the policy sequence verl trains on.

        ``extra_fields`` must carry the built replay payload under
        ``"duplex_replay_payload"``; the prompt becomes the session's system
        context and the response the interleaved unit-window sequence, with
        the validity mask marking the sampled actions only.
        """
        payload = output.extra_fields.get("duplex_replay_payload")
        if payload is None:
            raise RuntimeError(
                "MiniCPM-o duplex rollout did not return a replay payload; the duplex "
                "agent loop must assemble the trajectory and build the replay first."
            )
        if not isinstance(payload, DuplexReplay):
            raise TypeError(f"duplex replay payload must be a DuplexReplay, got {type(payload).__name__}.")
        if DUPLEX_REPLAY_KEY in output.extra_fields:
            raise RuntimeError(f"duplex rollout unexpectedly returned reserved field {DUPLEX_REPLAY_KEY!r}.")
        if payload.response_ids:
            output.prompt_ids = list(payload.prompt_ids)
            output.response_ids = list(payload.response_ids)
            output.response_mask = list(payload.response_mask)
        else:
            # An empty response cannot hold rm_scores[-1]. One masked token
            # keeps the reward write and stays out of the loss.
            output.prompt_ids = list(payload.prompt_ids) or [0]
            output.response_ids = [0]
            output.response_mask = [0]
            if not output.extra_fields.get("advantage_mask"):
                output.extra_fields["advantage_mask"] = [0]
        # ``response_logprobs`` stays unexported: verl pops it into the
        # ``rollout_log_probs`` column and engages rollout-correction /
        # debug metrics against the trainer-recomputed ``old_log_probs`` —
        # engine-captured and trainer-recomputed logprobs must never mix
        # (a 0.1–0.5-nat systematic offset). The captured logprobs ride the
        # replay payload for the M0 parity gate only.
        output.extra_fields[DUPLEX_REPLAY_KEY] = payload.to_extra_fields(include_pcm=False)
        del output.extra_fields["duplex_replay_payload"]
        return output
