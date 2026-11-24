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
"""Duplex trajectory contract for streaming duplex GRPO (RFC verl-omni #905).

The package owns the trainer side of the duplex data path: the wire mirror of
the engine's action records, trajectory assembly, unit-window replay, the
exact-distribution logprob recomputation, event mining, telemetry /
calibration / FDB-v1 metrics, and the replay dataset. Engine symbols are
imported lazily by the rollout bridge only.
"""

from verl_omni.utils.dataset.duplex.calibration import CalibrationReport, audit_calibration
from verl_omni.utils.dataset.duplex.records import (
    ACTION_FORCED,
    ACTION_REEMIT,
    ACTION_SAMPLED,
    DuplexActionRecordSet,
    RecordSetError,
)
from verl_omni.utils.dataset.duplex.replay import (
    DUPLEX_REPLAY_KEY,
    DuplexRefAudio,
    DuplexReplay,
    audio_tokens_for_mel_frames,
    build_replay,
    resolve_default_ref_audio,
    resolve_duplex_vocabulary,
)
from verl_omni.utils.dataset.duplex.sampling_logprobs import (
    ParityReport,
    compare_action_logprobs,
    engine_repetition_history,
    recompute_action_logprobs,
    unit_inputs_from_trajectory,
)
from verl_omni.utils.dataset.duplex.trajectory import (
    AXIS_INTERRUPTION,
    AXIS_PAUSE,
    AXIS_TURN,
    DuplexEpisodeEvent,
    DuplexTrajectory,
)

__all__ = [
    "ACTION_FORCED",
    "ACTION_REEMIT",
    "ACTION_SAMPLED",
    "AXIS_INTERRUPTION",
    "AXIS_PAUSE",
    "AXIS_TURN",
    "DUPLEX_REPLAY_KEY",
    "CalibrationReport",
    "DuplexActionRecordSet",
    "DuplexEpisodeEvent",
    "DuplexReplay",
    "DuplexRefAudio",
    "DuplexTrajectory",
    "ParityReport",
    "RecordSetError",
    "audio_tokens_for_mel_frames",
    "audit_calibration",
    "build_replay",
    "compare_action_logprobs",
    "engine_repetition_history",
    "recompute_action_logprobs",
    "resolve_default_ref_audio",
    "resolve_duplex_vocabulary",
    "unit_inputs_from_trajectory",
]
