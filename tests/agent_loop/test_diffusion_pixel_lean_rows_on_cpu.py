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
"""CPU tests for the opt-in pixel-lean train rows of the diffusion TQ worker.

Covers ``_write_trajectories_to_tq`` under ``VERL_OMNI_TQ_STRIP_TRAIN_MEDIA``:
decoded visual media (``responses`` + the decoded video artifact) leaves train
rows only when no dump / colocated-reward consumer is configured, and never
leaves validation or audio rows. The TransferQueue put is mocked, so the tests
run without a TransferQueue deployment.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

from verl_omni.agent_loop.diffusion_agent_loop import _InternalDiffusionAgentLoopOutput
from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ
from verl_omni.pipelines.rollout_media import (
    ARTIFACT_PREFIX,
    ARTIFACT_SPECS,
    PRIMARY_ARTIFACT,
    PREVIEW_ARTIFACT,
)

_WorkerBase = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class


def _config(rollout_data_dir=None, reward_enable=False):
    return OmegaConf.create(
        {
            "trainer": {"rollout_data_dir": rollout_data_dir},
            "reward": {"reward_model": {"enable": reward_enable}},
        }
    )


def _make_worker(config) -> _WorkerBase:
    worker = _WorkerBase.__new__(_WorkerBase)
    worker.config = config
    return worker


def _internal_output() -> _InternalDiffusionAgentLoopOutput:
    extra_fields = {
        "media_kind": "video",
        "all_latents": torch.zeros(1, 3, 4, 5, 6),
        "prompt_embeds": torch.zeros(1, 8),
        ARTIFACT_PREFIX + "video_latent": torch.zeros(1, 4),
        ARTIFACT_PREFIX + "video_preview": torch.zeros(1, 9, 3, 4, 4, dtype=torch.uint8),
        ARTIFACT_SPECS: {
            "video_latent": {"modality": "video", "representation": "latent", "layout": "LWC"},
            "video_preview": {"modality": "video", "representation": "decoded", "layout": "TCHW", "fps": 16.0},
        },
        PRIMARY_ARTIFACT: "video_preview",
        PREVIEW_ARTIFACT: "video_preview",
    }
    return _InternalDiffusionAgentLoopOutput(
        prompt_ids=torch.ones(1, 6, dtype=torch.long),
        response_diffusion_output=torch.full((1, 9, 3, 4, 4), 7, dtype=torch.uint8),
        response_logprobs=None,
        reward_score=0.5,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


def _run_write(worker, validate: bool = False, media_kind: str | None = "video"):
    captured: dict = {}

    async def _fake_put(keys, fields, tags, partition_id):
        captured["keys"] = list(keys)
        captured["fields"] = fields
        captured["tags"] = list(tags)
        captured["partition_id"] = partition_id

    import verl_omni.agent_loop.diffusion_agent_loop_tq as dalt

    dalt.tq.async_kv_batch_put = _fake_put
    output = _internal_output()
    if media_kind is None:
        output.extra_fields.pop("media_kind")
    else:
        output.extra_fields["media_kind"] = media_kind
    kwargs = {"index": 0, "reward_model": {"style": "rule"}, "data_source": "x"}
    asyncio.run(
        worker._write_trajectories_to_tq(
            [(0, output)],
            uid="uid-1",
            trajectory={"step": 1, "validate": validate},
            validate=validate,
            global_steps=1,
            **kwargs,
        )
    )
    return captured


def _row_field_names(captured):
    assert captured["fields"] is not None
    return set(captured["fields"].keys())


def test_gate_off_keeps_upstream_row_fields(monkeypatch):
    monkeypatch.delenv("VERL_OMNI_TQ_STRIP_TRAIN_MEDIA", raising=False)
    captured = _run_write(_make_worker(_config()))
    names = _row_field_names(captured)
    assert "responses" in names
    assert ARTIFACT_PREFIX + "video_preview" in names


def test_gate_on_drops_decoded_media_from_train_rows(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_TRAIN_MEDIA", "1")
    captured = _run_write(_make_worker(_config()))
    names = _row_field_names(captured)
    assert "responses" not in names
    assert ARTIFACT_PREFIX + "video_preview" not in names
    # Training-side fields stay.
    assert "all_latents" in names
    assert "prompt_embeds" in names
    assert ARTIFACT_PREFIX + "video_latent" in names
    assert captured["tags"][0]["response_shape"] == (9, 3, 4, 4)  # telemetry kept


def test_gate_on_keeps_validation_rows(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_TRAIN_MEDIA", "1")
    captured = _run_write(_make_worker(_config()), validate=True)
    names = _row_field_names(captured)
    assert "responses" in names
    assert ARTIFACT_PREFIX + "video_preview" in names
    assert captured["partition_id"] == "val"


def test_gate_on_keeps_rows_when_rollout_dump_configured(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_TRAIN_MEDIA", "1")
    captured = _run_write(_make_worker(_config(rollout_data_dir="/tmp/dump")))
    assert "responses" in _row_field_names(captured)


def test_gate_on_keeps_rows_when_colocated_reward_model_enabled(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_TRAIN_MEDIA", "1")
    captured = _run_write(_make_worker(_config(reward_enable=True)))
    assert "responses" in _row_field_names(captured)


def test_gate_on_keeps_audio_responses(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_TRAIN_MEDIA", "1")
    captured = _run_write(_make_worker(_config()), media_kind="audio")
    assert "responses" in _row_field_names(captured)
