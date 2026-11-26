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

Covers ``_write_trajectories_to_tq`` under ``trainer.strip_train_media``:
decoded visual media (``responses`` + the decoded artifact copy) leaves train
rows only when no dump / reward-model consumer is configured, and never leaves
validation or audio rows. The TransferQueue put is mocked, so the tests run
without a TransferQueue deployment.
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
    PREVIEW_ARTIFACT,
    PRIMARY_ARTIFACT,
)

_WorkerBase = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class


def _config(strip_train_media=False, rollout_data_dir=None, reward_enable=False, reward_models=None):
    reward: dict = {"reward_model": {"enable": reward_enable}}
    if reward_models is not None:
        reward["models"] = reward_models
    return OmegaConf.create(
        {
            "trainer": {"rollout_data_dir": rollout_data_dir, "strip_train_media": strip_train_media},
            "reward": reward,
        }
    )


def _make_worker(config) -> _WorkerBase:
    worker = _WorkerBase.__new__(_WorkerBase)
    worker.config = config
    return worker


def _internal_output(media_kind: str | None = "video") -> _InternalDiffusionAgentLoopOutput:
    if media_kind == "image":
        specs = {
            "image_latent": {"modality": "image", "representation": "latent", "layout": "LWC"},
            "image_preview": {"modality": "image", "representation": "decoded", "layout": "HWC"},
        }
        latent_name = "image_latent"
        primary = preview = "image_preview"
    else:
        specs = {
            "video_latent": {"modality": "video", "representation": "latent", "layout": "LWC"},
            "video_preview": {"modality": "video", "representation": "decoded", "layout": "TCHW", "fps": 16.0},
        }
        latent_name = "video_latent"
        primary = preview = "video_preview"
    extra_fields = {
        "all_latents": torch.zeros(1, 3, 4, 5, 6),
        "prompt_embeds": torch.zeros(1, 8),
        ARTIFACT_PREFIX + latent_name: torch.zeros(1, 4),
        ARTIFACT_PREFIX + primary: torch.zeros(1, 9, 3, 4, 4, dtype=torch.uint8),
        ARTIFACT_SPECS: specs,
        PRIMARY_ARTIFACT: primary,
        PREVIEW_ARTIFACT: preview,
    }
    if media_kind is not None:
        extra_fields["media_kind"] = media_kind
    return _InternalDiffusionAgentLoopOutput(
        prompt_ids=torch.ones(1, 6, dtype=torch.long),
        response_diffusion_output=torch.full((1, 9, 3, 4, 4), 7, dtype=torch.uint8),
        response_logprobs=None,
        reward_score=0.5,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


def _run_write(worker, monkeypatch, validate: bool = False, media_kind: str | None = "video") -> dict:
    captured: dict = {}

    async def _fake_put(keys, fields, tags, partition_id):
        captured["keys"] = list(keys)
        captured["fields"] = fields
        captured["tags"] = list(tags)
        captured["partition_id"] = partition_id

    import verl_omni.agent_loop.diffusion_agent_loop_tq as dalt

    monkeypatch.setattr(dalt.tq, "async_kv_batch_put", _fake_put)
    output = _internal_output(media_kind)
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


def test_strip_off_keeps_upstream_row_fields(monkeypatch):
    captured = _run_write(_make_worker(_config()), monkeypatch)
    names = _row_field_names(captured)
    assert "responses" in names
    assert ARTIFACT_PREFIX + "video_preview" in names


def test_strip_on_drops_decoded_media_from_train_rows(monkeypatch):
    captured = _run_write(_make_worker(_config(strip_train_media=True)), monkeypatch)
    names = _row_field_names(captured)
    assert "responses" not in names
    assert ARTIFACT_PREFIX + "video_preview" not in names
    # Training-side fields stay.
    assert "all_latents" in names
    assert "prompt_embeds" in names
    assert ARTIFACT_PREFIX + "video_latent" in names
    assert captured["tags"][0]["response_shape"] == (9, 3, 4, 4)  # telemetry kept


def test_strip_on_keeps_validation_rows(monkeypatch):
    captured = _run_write(_make_worker(_config(strip_train_media=True)), monkeypatch, validate=True)
    names = _row_field_names(captured)
    assert "responses" in names
    assert ARTIFACT_PREFIX + "video_preview" in names
    assert captured["partition_id"] == "val"


def test_strip_on_keeps_rows_when_rollout_dump_configured(monkeypatch):
    captured = _run_write(_make_worker(_config(strip_train_media=True, rollout_data_dir="/tmp/dump")), monkeypatch)
    assert "responses" in _row_field_names(captured)


def test_strip_on_keeps_rows_when_colocated_reward_model_enabled(monkeypatch):
    captured = _run_write(_make_worker(_config(strip_train_media=True, reward_enable=True)), monkeypatch)
    assert "responses" in _row_field_names(captured)


def test_strip_on_keeps_rows_when_named_reward_models_configured(monkeypatch):
    models = {"hpsv3": {"backend": "engine"}}
    captured = _run_write(_make_worker(_config(strip_train_media=True, reward_models=models)), monkeypatch)
    assert "responses" in _row_field_names(captured)


def test_strip_on_drops_decoded_media_from_image_rows(monkeypatch):
    captured = _run_write(_make_worker(_config(strip_train_media=True)), monkeypatch, media_kind="image")
    names = _row_field_names(captured)
    assert "responses" not in names
    assert ARTIFACT_PREFIX + "image_preview" not in names
    assert ARTIFACT_PREFIX + "image_latent" in names


def test_strip_on_keeps_audio_rows_intact(monkeypatch):
    captured = _run_write(_make_worker(_config(strip_train_media=True)), monkeypatch, media_kind="audio")
    names = _row_field_names(captured)
    # ``responses`` is the audio waveform payload and the decoded preview is
    # its only visual companion — neither is dropped.
    assert "responses" in names
    assert ARTIFACT_PREFIX + "video_preview" in names


@pytest.mark.parametrize("config_attr", ["trainer", "reward"])
def test_strip_stays_off_when_config_block_missing(config_attr, monkeypatch):
    config = _config(strip_train_media=True)
    del config[config_attr]
    captured = _run_write(_make_worker(config), monkeypatch)
    assert "responses" in _row_field_names(captured)
