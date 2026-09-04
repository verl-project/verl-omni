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
"""CPU tests for best-effort media dumping in the diffusion V1 trainer."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

import verl_omni.trainer.diffusion.ray_diffusion_trainer as ray_diffusion_trainer
import verl_omni.trainer.diffusion.v1.trainer_base as trainer_base_module
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1


class _ConcreteTrainer(PolicyGradientDiffusionTrainerV1):
    def on_step_end(self):
        return None

    def on_sample_end(self):
        return None


class _FakeData:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch

    def __len__(self):
        return len(self.batch["responses"])


def _trainer(global_steps: int = 1) -> _ConcreteTrainer:
    trainer = object.__new__(_ConcreteTrainer)
    trainer.global_steps = global_steps
    trainer._init_dump_executor()
    return trainer


def test_v1_video_dump_reuses_shared_export_and_honors_max_samples(monkeypatch, tmp_path):
    exported = []

    def fake_export(output, output_path, **kwargs):
        exported.append((output.clone(), output_path, kwargs))
        Path(output_path).write_bytes(b"video")

    monkeypatch.setattr(ray_diffusion_trainer, "_export_video", fake_export)
    trainer = _trainer(global_steps=7)
    outputs = torch.randint(256, (2, 4, 3, 8, 8), dtype=torch.uint8)

    trainer._dump_generations(
        inputs=["first", "second"],
        outputs=outputs,
        gts=[None, None],
        scores=[1.0, 2.0],
        reward_extra_infos_dict={},
        dump_path=str(tmp_path),
        max_samples=1,
        fps=12,
        media_kind="video",
    )
    trainer._shutdown_dump_executor()

    assert len(exported) == 1
    torch.testing.assert_close(exported[0][0], outputs[0])
    assert exported[0][1].endswith("7/0.mp4")
    assert exported[0][2]["fps"] == 12
    rows = [json.loads(line) for line in (tmp_path / "7.jsonl").read_text().splitlines()]
    assert rows == [{"input": "first", "output": str(tmp_path / "7/0.mp4"), "gts": None, "score": 1.0, "step": 7}]


def test_v1_background_dump_failure_is_logged_and_does_not_raise(caplog, tmp_path):
    trainer = _trainer(global_steps=3)

    with caplog.at_level(logging.WARNING):
        trainer._dump_generations(
            inputs=["prompt"],
            outputs=torch.zeros(1, 3, 8, 8),
            gts=[None],
            scores=[0.0],
            reward_extra_infos_dict={},
            dump_path=str(tmp_path),
            media_kind="image",
        )
        trainer._shutdown_dump_executor()

    assert "Ignoring background media dump failure at step 3" in caplog.text


def test_v1_rollout_dump_sorts_and_forwards_media_metadata(monkeypatch):
    trainer = object.__new__(_ConcreteTrainer)
    trainer.config = OmegaConf.create({"trainer": {"rollout_data_max_samples": 1, "video_fps": 12}})
    trainer.tokenizer = SimpleNamespace(
        pad_token_id=0,
        batch_decode=lambda prompts, skip_special_tokens: ["second", "first"],
    )
    captured = {}
    trainer._dump_generations = lambda **kwargs: captured.update(kwargs)

    audio = torch.stack((torch.full((1, 4), 2.0), torch.full((1, 4), 1.0)))
    data = _FakeData(
        batch={
            "prompts": torch.tensor([[2], [1]]),
            "responses": torch.stack(
                (
                    torch.full((2, 3, 4, 4), 2, dtype=torch.uint8),
                    torch.full((2, 3, 4, 4), 1, dtype=torch.uint8),
                )
            ),
            "sample_level_scores": torch.tensor([[2.0], [1.0]]),
            "audio": audio,
        },
        non_tensor_batch={
            "media_kind": np.array(["video", "video"], dtype=object),
            "audio_sample_rate": np.array([48_000, 48_000], dtype=object),
        },
    )
    monkeypatch.setattr(trainer_base_module, "diffusion_tq_batch_to_dataproto", lambda *args, **kwargs: data)

    batch_meta = SimpleNamespace(keys=["second_0_0", "first_0_0"], partition_id="train")
    trainer._log_rollout_data(batch_meta, {}, "/tmp/unused")

    assert captured["inputs"] == ["first", "second"]
    assert captured["scores"] == [1.0, 2.0]
    torch.testing.assert_close(captured["outputs"][0], data.batch["responses"][1])
    torch.testing.assert_close(captured["audios"][0], audio[1])
    assert captured["audio_sample_rates"] == [48_000, 48_000]
    assert captured["media_kind"] == "video"
    assert captured["max_samples"] == 1
    assert captured["fps"] == 12
