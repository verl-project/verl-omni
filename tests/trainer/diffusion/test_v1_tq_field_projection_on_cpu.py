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

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.trainer.diffusion.v1 import tq_utils
from verl_omni.trainer.diffusion.v1 import trainer_base as trainer_base_module


def test_tq_conversion_forwards_field_projection_and_unpacks_extra_fields(monkeypatch):
    captured = {}

    def get_fields(**kwargs):
        captured.update(kwargs)
        return {
            "sample_level_rewards": torch.ones(2, 3),
            "uid": ["first", "second"],
            "extra_fields": [
                {"reward_extra_info": {"ocr": 0.25}},
                {"reward_extra_info": {"ocr": 0.75}},
            ],
        }

    monkeypatch.setattr(tq_utils.tq, "kv_batch_get", get_fields)
    selected = ["sample_level_rewards", "uid", "extra_fields"]

    data = tq_utils.diffusion_tq_batch_to_dataproto(
        SimpleNamespace(keys=["first_0_0", "second_0_0"], partition_id="train"),
        select_fields=selected,
    )

    assert captured == {
        "keys": ["first_0_0", "second_0_0"],
        "partition_id": "train",
        "select_fields": selected,
    }
    assert set(data.batch.keys()) == {"sample_level_rewards"}
    assert data.non_tensor_batch["uid"].tolist() == ["first", "second"]
    assert data.non_tensor_batch["reward_extra_info"].tolist() == [{"ocr": 0.25}, {"ocr": 0.75}]


@pytest.mark.parametrize("algorithm", ["policy_gradient", "direct_preference"])
def test_metric_projection_is_subset_of_persisted_and_rollout_fields(algorithm):
    rollout_fields = {"uid", "extra_fields"}
    available_fields = set(tq_utils.diffusion_persisted_tq_fields(algorithm)) | rollout_fields

    assert set(tq_utils.diffusion_metric_tq_fields(algorithm)).issubset(available_fields)


@pytest.mark.parametrize("policy_gradient", [True, False])
def test_metrics_fetches_only_metric_fields_and_preserves_outputs(monkeypatch, caplog, policy_gradient):
    tensors = {
        "sample_level_rewards": torch.tensor([[1.0, 1.0], [3.0, 3.0]]),
        "sample_level_scores": torch.tensor([[1.0], [3.0]]),
    }
    if policy_gradient:
        tensors.update(
            {
                "advantages": torch.tensor([[1.0, -1.0], [2.0, -2.0]]),
                "returns": torch.tensor([[0.5, -0.5], [1.5, -1.5]]),
            }
        )
    reward_extra_info = np.empty(2, dtype=object)
    reward_extra_info[:] = [{"ocr": 0.25}, {"ocr": 0.75}]
    data = DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "uid": np.array(["prompt", "prompt"], dtype=object),
            "reward_extra_info": reward_extra_info,
        },
    )
    captured = {}

    def get_data(batch_meta, pad_token_id, select_fields):
        captured.update(batch_meta=batch_meta, pad_token_id=pad_token_id, select_fields=select_fields)
        return data

    monkeypatch.setattr(trainer_base_module, "diffusion_tq_batch_to_dataproto", get_data)
    trainer = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=7),
        _get_n_gpus_for_throughput=lambda: 2,
        _is_direct_preference=not policy_gradient,
    )
    batch_meta = SimpleNamespace(
        keys=["prompt_0_0", "prompt_1_0"],
        tags=[
            {
                "response_shape": (np.int64(3), np.int64(256), np.int64(384)),
                "is_padding": False,
                "min_global_steps": 1,
                "max_global_steps": 1,
            },
            {"response_shape": (3, 384, 256), "is_padding": False, "min_global_steps": 1, "max_global_steps": 1},
        ],
        partition_id="train",
    )
    metrics = {"actor/grad_norm": 2.0}

    with caplog.at_level(logging.INFO, logger=trainer_base_module.logger.name):
        trainer_base_module.PolicyGradientDiffusionTrainerV1._compute_metrics(
            trainer,
            batch_meta,
            metrics,
            {"step": 2.0, "gen": 1.0},
            global_steps=2,
            epoch=0,
        )

    selected = set(captured["select_fields"])
    expected_fields = {
        "sample_level_rewards",
        "sample_level_scores",
        "uid",
        "extra_fields",
    }
    if policy_gradient:
        expected_fields.update({"advantages", "returns"})
    assert selected == expected_fields
    assert captured["pad_token_id"] == 7
    assert metrics["critic/rewards/mean"] == pytest.approx(2.0)
    assert metrics["critic/rewards/group_size"] == pytest.approx(2.0)
    assert metrics["critic/ocr/mean"] == pytest.approx(0.5)
    assert metrics["perf/total_num_images"] == 2
    assert metrics["perf/throughput"] == pytest.approx(0.5)
    assert metrics["training/tq_response_shape_unavailable"] == 0.0
    assert "variance_proxy/proxy2_total_power" not in metrics
    assert "2 trajectories, 2 real images, responses shape=(2, 3, 384, 384)" in caplog.text


@pytest.mark.parametrize(
    "response_shapes",
    [
        pytest.param([None], id="missing"),
        pytest.param(["3x256x256"], id="scalar"),
        pytest.param([(3, True, 256)], id="bool-dimension"),
        pytest.param([(3, -1, 256)], id="negative-dimension"),
        pytest.param([(3, 0, 256)], id="zero-dimension"),
        pytest.param([(3, 256, 256), (3, 256)], id="mixed-rank"),
    ],
)
def test_metrics_keeps_training_when_response_shape_telemetry_is_unavailable(monkeypatch, caplog, response_shapes):
    batch_size = len(response_shapes)
    data = DataProto.from_dict(
        tensors={
            "sample_level_rewards": torch.ones(batch_size, 1),
            "sample_level_scores": torch.ones(batch_size, 1),
        }
    )
    captured = {}

    def get_data(batch_meta, pad_token_id, select_fields):
        captured["select_fields"] = select_fields
        return data

    monkeypatch.setattr(trainer_base_module, "diffusion_tq_batch_to_dataproto", get_data)
    trainer = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        _get_n_gpus_for_throughput=lambda: 1,
        _is_direct_preference=True,
    )
    tags = []
    for response_shape in response_shapes:
        tag = {"is_padding": False, "min_global_steps": 0, "max_global_steps": 0}
        if response_shape is not None:
            tag["response_shape"] = response_shape
        tags.append(tag)
    batch_meta = SimpleNamespace(
        keys=[f"sample_{index}_0" for index in range(batch_size)],
        tags=tags,
        partition_id="train",
    )
    metrics = {}

    with caplog.at_level(logging.INFO, logger=trainer_base_module.logger.name):
        trainer_base_module.PolicyGradientDiffusionTrainerV1._compute_metrics(
            trainer,
            batch_meta,
            metrics,
            {"step": 1.0},
            global_steps=1,
            epoch=0,
        )

    assert metrics["training/tq_response_shape_unavailable"] == 1.0
    assert metrics["perf/total_num_images"] == batch_size
    assert "responses" not in captured["select_fields"]
    assert "response_shape telemetry is unavailable" in caplog.text
    assert f"{batch_size} trajectories, unknown real images, responses shape=None" in caplog.text


def test_metrics_ignores_padding_without_response_shape(monkeypatch, caplog):
    data = DataProto.from_dict(
        tensors={
            "sample_level_rewards": torch.ones(2, 1),
            "sample_level_scores": torch.ones(2, 1),
        }
    )

    monkeypatch.setattr(trainer_base_module, "diffusion_tq_batch_to_dataproto", lambda *args, **kwargs: data)
    trainer = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        _get_n_gpus_for_throughput=lambda: 1,
        _is_direct_preference=True,
    )
    batch_meta = SimpleNamespace(
        keys=["sample_0_0", "padding_0_0"],
        tags=[
            {
                "response_shape": (3, 256, 256),
                "is_padding": False,
                "min_global_steps": 0,
                "max_global_steps": 0,
            },
            {"is_padding": True, "min_global_steps": 0, "max_global_steps": 0},
        ],
        partition_id="train",
    )
    metrics = {}

    with caplog.at_level(logging.INFO, logger=trainer_base_module.logger.name):
        trainer_base_module.PolicyGradientDiffusionTrainerV1._compute_metrics(
            trainer,
            batch_meta,
            metrics,
            {"step": 1.0},
            global_steps=1,
            epoch=0,
        )

    assert metrics["training/tq_response_shape_unavailable"] == 0.0
    assert "response_shape telemetry is unavailable" not in caplog.text
    assert "2 trajectories, 1 real images, responses shape=(1, 3, 256, 256)" in caplog.text
