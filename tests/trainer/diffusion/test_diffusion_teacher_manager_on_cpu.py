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
"""CPU tests for routing rollout batches to frozen diffusion teachers."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.trainer.diffusion.teacher_manager import DiffusionTeacherManager

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")

SINGLE = [
    "distillation.enabled=true",
    "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
]
MULTI = [
    "distillation.enabled=true",
    "+distillation.teacher_models.a.key=ocr",
    "+distillation.teacher_models.a.model_path=/ckpt/a",
    "+distillation.teacher_models.b.key=aes",
    "+distillation.teacher_models.b.model_path=/ckpt/b",
]


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


class FakeTeacherWorkerGroup:
    """Records dispatch order and returns a key-specific constant as a future."""

    def __init__(self, value, world_size, log):
        self.value = value
        self.world_size = world_size
        self.log = log
        self.calls = []

    def infer_teacher_batch(self, td):
        teacher_key = tu.get(td, "teacher_key")
        self.calls.append((teacher_key, td.batch_size[0]))
        self.log.append(("infer", teacher_key))
        prev = torch.full((td.batch_size[0], 4, 8, 3), self.value, dtype=torch.bfloat16)

        def get():
            self.log.append(("get", teacher_key))
            return tu.get_tensordict({"prev_sample_mean": prev})

        return SimpleNamespace(get=get)


def make_batch(data_source):
    n = len(data_source)
    batch = DataProto.from_tensordict(
        tu.get_tensordict({"all_latents": torch.randn(n, 5, 8, 3), "all_timesteps": torch.zeros(n, 4)})
    )
    batch.non_tensor_batch["data_source"] = np.array(data_source, dtype=object)
    return batch


def make_manager(overrides, teacher_wg, **kwargs):
    cfg = compose_cfg(overrides)
    return DiffusionTeacherManager(
        omega_conf_to_dataclass(cfg.distillation), cfg.actor_rollout_ref.model, teacher_wg, **kwargs
    )


class TestDiffusionTeacherManagerInit:
    def test_rejects_mismatched_worker_group_keys(self):
        with pytest.raises(ValueError, match="do not match teacher routing keys"):
            make_manager(MULTI, {"ocr": SimpleNamespace(), "other": SimpleNamespace()})


class TestResolveTeacherKeys:
    def test_single_teacher_ignores_routing_column(self):
        manager = make_manager(SINGLE, {"default": SimpleNamespace()})
        keys = manager._resolve_teacher_keys(make_batch(["x", "y"]))
        assert list(keys) == ["default", "default"]

    def test_multi_teacher_requires_routing_column(self):
        manager = make_manager(MULTI, {"ocr": SimpleNamespace(), "aes": SimpleNamespace()})
        batch = make_batch(["ocr"])
        del batch.non_tensor_batch["data_source"]
        with pytest.raises(ValueError, match="Routing key is required for multi-teacher distillation"):
            manager._resolve_teacher_keys(batch)

    def test_multi_teacher_rejects_unknown_key(self):
        manager = make_manager(MULTI, {"ocr": SimpleNamespace(), "aes": SimpleNamespace()})
        with pytest.raises(ValueError, match="No teacher configured for routing key"):
            manager._resolve_teacher_keys(make_batch(["ocr", "bogus"]))

    def test_multi_teacher_returns_column(self):
        manager = make_manager(MULTI, {"ocr": SimpleNamespace(), "aes": SimpleNamespace()})
        keys = manager._resolve_teacher_keys(make_batch(["aes", "ocr"]))
        assert list(keys) == ["aes", "ocr"]


class TestComputePrevSampleMean:
    def test_multi_teacher_routes_groups_and_restores_order(self):
        log = []
        wg = {"ocr": FakeTeacherWorkerGroup(1.0, 2, log), "aes": FakeTeacherWorkerGroup(2.0, 1, log)}
        manager = make_manager(MULTI, wg)
        data_source = ["ocr", "aes", "ocr", "aes", "ocr"]

        out = manager.compute_prev_sample_mean(make_batch(data_source))

        assert set(out.batch.keys()) == {"teacher_prev_sample_mean"}
        prev = out.batch["teacher_prev_sample_mean"]
        assert prev.dtype == torch.float32
        assert prev.shape == (5, 4, 8, 3)
        expected = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0]).view(5, 1, 1, 1).expand(5, 4, 8, 3)
        torch.testing.assert_close(prev, expected)
        assert wg["ocr"].calls == [("ocr", 4)]
        assert wg["aes"].calls == [("aes", 2)]
        assert [kind for kind, _ in log] == ["infer", "infer", "get", "get"]

    def test_odd_sub_batch_padded_to_micro_batch_divisor(self):
        log = []
        wg = {"ocr": FakeTeacherWorkerGroup(1.0, 2, log), "aes": FakeTeacherWorkerGroup(2.0, 2, log)}
        manager = make_manager(MULTI, wg, infer_micro_batch_size_per_gpu=8)
        data_source = ["ocr"] * 3 + ["aes"] * 5

        out = manager.compute_prev_sample_mean(make_batch(data_source))

        # 3 and 5 rows are both padded to world_size * micro so every rank's shard splits into micro-batches
        assert wg["ocr"].calls == [("ocr", 16)]
        assert wg["aes"].calls == [("aes", 16)]
        prev = out.batch["teacher_prev_sample_mean"]
        assert prev.shape == (8, 4, 8, 3)
        expected = torch.tensor([1.0] * 3 + [2.0] * 5).view(8, 1, 1, 1).expand(8, 4, 8, 3)
        torch.testing.assert_close(prev, expected)

    def test_without_micro_batch_size_pads_to_world_size_only(self):
        log = []
        wg = {"ocr": FakeTeacherWorkerGroup(1.0, 2, log), "aes": FakeTeacherWorkerGroup(2.0, 1, log)}
        manager = make_manager(MULTI, wg)

        manager.compute_prev_sample_mean(make_batch(["ocr", "ocr", "ocr", "aes"]))

        assert wg["ocr"].calls == [("ocr", 4)]
        assert wg["aes"].calls == [("aes", 1)]

    def test_single_teacher_dispatches_whole_batch_once(self):
        log = []
        wg = {"default": FakeTeacherWorkerGroup(3.0, 2, log)}
        manager = make_manager(SINGLE, wg)

        out = manager.compute_prev_sample_mean(make_batch(["x", "y", "z"]))

        assert wg["default"].calls == [("default", 3)]
        torch.testing.assert_close(out.batch["teacher_prev_sample_mean"], torch.full((3, 4, 8, 3), 3.0))
