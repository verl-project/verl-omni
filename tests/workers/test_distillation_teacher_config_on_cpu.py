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
"""CPU tests for the frozen diffusion teacher's derived training config."""

import os

import pytest
from hydra import compose, initialize_config_dir

import verl_omni
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.config.diffusion import DiffusionDistillationTeacherModelConfig
from verl_omni.workers.engine_workers import build_teacher_training_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


@pytest.fixture
def actor_rollout_ref_config():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8"],
        )
    return cfg.actor_rollout_ref


def test_teacher_training_config_derivation(actor_rollout_ref_config, tmp_path):
    student_dir, teacher_dir = tmp_path / "student", tmp_path / "teacher"
    student_dir.mkdir()
    teacher_dir.mkdir()
    model_config = DiffusionModelConfig(
        path=str(student_dir),
        architecture="StableDiffusion3Pipeline",
        algorithm="flow_grpo",
        attn_backend="native",
        load_tokenizer=False,
        transformer_config={},
        lora_rank=32,
    )
    teacher_training_config = build_teacher_training_config(
        config=actor_rollout_ref_config,
        model_config=model_config,
        teacher_model_config=DiffusionDistillationTeacherModelConfig(model_path=str(teacher_dir)),
    )
    assert teacher_training_config.model_type == "diffusion_model"
    assert teacher_training_config.model_config.path == str(teacher_dir)
    assert teacher_training_config.model_config.lora_rank == 0
    assert teacher_training_config.model_config.lora_adapter_path is None
    assert teacher_training_config.model_config.load_tokenizer is False
    assert teacher_training_config.engine_config.forward_only is True
    # ref gives no scoring micro-batch by default, so the actor training micro-batch applies
    assert teacher_training_config.engine_config.infer_micro_batch_size_per_gpu == 8


def test_infer_teacher_batch_routes_by_teacher_key():
    from types import SimpleNamespace

    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    from verl_omni.workers.engine_workers import ActorRolloutRefWorker

    class FakeTeacher:
        def __init__(self, value):
            self.value = value
            self.seen = None

        def infer_batch(self, data):
            self.seen = data
            return TensorDict(
                {"prev_sample_mean": torch.full((data.batch_size[0], 1), self.value)}, batch_size=data.batch_size
            )

    teachers = {"ocr": FakeTeacher(1.0), "aes": FakeTeacher(2.0)}
    worker = SimpleNamespace(teachers=teachers, enable_routing_replay=False, profiler=None)
    data = TensorDict({"x": torch.zeros(3, 1)}, batch_size=[3])
    data = tu.assign_non_tensor(data, teacher_key="aes")

    output = ActorRolloutRefWorker.infer_teacher_batch.__wrapped__(worker, data)

    assert torch.equal(output["prev_sample_mean"], torch.full((3, 1), 2.0))
    assert teachers["ocr"].seen is None
    assert "teacher_key" not in teachers["aes"].seen.keys()
