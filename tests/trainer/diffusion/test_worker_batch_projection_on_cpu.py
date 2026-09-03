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
"""CPU contracts for excluding driver-only pixels from diffusion worker RPCs."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    DirectPreferenceRayTrainer,
    PolicyGradientRayTrainer,
    _to_diffusion_worker_tensordict,
)
from verl_omni.trainer.diffusion.teacher_manager import DiffusionTeacherManager
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1


def _make_config():
    return OmegaConf.create(
        {
            "algorithm": {"paired_preference": False},
            "actor_rollout_ref": {
                "model": {
                    "pipeline": {"height": 64, "width": 64},
                    "vae_scale_factor": 8,
                },
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": False,
                },
                "rollout": {"n": 1, "multi_turn": {"enable": False}},
            },
        }
    )


def _make_batch(*, include_responses: bool = True):
    latents = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    tensors = {"latents": latents}
    if include_responses:
        tensors["responses"] = torch.randint(256, (2, 3, 4, 4), dtype=torch.uint8)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={"sample_id": np.array(["sample-0", "sample-1"], dtype=object)},
        meta_info={"driver_tag": "keep-me"},
    )


def _make_infer_output():
    shape = (2, 1)
    return tu.get_tensordict(
        {
            "log_probs": torch.zeros(shape),
            "prev_sample_mean": torch.zeros(shape),
            "noise_pred": torch.zeros(shape),
            "noise": torch.zeros(shape),
            "timesteps": torch.zeros(shape),
        },
        non_tensor_dict={"metrics": {}},
    )


def _make_worker_groups():
    actor = MagicMock()
    ref = MagicMock()
    actor.infer_actor_batch.return_value = _make_infer_output()
    actor.infer_teacher_batch.return_value = _make_infer_output()
    actor.update_actor.return_value = tu.get_tensordict({}, non_tensor_dict={"metrics": {}})
    ref.infer_ref_batch.return_value = _make_infer_output()
    return actor, ref


def _assert_worker_projection(sent_batch, driver_batch):
    assert "responses" not in sent_batch
    assert "responses" in driver_batch.batch
    assert sent_batch["latents"].data_ptr() == driver_batch.batch["latents"].data_ptr()


@pytest.mark.parametrize("include_responses", [True, False])
def test_worker_projection_preserves_driver_batch_and_shared_tensor_storage(include_responses):
    driver_batch = _make_batch(include_responses=include_responses)
    responses = driver_batch.batch.get("responses")

    worker_batch = _to_diffusion_worker_tensordict(driver_batch)

    assert "responses" not in worker_batch
    assert ("responses" in driver_batch.batch) is include_responses
    assert driver_batch.batch.get("responses") is responses
    assert worker_batch["latents"].data_ptr() == driver_batch.batch["latents"].data_ptr()
    assert tu.get(worker_batch, "sample_id") == ["sample-0", "sample-1"]
    assert tu.get(worker_batch, "driver_tag") == "keep-me"


@pytest.mark.parametrize(
    ("trainer_cls", "method_name", "worker_attr", "remote_method"),
    [
        (PolicyGradientRayTrainer, "_compute_old_log_prob", "actor_rollout_wg", "infer_actor_batch"),
        (PolicyGradientRayTrainer, "_compute_ref_log_prob", "ref_policy_wg", "infer_ref_batch"),
        (PolicyGradientRayTrainer, "_update_actor", "actor_rollout_wg", "update_actor"),
        (DirectPreferenceRayTrainer, "_compute_ref_noise_pred", "ref_policy_wg", "infer_ref_batch"),
        (DirectPreferenceRayTrainer, "_update_actor", "actor_rollout_wg", "update_actor"),
    ],
)
def test_v0_diffusion_worker_hops_exclude_responses(trainer_cls, method_name, worker_attr, remote_method):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = _make_config()
    trainer.ref_in_actor = False
    trainer.actor_rollout_wg, trainer.ref_policy_wg = _make_worker_groups()
    driver_batch = _make_batch()

    getattr(trainer, method_name)(driver_batch)

    worker = getattr(trainer, worker_attr)
    sent_batch = getattr(worker, remote_method).call_args.args[0]
    _assert_worker_projection(sent_batch, driver_batch)


def test_teacher_manager_hop_excludes_responses():
    actor, _ = _make_worker_groups()
    manager = DiffusionTeacherManager.__new__(DiffusionTeacherManager)
    manager.model_config = _make_config().actor_rollout_ref.model
    manager.teacher_wg = {"default": actor}
    driver_batch = _make_batch()

    manager._infer(driver_batch, "default")

    sent_batch = actor.infer_teacher_batch.call_args.args[0]
    _assert_worker_projection(sent_batch, driver_batch)


@pytest.mark.parametrize(
    ("method_name", "worker_attr", "remote_method"),
    [
        ("_compute_old_log_prob", "actor_rollout_wg", "infer_actor_batch"),
        ("_compute_ref_log_prob", "ref_policy_wg", "infer_ref_batch"),
        ("_update_actor", "actor_rollout_wg", "update_actor"),
    ],
)
def test_v1_diffusion_worker_hops_exclude_responses(method_name, worker_attr, remote_method):
    actor, ref = _make_worker_groups()
    trainer = SimpleNamespace(
        config=_make_config(),
        ref_in_actor=False,
        actor_rollout_wg=actor,
        ref_policy_wg=ref,
    )
    driver_batch = _make_batch()

    method = getattr(PolicyGradientDiffusionTrainerV1, method_name)
    method(trainer, driver_batch)

    worker = getattr(trainer, worker_attr)
    sent_batch = getattr(worker, remote_method).call_args.args[0]
    _assert_worker_projection(sent_batch, driver_batch)
