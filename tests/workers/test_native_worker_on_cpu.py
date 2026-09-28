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

"""Native worker role and distributed dispatch regression tests."""

from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl.trainer.ppo.utils import Role

from verl_omni.trainer import main_diffusion
from verl_omni.workers.engine_workers import ActorRolloutRefWorker
from verl_omni.workers.native_workers import NativeRolloutWorker


@pytest.mark.parametrize(
    "rollout_name, expected", [("native", NativeRolloutWorker), ("vllm_omni", ActorRolloutRefWorker)]
)
def test_runner_selects_worker_without_changing_group_role(monkeypatch, rollout_name, expected):
    config_dir = Path(main_diffusion.__file__).parent / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "algorithm.trainer_type=policy_gradient"
                if rollout_name == "native"
                else "algorithm.trainer_type=unigrpo",
                f"actor_rollout_ref.rollout.name={rollout_name}",
            ],
        )
    monkeypatch.setattr(main_diffusion.ray, "remote", lambda cls: cls)
    runner = main_diffusion.TaskRunner()
    worker_cls, _ = runner.add_actor_rollout_worker(config)
    assert worker_cls is expected
    assert runner.role_worker_mapping == {Role.ActorRollout: expected}
    assert runner.mapping == {Role.ActorRollout: "global_pool"}


@pytest.mark.parametrize("role", ["actor", "actor_rollout"])
def test_native_initializes_parent_as_actor(monkeypatch, role):
    init = Mock(return_value=None)
    monkeypatch.setattr(ActorRolloutRefWorker, "__init__", init)
    config = object()
    NativeRolloutWorker(config=config, role=role)
    init.assert_called_once_with(config=config, role="actor", distillation_config=None, teacher_key=None)
    assert NativeRolloutWorker.init_model is ActorRolloutRefWorker.init_model


def test_reference_role_is_not_silently_dropped():
    with pytest.raises(ValueError, match="without a reference policy"):
        NativeRolloutWorker(config=None, role="actor_rollout_ref")


@pytest.mark.parametrize("method, engine_method", [("generate", "generate_rollout")])
@pytest.mark.parametrize("has_output", [True, False])
def test_dispatch_preserves_request_and_optional_output(method, engine_method, has_output):
    request = TensorDict({"input": torch.arange(2)}, [2])
    output = TensorDict({"result": torch.ones(2)}, [2]) if has_output else None
    callback = Mock(return_value=output)
    worker = SimpleNamespace(actor=SimpleNamespace(engine=SimpleNamespace(**{engine_method: callback})))
    result = unwrap(getattr(NativeRolloutWorker, method))(worker, request)
    callback.assert_called_once_with(request)
    if has_output:
        torch.testing.assert_close(result["result"], output["result"])
        assert result["result"].device.type == "cpu"
    else:
        assert result is None
