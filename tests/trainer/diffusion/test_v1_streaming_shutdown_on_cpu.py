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
"""V1 owner drains agent work before model and transport teardown on failure."""

from types import SimpleNamespace

import pytest
import transfer_queue as tq
from omegaconf import OmegaConf

from verl_omni.trainer.main_diffusion_v1 import DiffusionTaskRunnerV1


@pytest.mark.parametrize("failure", ["init", "fit", "close"])
def test_runner_failure_orders_streaming_shutdown(monkeypatch, failure):
    events = []

    async def close_native_models():
        events.append("close_models")

    def init():
        if failure == "init":
            raise ValueError("trainer init failed")

    def fit(manager):
        if failure == "fit":
            raise ValueError("trainer fit failed")

    trainer = SimpleNamespace(
        init=init,
        fit=fit,
        reward_loop_manager=SimpleNamespace(
            multi_reward_model_manager=SimpleNamespace(close_native_models=close_native_models)
        ),
    )
    runner = DiffusionTaskRunnerV1.__ray_metadata__.modified_class()

    def init_agents():
        runner.agent_loop_manager = SimpleNamespace(
            agent_loop_workers=[SimpleNamespace(close_reward_streaming=SimpleNamespace(remote=lambda: "drain_ref"))]
        )

    def get(refs):
        assert refs == ["drain_ref"]
        if failure == "close":
            raise ValueError("trainer close failed")
        events.append("agents_drained")

    def wait(refs, *, num_returns):
        assert refs == ["drain_ref"] and num_returns == 1
        events.append("agents_settled")
        return refs, []

    monkeypatch.setattr("verl_omni.trainer.diffusion.v1.get_diffusion_trainer_cls", lambda mode: lambda config: trainer)
    monkeypatch.setattr(runner, "init_agent_loop_manager", init_agents)
    monkeypatch.setattr("verl_omni.trainer.main_diffusion_v1.ray.get", get)
    monkeypatch.setattr("verl_omni.trainer.main_diffusion_v1.ray.wait", wait)
    monkeypatch.setattr(tq, "init", lambda config: events.append("init_tq"))
    monkeypatch.setattr(tq, "close", lambda: events.append("close_tq"))
    config = OmegaConf.create(
        {
            "transfer_queue": {"enable": True},
            "trainer": {"v1": {"trainer_mode": "sync"}},
            "reward": {
                "streaming": {"_target_": "verl_omni.workers.config.reward.StreamingRewardConfig", "enabled": True}
            },
        }
    )
    with pytest.raises(ValueError, match=f"trainer {failure} failed"):
        runner.run(config)
    assert (
        events
        == {
            "init": ["init_tq", "close_models", "close_tq"],
            "fit": ["init_tq", "agents_settled", "agents_drained", "close_models", "close_tq"],
            "close": ["init_tq", "agents_settled"],
        }[failure]
    )
