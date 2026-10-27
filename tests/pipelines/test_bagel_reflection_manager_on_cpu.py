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

"""Two-stage manager boundary checks without a Ray cluster or reward server."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl_omni.pipelines.bagel_alphagrpo.reflection_agent import (
    AgentLoopManager,
    BagelReflectionAgentLoopManager,
)


class Batch:
    """Minimal batch fixture for testing the manager's public transport operations."""

    def __init__(self, batch, non_tensor_batch, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(self.non_tensor_batch["uid"])

    def __getitem__(self, rows):
        return Batch(
            {k: v[rows] for k, v in self.batch.items()},
            {k: v[rows] for k, v in self.non_tensor_batch.items()},
            deepcopy(self.meta_info),
        )


def object_rows(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


@pytest.mark.parametrize("improves,validate", [(True, False), (False, False), (True, True)])
def test_manager_selects_globally_and_preserves_training_prompts(monkeypatch, improves, validate):
    manager = object.__new__(BagelReflectionAgentLoopManager)
    manager.rollout_config = SimpleNamespace(n=4, val_kwargs=SimpleNamespace(n=4))
    manager.agent_loop_workers = [object(), object()]
    prompts = Batch(
        {"prompts": torch.tensor([[1, 10, 2]] * 4 + [[1, 11, 2]] * 4)},
        {
            "uid": np.array(["a"] * 4 + ["b"] * 4),
            "prompt_token_ids": object_rows([[1, 10, 2]] * 4 + [[1, 11, 2]] * 4),
            "raw_prompt": object_rows([[{"role": "user", "content": "caption"}]] * 8),
        },
        {"validate": validate},
    )
    calls = []

    async def generate(self, batch):
        calls.append(deepcopy(batch))
        count = len(batch)
        output = Batch(
            dict(batch.batch), dict(batch.non_tensor_batch), {"timing": {"generate": 1}, "reward_extra_keys": []}
        )
        if len(calls) == 1:
            assert count == 4
            output.batch["rm_scores"] = torch.tensor([[0.7], [0.2], [0.6], [0.3]])
            output.batch["responses"] = torch.arange(4, dtype=torch.uint8)[:, None, None, None].expand(4, 3, 2, 2)
        else:
            assert count == 8
            for i, images in enumerate(batch.non_tensor_batch["condition_images"]):
                assert np.asarray(images[0])[0, 0, 0] == (1 if i < 4 else 3)
            output.batch["prompts"] = torch.zeros_like(batch.batch["prompts"])
            scores = [0.1, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.8] if improves else [0.1] * 8
            output.batch["rm_scores"] = torch.tensor(scores)[:, None]
        return output

    monkeypatch.setattr(AgentLoopManager, "generate_sequences", generate)

    async def run():
        return await manager.generate_sequences(prompts)

    result = asyncio.run(run())
    assert len(calls) == 2
    torch.testing.assert_close(result.batch["prompts"], prompts.batch["prompts"])
    assert "condition_images" not in prompts.non_tensor_batch
    assert "condition_images" not in result.non_tensor_batch
    assert result.non_tensor_batch["__num_turns__"].tolist() == [4] * 8
    assert result.meta_info["timing"] == {"initial/generate": 1, "refinement/generate": 1}
    if validate:
        assert "skip_actor_update" not in result.meta_info
        assert result.batch["rm_scores"][0].item() == pytest.approx(0.1)
    else:
        assert result.meta_info["skip_actor_update"] == (not improves)
        assert result.batch["rm_scores"][0].item() == pytest.approx(0.2)


def test_missing_uid_fails_before_generation():
    manager = object.__new__(BagelReflectionAgentLoopManager)
    prompts = Batch({}, {})

    async def run():
        return await manager.generate_sequences(prompts)

    with pytest.raises(ValueError, match="UIDs"):
        asyncio.run(run())


@pytest.mark.parametrize("free_cache_engine", [True, False])
def test_manager_requires_resident_judge(monkeypatch, free_cache_engine):
    from verl_omni.pipelines.bagel_alphagrpo import reflection_agent

    config = SimpleNamespace(
        trainer={"use_v1": False},
        reward=SimpleNamespace(
            reward_model=SimpleNamespace(rollout=SimpleNamespace(free_cache_engine=free_cache_engine)),
            custom_reward_function=SimpleNamespace(name="compute_score_dvreward"),
        ),
    )

    def initialize(self):
        self.config = config
        self.reward_loop_worker_handles = [object()]
        self.model_config = SimpleNamespace(algorithm="alphagrpo")
        self.rollout_config = SimpleNamespace(pipeline={}, n=2)

    monkeypatch.setattr(AgentLoopManager, "__init__", initialize)
    monkeypatch.setattr(reflection_agent.ray, "remote", lambda cls: cls, raising=False)
    if free_cache_engine:
        with pytest.raises(ValueError, match="resident judge"):
            BagelReflectionAgentLoopManager()
    else:
        assert BagelReflectionAgentLoopManager().config is config
