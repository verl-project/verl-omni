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
"""CPU tests for the legacy UnifiedReward sampling contract of VisualRewardManager."""

import functools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn

from verl_omni.reward_loop.reward_manager import visual as visual_module
from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager
from verl_omni.utils.reward_score import unified_reward, vlm_as_a_judge

REPO_ROOT = Path(__file__).parents[2]
SHARED_PATH = "pkg://verl_omni.utils.reward_score.vlm_as_a_judge"
COMPAT_PATH = "pkg://verl_omni.utils.reward_score.unified_reward"
UNIFIED_OUTPUT = "Alignment Score (1-5): 4\nCoherence Score (1-5): 5\nStyle Score (1-5): 3"


async def compute_score_unified_reward(**kwargs):
    """Same-name scorer outside this repository; it must keep rollout sampling."""
    return {"score": 0.5, "sampling_params": kwargs.get("sampling_params")}


def _config(
    *,
    path=None,
    name="compute_score_unified_reward",
    reward_kwargs=None,
    response_length=2048,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    full_determinism=False,
    seed=42,
):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"pipeline": {"output_type": "image"}}},
            "reward": {
                "custom_reward_function": {
                    "path": path,
                    "name": name,
                    "reward_kwargs": reward_kwargs or {},
                },
                "reward_model": {
                    "model_path": "judge-model",
                    "rollout": {
                        "response_length": response_length,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "full_determinism": full_determinism,
                        "seed": seed,
                    },
                },
            },
        }
    )


def _data() -> DataProto:
    return DataProto.from_dict(
        tensors={"responses": torch.zeros(1, 3, 8, 8, dtype=torch.uint8)},
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "a cat"}],
            "media_kind": ["image"],
            "extra_info": [{}],
        },
    )


def _capture_chat(monkeypatch, content=UNIFIED_OUTPUT) -> list[dict]:
    requests: list[dict] = []

    async def transport(router_address, chat_complete_request, session=None):
        requests.append(chat_complete_request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(vlm_as_a_judge, "_post_chat_completion", transport)
    return requests


def _manager(config, compute_score) -> VisualRewardManager:
    return VisualRewardManager(config, tokenizer=None, compute_score=compute_score, reward_router_address="judge:8000")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [SHARED_PATH, COMPAT_PATH], ids=["shared", "compat"])
@pytest.mark.parametrize("response_length", [2048, 4096])
async def test_config_loaded_unified_reward_ignores_rollout_response_length(monkeypatch, path, response_length):
    requests = _capture_chat(monkeypatch)
    config = _config(path=path, response_length=response_length)

    result = await _manager(config, get_custom_reward_fn(config)).run_single(_data())

    assert requests[0]["max_tokens"] == 512
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["top_p"] == 1.0
    assert result["reward_score"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_tuned_rollout_fields_are_kept_and_config_is_not_mutated(monkeypatch):
    requests = _capture_chat(monkeypatch)
    config = _config(
        path=SHARED_PATH,
        response_length=2048,
        temperature=0.4,
        top_p=0.9,
        top_k=20,
        full_determinism=True,
        seed=123,
    )
    defaults_before = dict(vlm_as_a_judge.DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS)

    result = await _manager(config, get_custom_reward_fn(config)).run_single(_data())

    assert requests[0]["max_tokens"] == 512
    assert requests[0]["temperature"] == 0.4
    assert requests[0]["top_p"] == 0.9
    assert requests[0]["top_k"] == 20
    assert requests[0]["seed"] == 123
    assert result["reward_score"] == pytest.approx(0.75)
    assert config.reward.reward_model.rollout.response_length == 2048
    assert config.reward.reward_model.rollout.temperature == 0.4
    assert vlm_as_a_judge.DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS == defaults_before


@pytest.mark.asyncio
async def test_explicit_reward_kwargs_max_tokens_wins_over_rollout_length(monkeypatch):
    requests = _capture_chat(monkeypatch)
    config = _config(
        path=SHARED_PATH,
        reward_kwargs={"sampling_params": {"max_tokens": 64, "temperature": 0.2}},
        response_length=4096,
    )

    await _manager(config, get_custom_reward_fn(config)).run_single(_data())

    assert requests[0]["max_tokens"] == 64
    assert requests[0]["temperature"] == 0.2
    assert requests[0]["top_p"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scorer",
    [vlm_as_a_judge.compute_score_unified_reward, unified_reward.compute_score_unified_reward],
    ids=["shared-alias", "compat-alias"],
)
async def test_directly_injected_alias_uses_scorer_default_max_tokens(monkeypatch, scorer):
    requests = _capture_chat(monkeypatch)
    config = _config(path=None, response_length=4096)

    await _manager(config, scorer).run_single(_data())

    assert requests[0]["max_tokens"] == 512


@pytest.mark.asyncio
async def test_direct_foreign_same_name_scorer_forwards_rollout_response_length():
    config = _config(path=None, response_length=2048)

    result = await _manager(config, compute_score_unified_reward).run_single(_data())

    assert result["reward_extra_info"]["sampling_params"]["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_standard_partial_sampling_params_take_priority(monkeypatch):
    requests = _capture_chat(monkeypatch)
    scorer = functools.partial(
        vlm_as_a_judge.compute_score_unified_reward,
        sampling_params={"max_tokens": 33, "seed": 5},
    )
    config = _config(path=None, response_length=4096)

    await _manager(config, scorer).run_single(_data())

    assert requests[0]["max_tokens"] == 33
    assert requests[0]["seed"] == 5
    assert requests[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_ocr_scorer_still_forwards_rollout_response_length(monkeypatch):
    requests = _capture_chat(monkeypatch, content="a cat")
    config = _config(path=SHARED_PATH, name="compute_score_ocr", response_length=2048)

    result = await _manager(config, get_custom_reward_fn(config)).run_single(_data())

    assert requests[0]["max_tokens"] == 2048
    assert result["reward_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_unknown_same_name_scorer_still_forwards_rollout_response_length(monkeypatch):
    config = _config(path=str(Path(__file__).resolve()), response_length=2048)

    result = await _manager(config, get_custom_reward_fn(config)).run_single(_data())

    assert result["reward_extra_info"]["sampling_params"]["max_tokens"] == 2048


def test_repo_unified_reward_source_paths_are_normalized():
    assert visual_module._is_repo_unified_reward_path(SHARED_PATH)
    assert visual_module._is_repo_unified_reward_path(COMPAT_PATH)
    assert visual_module._is_repo_unified_reward_path("verl_omni/utils/reward_score/unified_reward.py")
    assert visual_module._is_repo_unified_reward_path(
        str((REPO_ROOT / "verl_omni/utils/reward_score/vlm_as_a_judge.py").resolve())
    )
    assert not visual_module._is_repo_unified_reward_path(str(Path(__file__).resolve()))
