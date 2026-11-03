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
"""CPU tests for the ``grpo_std_floor`` UND advantage estimator."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, get_adv_estimator_fn

from verl_omni.trainer.config.algorithm import OmniAlgoConfig
from verl_omni.trainer.omni.bagel_corl_und_adv import (
    DEFAULT_ADV_STD_FLOOR,
    GRPO_STD_FLOOR,
    compute_grpo_std_floor_outcome_advantage,
    resolve_adv_std_floor,
)


def _rewards(scores: list[float], resp_len: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Outcome rewards on the last token, full response mask."""
    tlr = torch.zeros(len(scores), resp_len)
    tlr[:, -1] = torch.tensor(scores)
    return tlr, torch.ones(len(scores), resp_len)


def _cfg(floor: float):
    return OmegaConf.create({"adv_std_floor": floor})


def test_registered_under_adv_estimator_name():
    assert get_adv_estimator_fn(GRPO_STD_FLOOR) is compute_grpo_std_floor_outcome_advantage


def test_near_tie_group_is_scaled_by_floor_not_its_own_std():
    scores = [0.500, 0.504, 0.508, 0.512]  # spread 0.012: image-seed noise
    tlr, mask = _rewards(scores)
    index = np.array(["p"] * 4)

    adv, ret = compute_grpo_std_floor_outcome_advantage(tlr, mask, index, config=_cfg(0.05))
    std_adv, _ = compute_grpo_outcome_advantage(tlr, mask, index, norm_adv_by_std_in_grpo=True)

    assert torch.equal(adv, ret)
    # Plain std-GRPO blows the tie up to roughly +/-1.1; the floor keeps it at ~0.1.
    assert std_adv.abs().max() > 1.0
    assert adv.abs().max() < 0.15
    expected = (torch.tensor(scores) - np.mean(scores)) / 0.05
    torch.testing.assert_close(adv[:, 0], expected.float(), atol=1e-4, rtol=0)


def test_real_spread_group_matches_std_normalized_grpo():
    scores = [0.30, 0.45, 0.60, 0.75]  # std ~0.19 >> floor
    tlr, mask = _rewards(scores)
    index = np.array(["p"] * 4)

    adv, _ = compute_grpo_std_floor_outcome_advantage(tlr, mask, index, config=_cfg(0.05))
    std_adv, _ = compute_grpo_outcome_advantage(tlr, mask, index, norm_adv_by_std_in_grpo=True)
    torch.testing.assert_close(adv, std_adv, atol=1e-5, rtol=0)


def test_zero_floor_reduces_to_std_grpo_and_large_floor_to_raw_over_floor():
    scores = [0.1, 0.2, 0.3, 0.4]
    tlr, mask = _rewards(scores)
    index = np.array(["p"] * 4)

    zero, _ = compute_grpo_std_floor_outcome_advantage(tlr, mask, index, config=_cfg(0.0))
    std_adv, _ = compute_grpo_outcome_advantage(tlr, mask, index, norm_adv_by_std_in_grpo=True)
    torch.testing.assert_close(zero, std_adv, atol=1e-5, rtol=0)

    big, _ = compute_grpo_std_floor_outcome_advantage(tlr, mask, index, config=_cfg(1.0))
    raw, _ = compute_grpo_outcome_advantage(tlr, mask, index, norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(big, raw / (1.0 + 1e-6), atol=1e-5, rtol=0)


def test_groups_are_independent_and_masked():
    scores = [0.5, 0.5, 0.0, 1.0, 0.7]  # tie group, spread group, singleton
    tlr, mask = _rewards(scores, resp_len=3)
    mask[1, 2] = 0.0
    index = np.array(["a", "a", "b", "b", "c"])

    adv, _ = compute_grpo_std_floor_outcome_advantage(tlr, mask, index, config=_cfg(0.05))

    assert torch.all(adv[0] == 0) and torch.all(adv[1] == 0)  # exact tie -> no push
    assert adv[1, 2] == 0  # masked token
    assert adv[2, 0] < 0 < adv[3, 0]
    torch.testing.assert_close(adv[3, 0], -adv[2, 0])
    torch.testing.assert_close(adv[4, 0], torch.tensor(0.7 / (1 + 1e-6)))  # singleton: raw score


def test_resolve_floor_defaults_and_validates():
    assert resolve_adv_std_floor(None) == DEFAULT_ADV_STD_FLOOR
    assert resolve_adv_std_floor(OmegaConf.create({})) == DEFAULT_ADV_STD_FLOOR
    assert resolve_adv_std_floor(_cfg(0.2)) == pytest.approx(0.2)
    assert resolve_adv_std_floor(OmniAlgoConfig(adv_std_floor=0.3)) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        resolve_adv_std_floor(_cfg(-0.1))


def test_omni_algo_config_field_and_validation():
    assert OmniAlgoConfig().adv_std_floor == pytest.approx(0.05)
    assert OmniAlgoConfig(adv_estimator=GRPO_STD_FLOOR, adv_std_floor=0.1).adv_std_floor == pytest.approx(0.1)
    with pytest.raises(ValueError):
        OmniAlgoConfig(adv_std_floor=-1.0)


def test_trainer_import_registers_estimator():
    """Importing the Bagel Co-RL trainer must be enough for verl to resolve the name."""
    import importlib

    importlib.import_module("verl_omni.trainer.omni.bagel_corl_trainer")
    assert get_adv_estimator_fn(GRPO_STD_FLOOR) is compute_grpo_std_floor_outcome_advantage
