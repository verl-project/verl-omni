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
"""GEN FlowGRPO can normalize advantages differently from UND token GRPO."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(("override", "expected"), [(None, False), (True, True), (False, False)])
def test_gen_lane_view_takes_its_own_norm_flag(override, expected):
    from omegaconf import OmegaConf

    from verl_omni.trainer.omni.bagel_corl_diff_v1 import _lane_config_with_gen_estimator

    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": False,
                "gen_norm_adv_by_std_in_grpo": override,
            }
        }
    )
    view = _lane_config_with_gen_estimator(config, "flow_grpo")
    assert view.algorithm.adv_estimator == "flow_grpo"
    assert view.algorithm.norm_adv_by_std_in_grpo is expected
    assert config.algorithm.norm_adv_by_std_in_grpo is False


def test_gen_lane_namespace_view_takes_its_own_norm_flag():
    from verl_omni.trainer.omni.bagel_corl_diff_v1 import _lane_config_with_gen_estimator

    config = SimpleNamespace(
        algorithm=SimpleNamespace(adv_estimator="grpo", norm_adv_by_std_in_grpo=False, gen_norm_adv_by_std_in_grpo=True)
    )
    view = _lane_config_with_gen_estimator(config, "flow_grpo")
    assert view.algorithm.norm_adv_by_std_in_grpo is True
    assert config.algorithm.norm_adv_by_std_in_grpo is False
