# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""``trainer.max_colocate_count`` must reach the V1 base's own ``ResourcePoolManager``.

The upstream V1 trainer builds the manager itself with the default of three colocated
worker groups per GPU bundle; a second colocated reward engine needs a fourth slot. The
first dual-engine run hung silently before step 0 because the knob only reached
``main_omni.init_resource_pool_mgr``, which this trainer never calls.
"""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from verl.single_controller.ray.base import ResourcePoolManager
from verl.trainer.ppo.utils import Role

from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync


def _trainer(trainer_cfg: dict) -> OmniBagelCoRLTrainerSync:
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create({"trainer": trainer_cfg})
    return trainer


def _fake_base_init(self):
    self.role_worker_mapping = {Role.ActorRollout: object()}
    self.mapping = {Role.ActorRollout: "global_pool"}
    self.resource_pool_manager = ResourcePoolManager(resource_pool_spec={"global_pool": [4]}, mapping=self.mapping)


@pytest.mark.parametrize("configured, expected", [({}, 3), ({"max_colocate_count": 4}, 4)])
def test_max_colocate_count_reaches_the_v1_resource_pool_manager(configured, expected):
    trainer = _trainer(configured)
    with patch(
        "verl_omni.trainer.omni.bagel_corl_trainer.OmniPPOTrainerSync._init_resource_pool_mgr",
        _fake_base_init,
    ):
        trainer._init_resource_pool_mgr()
    assert trainer.resource_pool_manager.max_colocate_count == expected


def test_max_colocate_count_rejects_zero():
    trainer = _trainer({"max_colocate_count": 0})
    with (
        patch(
            "verl_omni.trainer.omni.bagel_corl_trainer.OmniPPOTrainerSync._init_resource_pool_mgr",
            _fake_base_init,
        ),
        pytest.raises(ValueError, match="max_colocate_count"),
    ):
        trainer._init_resource_pool_mgr()
