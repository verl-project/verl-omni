# Copyright 2026 verl-omni contributors
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
"""Exercise the public selector through the actual Hydra composition."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[3] / "verl_omni/trainer/config"


@pytest.mark.parametrize("config_name", ["omni_trainer", "omni_megatron_trainer"])
def test_selector_can_be_enabled_without_ad_hoc_hydra_key(config_name):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        default = compose(config_name=config_name)
        selected = compose(config_name=config_name, overrides=["trainer.v1.step_extension=proximal_actor_reuse"])
    assert default.trainer.v1.step_extension is None
    assert selected.trainer.v1.step_extension == "proximal_actor_reuse"
    reference = OmegaConf.load(CONFIG_DIR / f"_generated_{config_name}.yaml")
    assert reference.trainer.v1.step_extension is None
