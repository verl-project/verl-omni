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
"""CPU tests for the print_cfg.py reverse-merge fix.

Ensures that the full config composition (hydra omni_trainer + raw YAML
reverse-merge) preserves keys defined in ``omni_trainer.yaml`` even when
they are not present in verl's structured ``ppo_trainer`` schema.
"""

import os
import tempfile

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl_omni

_OMNI_CONFIG_DIR = os.path.join(os.path.dirname(verl_omni.__file__), "trainer", "config")


@pytest.fixture(scope="module")
def _composed_cfg():
    """Return the hydra-composed omni_trainer config (cached per module)."""
    # The omni_trainer.yaml already has hydra.searchpath: [pkg://verl.trainer.config]
    # so defaults like ppo_trainer resolve against the installed verl package.
    with initialize_config_dir(config_dir=_OMNI_CONFIG_DIR, version_base=None):
        return compose(config_name="omni_trainer")


class TestReverseMergePreservesRawKeys:
    """Verify that OmegaConf.merge(raw, composed) keeps keys from the raw
    YAML that are absent from the hydra-composed DictConfig.

    This is the core logic added to ``scripts/print_cfg.py`` to fix the
    generated-config CI check.
    """

    def test_extra_top_level_key_survives(self, _composed_cfg):
        raw = OmegaConf.create({"_test_extra_key": 1})
        merged = OmegaConf.merge(raw, _composed_cfg)
        assert merged._test_extra_key == 1

    def test_extra_nested_key_survives(self, _composed_cfg):
        raw = OmegaConf.create({"trainer": {"v1": {"test_struct_fix_field": {"value": 42}}}})
        merged = OmegaConf.merge(raw, _composed_cfg)
        assert OmegaConf.select(merged, "trainer.v1.test_struct_fix_field.value") == 42

    def test_existing_key_is_overridden_by_composed(self, _composed_cfg):
        raw = OmegaConf.create({"trainer": {"total_epochs": 999}})
        merged = OmegaConf.merge(raw, _composed_cfg)
        # Composed value wins (the second arg); raw's 999 is replaced by the
        # real default from ppo_trainer.
        assert merged.trainer.total_epochs == _composed_cfg.trainer.total_epochs


class TestEndToEndWithInjectedField:
    """End-to-end test: inject a synthetic key into omni_trainer.yaml, compose
    with hydra, apply the reverse merge, and verify the key is visible.
    """

    def test_injected_field_survives_reverse_merge(self, _composed_cfg):
        # Load the *real* omni_trainer.yaml as a plain dict so we can inject
        # a synthetic field that verl's schema has never seen.
        raw_path = os.path.join(_OMNI_CONFIG_DIR, "omni_trainer.yaml")
        with open(raw_path) as fh:
            raw_dict = yaml.safe_load(fh)

        raw_dict.setdefault("trainer", {}).setdefault("v1", {})
        raw_dict["trainer"]["v1"]["_test_ci_guard_field"] = {"enabled": True, "count": 7}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.dump(raw_dict, tf)
            tmp_path = tf.name

        try:
            raw = OmegaConf.load(tmp_path)
            merged = OmegaConf.merge(raw, _composed_cfg)
            assert OmegaConf.select(merged, "trainer.v1._test_ci_guard_field.enabled") is True
            assert OmegaConf.select(merged, "trainer.v1._test_ci_guard_field.count") == 7
            # Existing keys from the composed config are still present.
            assert merged.trainer.v1.trainer_mode == _composed_cfg.trainer.v1.trainer_mode
        finally:
            os.unlink(tmp_path)
