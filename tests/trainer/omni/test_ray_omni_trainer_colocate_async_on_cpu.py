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

"""CPU tests for ``OmniPPOTrainerColocateAsync`` (RFC #320 §7): registration,
``_init_tokenizer`` wiring, and warmup flowing from the generic
``trainer.v1.colocate_async`` key.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.trainer_base import get_trainer_cls
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

from verl_omni.trainer.omni.ray_omni_trainer_colocate_async import (
    OmniPPOTrainerColocateAsync,
)

_CONFIG_DIR = str((Path(__file__).parents[3] / "verl_omni" / "trainer" / "config").resolve())

_BASE_OVERRIDES = [
    "trainer.v1.trainer_mode=omni_colocate_async",
    "actor_rollout_ref.rollout.nnodes=1",
    "actor_rollout_ref.rollout.n_gpus_per_node=2",
    "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    "actor_rollout_ref.model.path=/dummy/model",
]


def _compose_config(extra_overrides=()):
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name="omni_trainer", overrides=[*_BASE_OVERRIDES, *extra_overrides])


class TestOmniColocateAsyncRegistration:
    def test_registered_under_omni_colocate_async(self):
        assert get_trainer_cls("omni_colocate_async") is OmniPPOTrainerColocateAsync

    def test_subclass_of_ppo_trainer_colocate_async(self):
        assert issubclass(OmniPPOTrainerColocateAsync, PPOTrainerColocateAsync)

    def test_unknown_mode_fails_fast(self):
        with pytest.raises(ValueError, match="Unknown trainer"):
            get_trainer_cls("omni_not_a_mode")


class TestOmniColocateAsyncInitTokenizer:
    def test_init_tokenizer_wires_omni_model_config(self):
        trainer = OmniPPOTrainerColocateAsync.__new__(OmniPPOTrainerColocateAsync)
        trainer.config = OmegaConf.create({"actor_rollout_ref": {"model": {"path": "/dummy"}}})

        fake_cfg = SimpleNamespace(tokenizer="fake_tok", processor="fake_proc")
        with patch(
            "verl_omni.trainer.omni.ray_omni_trainer_colocate_async.omega_conf_to_dataclass",
            return_value=fake_cfg,
        ):
            trainer._init_tokenizer()

        assert trainer.tokenizer == "fake_tok"
        assert trainer.processor == "fake_proc"


class TestOmniColocateAsyncOnTrainBegin:
    @staticmethod
    def _config(num_warmup_batches, skip_warmup=False):
        # on_train_begin is inherited from PPOTrainerColocateAsync and must read
        # the generic trainer.v1.colocate_async key, not an omni_* stub (#320 §3.6).
        return OmegaConf.create(
            {
                "skip": {"rollout_tq": {"enable": skip_warmup}},
                "trainer": {
                    "v1": {
                        "colocate_async": {"num_warmup_batches": num_warmup_batches},
                    }
                },
            }
        )

    def test_warmup_reads_generic_colocate_async_key(self):
        trainer = OmniPPOTrainerColocateAsync.__new__(OmniPPOTrainerColocateAsync)
        trainer.config = self._config(num_warmup_batches=3)

        with patch.object(trainer, "_add_batch_to_generate") as mock_add:
            trainer.on_train_begin()

        assert mock_add.call_count == 3

    def test_omni_key_if_present_is_ignored(self):
        trainer = OmniPPOTrainerColocateAsync.__new__(OmniPPOTrainerColocateAsync)
        trainer.config = self._config(num_warmup_batches=2)
        OmegaConf.update(trainer.config, "trainer.v1.omni_colocate_async.num_warmup_batches", 7)

        with patch.object(trainer, "_add_batch_to_generate") as mock_add:
            trainer.on_train_begin()

        assert mock_add.call_count == 2

    def test_skip_guard_disables_warmup(self):
        trainer = OmniPPOTrainerColocateAsync.__new__(OmniPPOTrainerColocateAsync)
        trainer.config = self._config(num_warmup_batches=3, skip_warmup=True)

        with patch.object(trainer, "_add_batch_to_generate") as mock_add:
            trainer.on_train_begin()

        mock_add.assert_not_called()


class TestOmniColocateAsyncConfig:
    def test_composed_config_resolves_end_to_end_with_only_generic_stubs(self):
        # RFC #320 §7.7: trainer_mode=omni_colocate_async must resolve end-to-end
        # with only the generic v1.colocate_async stubs — no omni_* keys required.
        trainer = OmniPPOTrainerColocateAsync(_compose_config())

        assert trainer.config.trainer.v1.trainer_mode == "omni_colocate_async"
        assert trainer.config.trainer.v1.colocate_async.num_warmup_batches == 1
        assert "omni_colocate_async" not in trainer.config.trainer.v1
