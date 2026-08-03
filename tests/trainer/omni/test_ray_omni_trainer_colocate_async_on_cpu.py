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

from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
from verl.trainer.ppo.v1.trainer_base import get_trainer_cls
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

from verl_omni.trainer.omni.ray_omni_trainer_colocate_async import (
    OmniPPOTrainerColocateAsync,
)


class TestOmniColocateAsyncRegistration:
    def test_registered_under_omni_colocate_async(self):
        assert get_trainer_cls("omni_colocate_async") is OmniPPOTrainerColocateAsync

    def test_subclass_of_ppo_trainer_colocate_async(self):
        assert issubclass(OmniPPOTrainerColocateAsync, PPOTrainerColocateAsync)


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
    def test_warmup_reads_from_omni_colocate_async_key(self):
        trainer = OmniPPOTrainerColocateAsync.__new__(OmniPPOTrainerColocateAsync)
        trainer.config = OmegaConf.create(
            {
                "skip": {"rollout_tq": {"enable": False}},
                "trainer": {
                    "v1": {
                        "omni_colocate_async": {"num_warmup_batches": 3},
                    }
                },
            }
        )

        with patch.object(trainer, "_add_batch_to_generate") as mock_add:
            trainer.on_train_begin()

        assert mock_add.call_count == 3

    def test_skip_rollout_tq_disables_warmup(self):
        trainer = OmniPPOTrainerColocateAsync.__new__(OmniPPOTrainerColocateAsync)
        trainer.config = OmegaConf.create(
            {
                "skip": {"rollout_tq": {"enable": True}},
                "trainer": {
                    "v1": {
                        "omni_colocate_async": {"num_warmup_batches": 3},
                    }
                },
            }
        )

        with patch.object(trainer, "_add_batch_to_generate") as mock_add:
            trainer.on_train_begin()

        assert mock_add.call_count == 0
