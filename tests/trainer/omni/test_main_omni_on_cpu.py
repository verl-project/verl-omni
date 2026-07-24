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
"""CPU tests for omni training entrypoint routing."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer, PolicyGradientRayTrainer
from verl_omni.trainer.main_omni import RayTrainerTaskRunner, get_ray_trainer_cls, uses_v1_trainer
from verl_omni.trainer.omni.ray_omni_dpo_trainer import OmniDirectPreferenceRayTrainer


def _make_config(**overrides):
    config = OmegaConf.create(
        {
            "algorithm": {
                "sample_source": "online",
                "trainer_type": "policy_gradient",
            },
            "actor_rollout_ref": {
                "model": {"model_type": "omni_model"},
            },
            "trainer": {"use_v1": None},
        }
    )
    OmegaConf.set_struct(config, False)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and OmegaConf.is_dict(config[key]):
            config[key] = OmegaConf.merge(config[key], value)
        else:
            config[key] = value
    return config


class TestUsesV1Trainer:
    @pytest.mark.parametrize(
        ("sample_source", "trainer_type", "use_v1", "expected"),
        [
            ("online", "policy_gradient", None, True),
            ("offline", "direct_preference", None, False),
            ("offline", "direct_preference", False, False),
            ("online", "direct_preference", None, True),
        ],
    )
    def test_routes_online_rl_to_v1_and_offline_preference_to_legacy(
        self, sample_source, trainer_type, use_v1, expected
    ):
        config = _make_config(
            algorithm={"sample_source": sample_source, "trainer_type": trainer_type},
            trainer={"use_v1": use_v1},
        )
        assert uses_v1_trainer(config) is expected


class TestGetRayTrainerCls:
    def test_policy_gradient_returns_policy_gradient_trainer(self):
        config = _make_config(algorithm={"trainer_type": "policy_gradient"})
        assert get_ray_trainer_cls(config) is PolicyGradientRayTrainer

    def test_direct_preference_omni_returns_omni_trainer(self):
        config = _make_config(
            algorithm={"trainer_type": "direct_preference"},
            actor_rollout_ref={"model": {"model_type": "omni_model"}},
        )
        assert get_ray_trainer_cls(config) is OmniDirectPreferenceRayTrainer

    def test_direct_preference_non_omni_returns_diffusion_trainer(self):
        config = _make_config(
            algorithm={"trainer_type": "direct_preference"},
            actor_rollout_ref={"model": {"model_type": "diffusion_model"}},
        )
        assert get_ray_trainer_cls(config) is DirectPreferenceRayTrainer

    def test_unsupported_trainer_type_raises(self):
        config = _make_config(algorithm={"trainer_type": "unknown"})
        with pytest.raises(ValueError, match="Unsupported trainer_type"):
            get_ray_trainer_cls(config)


class TestLoadTokenizerAndProcessor:
    def test_omni_model_config_loads_tokenizer_and_processor_via_adapter(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from verl_omni.workers.config.omni import model as model_config_module
        from verl_omni.workers.config.omni.model import OmniModelConfig

        mock_adapter = MagicMock()
        mock_adapter.configure_tokenizer.return_value = "tokenizer"
        mock_adapter.configure_processor.return_value = "processor"
        monkeypatch.setattr(
            "verl_omni.pipelines.model_base.OmniModelBase.get_class_by_name",
            lambda *_args, **_kwargs: mock_adapter,
        )
        monkeypatch.setattr(model_config_module, "resolve_model_local_dir", lambda path, use_shm=False: str(tmp_path))
        monkeypatch.setattr(model_config_module, "copy_to_local", lambda path, use_shm=False: f"local:{path}")
        monkeypatch.setattr(
            model_config_module.AutoConfig,
            "from_pretrained",
            lambda *_args, **_kwargs: SimpleNamespace(tie_word_embeddings=False, architectures=["arch"]),
        )

        model_config = OmniModelConfig(
            path=str(tmp_path),
            architecture="Qwen3OmniMoeForConditionalGeneration",
            model_stage="thinker",
            tokenizer_path="tokenizer-path",
            external_lib=None,
            trust_remote_code=False,
        )

        assert model_config.tokenizer == "tokenizer"
        assert model_config.processor == "processor"
        mock_adapter.configure_tokenizer.assert_called_once_with("local:tokenizer-path", model_config)
        mock_adapter.configure_processor.assert_called_once_with(str(tmp_path), model_config)

    def test_ray_task_runner_delegates_omni_model_to_adapter(self, monkeypatch, tmp_path):
        config = OmegaConf.create(
            {
                "data": {"trust_remote_code": False},
                "actor_rollout_ref": {
                    "model": {
                        "model_type": "omni_model",
                        "architecture": "Qwen3OmniMoeForConditionalGeneration",
                        "model_stage": "thinker",
                        "tokenizer_path": str(tmp_path),
                    }
                },
            }
        )

        mock_model_config = MagicMock()
        mock_model_config.tokenizer = "tokenizer"
        mock_model_config.get_processor.return_value = "processor"
        monkeypatch.setattr(
            "verl.utils.config.omega_conf_to_dataclass",
            lambda _cfg: mock_model_config,
        )

        tokenizer, processor = RayTrainerTaskRunner._load_tokenizer_and_processor(config, str(tmp_path))

        assert tokenizer == "tokenizer"
        assert processor == "processor"
