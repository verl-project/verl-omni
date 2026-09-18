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

import json
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


class TestProcessorPreparationHook:
    def test_external_library_can_override_registered_adapter(self):
        @DiffusionModelBase.register("_ExternalOverridePipeline", algorithm="flow_grpo")
        class _BuiltinModel(DiffusionModelBase):
            pass

        class _ExternalModel(DiffusionModelBase):
            pass

        def import_external(_external_lib):
            DiffusionModelBase._registry[("_ExternalOverridePipeline", "flow_grpo")] = _ExternalModel

        with patch("verl.utils.import_utils.import_external_libs", side_effect=import_external) as import_mock:
            model_cls = DiffusionModelBase.get_class_by_name(
                "_ExternalOverridePipeline",
                "flow_grpo",
                "external_adapter",
            )

        assert model_cls is _ExternalModel
        import_mock.assert_called_once_with("external_adapter")

    def test_diffusion_model_config_loads_processor_without_preparing_files(self, tmp_path):
        model_dir = tmp_path / "model"
        processor_dir = model_dir / "processor"
        processor_dir.mkdir(parents=True)
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "_ImageGenerationHookPipeline"}))
        events = []

        @DiffusionModelBase.register("_ImageGenerationHookPipeline", algorithm="flow_grpo")
        class _HookModel(DiffusionModelBase):
            @classmethod
            def prepare_processor_files(cls, model_path: str) -> None:
                raise AssertionError("processor preparation must run on the driver")

            @classmethod
            def build_scheduler(cls, model_config):
                pass

            @classmethod
            def set_timesteps(cls, scheduler, model_config, device):
                pass

            @classmethod
            def prepare_model_inputs(cls, module, model_config, *args, **kwargs):
                pass

            @classmethod
            def forward_and_sample_previous_step(cls, *args, **kwargs):
                pass

        def _fake_hf_processor(path, **kwargs):
            events.append(("processor", path))
            return "processor"

        with (
            patch("verl_omni.workers.config.diffusion.model.copy_to_local", return_value=str(model_dir)),
            patch("verl_omni.workers.config.diffusion.model.hf_tokenizer", return_value="tokenizer"),
            patch("verl_omni.workers.config.diffusion.model.hf_processor", side_effect=_fake_hf_processor),
            patch("verl_omni.workers.config.diffusion.model.import_external_libs") as import_external_mock,
        ):
            cfg = DiffusionModelConfig(
                path=str(model_dir),
                tokenizer_path=str(model_dir),
                algorithm="flow_grpo",
                attn_backend="native",
                external_lib="external_adapter",
            )

        assert cfg.processor == "processor"
        assert events == [("processor", str(processor_dir))]
        import_external_mock.assert_called_once_with("external_adapter")

    def test_driver_prepares_processor_before_loading_alternate_path(self, tmp_path):
        from verl_omni.trainer.main_diffusion import TaskRunner

        class _StopAfterProcessor(Exception):
            pass

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        alternate_processor = tmp_path / "prepared-processor"
        alternate_processor.mkdir()
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "_AlternateProcessorPipeline"}))
        events = []

        @DiffusionModelBase.register("_AlternateProcessorPipeline", algorithm="flow_grpo")
        class _AlternateProcessorModel(DiffusionModelBase):
            @classmethod
            def prepare_processor_files(cls, model_path: str) -> str:
                events.append(("hook", model_path))
                return str(alternate_processor)

            @classmethod
            def build_scheduler(cls, model_config):
                pass

            @classmethod
            def set_timesteps(cls, scheduler, model_config, device):
                pass

            @classmethod
            def prepare_model_inputs(cls, module, model_config, *args, **kwargs):
                pass

            @classmethod
            def forward_and_sample_previous_step(cls, *args, **kwargs):
                pass

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {
                        "path": str(model_dir),
                        "tokenizer_path": str(model_dir),
                        "architecture": None,
                        "algorithm": "flow_grpo",
                        "external_lib": None,
                        "use_shm": False,
                    }
                },
                "data": {"trust_remote_code": False},
            }
        )

        def _fake_hf_processor(path, **kwargs):
            events.append(("processor", path))
            return "processor"

        runner = TaskRunner()
        with (
            patch.object(runner, "add_actor_rollout_worker", return_value=(object(), object())),
            patch.object(runner, "add_reward_model_resource_pool"),
            patch.object(runner, "add_ref_policy_worker"),
            patch.object(runner, "init_resource_pool_mgr", side_effect=_StopAfterProcessor),
            patch("verl_omni.utils.fs.resolve_model_local_dir", return_value=str(model_dir)),
            patch("verl.utils.hf_tokenizer", return_value="tokenizer"),
            patch("verl.utils.hf_processor", side_effect=_fake_hf_processor),
            pytest.raises(_StopAfterProcessor),
        ):
            runner.run(config)

        assert events == [
            ("hook", str(model_dir)),
            ("processor", str(alternate_processor)),
        ]
