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
from verl_omni.pipelines.utils import ImageGenerationRequest
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


class TestImageGenerationRequest:
    def test_request_stores_image_generation_fields(self):
        request = ImageGenerationRequest(
            prompt="make it brighter", images=["image"], negative_prompt="", metadata={"id": 1}
        )

        assert request.prompt == "make it brighter"
        assert request.images == ["image"]
        assert request.negative_prompt == ""
        assert request.metadata == {"id": 1}

    def test_request_defaults_to_t2i_without_images(self):
        request = ImageGenerationRequest(prompt="draw a cat")

        assert request.images == []

    def test_request_supports_multi_image_conditioning(self):
        request = ImageGenerationRequest(prompt="combine them", images=["img0", "img1"])

        assert request.images == ["img0", "img1"]

    def test_from_request_payload_resolves_top_level_images_first(self):
        custom_prompt = {
            "prompt": "edit instruction",
            "negative_prompt": "",
            "images": ["top-level"],
            "image": "single",
            "multi_modal_data": {"image": ["raw"]},
            "extra_args": {"multi_modal_data": {"image": ["extra"]}},
            "additional_information": {"condition_images": ["fallback"]},
        }

        request = ImageGenerationRequest.from_request_payload(custom_prompt)

        assert request.prompt == "edit instruction"
        assert request.images == ["top-level"]
        assert request.negative_prompt == ""
        assert request.metadata == {"condition_images": ["fallback"]}

    def test_from_request_payload_resolves_single_image_key(self):
        request = ImageGenerationRequest.from_request_payload({"prompt": "edit", "image": "img"})

        assert request.images == ["img"]

    def test_from_request_payload_resolves_multimodal_images(self):
        custom_prompt = {"prompt": "edit", "multi_modal_data": {"image": ["raw"]}}

        request = ImageGenerationRequest.from_request_payload(custom_prompt)

        assert request.images == ["raw"]

    def test_from_request_payload_resolves_extra_args_multimodal_images(self):
        custom_prompt = {"prompt": "edit", "extra_args": {"multi_modal_data": {"image": ("img0", "img1")}}}

        request = ImageGenerationRequest.from_request_payload(custom_prompt)

        assert request.images == ["img0", "img1"]

    def test_from_request_payload_resolves_additional_information_fallback(self):
        custom_prompt = {"prompt": "edit", "additional_information": {"condition_images": "img"}}

        request = ImageGenerationRequest.from_request_payload(custom_prompt)

        assert request.images == ["img"]

    def test_from_request_payload_allows_t2i_without_images(self):
        request = ImageGenerationRequest.from_request_payload({"prompt": "draw a cat"})

        assert request.prompt == "draw a cat"
        assert request.images == []

        request = ImageGenerationRequest.from_request_payload(
            {"prompt": "draw a cat", "extra_args": {"multi_modal_data": {"image": []}}}
        )

        assert request.images == []

    def test_from_request_payload_allows_prompt_token_ids_without_prompt_text(self):
        request = ImageGenerationRequest.from_request_payload({"prompt_token_ids": [1, 2], "images": ["img"]})

        assert request.prompt == [1, 2]
        assert request.images == ["img"]

    def test_from_request_payload_uses_token_ids_when_prompt_is_none(self):
        request = ImageGenerationRequest.from_request_payload(
            {"prompt": None, "prompt_token_ids": [1, 2], "images": ["img"]}
        )

        assert request.prompt == [1, 2]

    def test_from_request_payload_requires_prompt_or_prompt_token_ids(self):
        with pytest.raises(ValueError, match="missing required 'prompt' or 'prompt_token_ids'"):
            ImageGenerationRequest.from_request_payload({"images": ["img"]})

    def test_from_request_payload_preserves_empty_metadata(self):
        request = ImageGenerationRequest.from_request_payload(
            {
                "prompt": "edit",
                "metadata": {},
                "extra_info": {"id": 1},
                "additional_information": {"condition_images": "img"},
            }
        )

        assert request.metadata == {}


class TestProcessorPreparationHook:
    def test_diffusion_model_config_calls_registered_processor_hook(self, tmp_path):
        model_dir = tmp_path / "model"
        processor_dir = model_dir / "processor"
        processor_dir.mkdir(parents=True)
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "_ImageGenerationHookPipeline"}))
        events = []

        @DiffusionModelBase.register("_ImageGenerationHookPipeline", algorithm="flow_grpo")
        class _HookModel(DiffusionModelBase):
            @classmethod
            def prepare_processor_files(cls, model_path: str) -> None:
                events.append(("hook", model_path))

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
        ):
            cfg = DiffusionModelConfig(
                path=str(model_dir),
                tokenizer_path=str(model_dir),
                algorithm="flow_grpo",
                attn_backend="native",
            )

        assert cfg.processor == "processor"
        assert events == [("hook", str(model_dir)), ("processor", str(processor_dir))]

    def test_driver_prepares_processor_before_loading_it(self, tmp_path):
        from verl_omni.trainer.main_diffusion import _prepare_processor_files

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "_DriverHookPipeline"}))
        events = []

        @DiffusionModelBase.register("_DriverHookPipeline", algorithm="flow_grpo")
        class _DriverHookModel(DiffusionModelBase):
            @classmethod
            def prepare_processor_files(cls, model_path: str) -> None:
                events.append(("hook", model_path))

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

        model_config = OmegaConf.create({"architecture": None, "algorithm": "flow_grpo", "external_lib": None})
        _prepare_processor_files(str(model_dir), model_config)
        events.append(("processor", str(model_dir / "processor")))

        assert events == [
            ("hook", str(model_dir)),
            ("processor", str(model_dir / "processor")),
        ]

    def test_driver_accepts_alternate_processor_path(self, tmp_path):
        from verl_omni.trainer.main_diffusion import _prepare_processor_files

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        alternate_processor = tmp_path / "prepared-processor"
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "_AlternateProcessorPipeline"}))

        @DiffusionModelBase.register("_AlternateProcessorPipeline", algorithm="flow_grpo")
        class _AlternateProcessorModel(DiffusionModelBase):
            @classmethod
            def prepare_processor_files(cls, model_path: str) -> str:
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

        model_config = OmegaConf.create({"architecture": None, "algorithm": "flow_grpo", "external_lib": None})

        assert _prepare_processor_files(str(model_dir), model_config) == str(alternate_processor)
