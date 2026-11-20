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

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_21_diffusion_nft.agent_loop import QwenImage21DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.qwen_image_21_diffusion_nft.diffusers_training_adapter import QwenImage21DiffusionNFT
from verl_omni.pipelines.qwen_image_21_diffusion_nft.prompt import prepare_reference_media
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
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


@pytest.fixture
def processor_checkpoint(tmp_path, monkeypatch):
    """只替换外部模型加载，保留配置、registry 和 agent 的生产控制流。"""
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    encoder_dir = tmp_path / "text_encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "text_config": {}}))
    tokenizer = Mock()
    tokenizer.encode.return_value = [198]
    tokenizer.convert_tokens_to_ids.return_value = 151645
    processor = Mock(spec=["tokenizer"], tokenizer=tokenizer)
    processor.return_value = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long)
    )

    def load_processor(path, **kwargs):
        assert json.loads((Path(path) / "config.json").read_text())["model_type"] == "qwen3_vl"
        return processor

    loader = Mock(side_effect=load_processor)
    monkeypatch.setattr("verl_omni.workers.config.diffusion.model.hf_processor", loader)
    monkeypatch.setattr("verl.utils.hf_processor", loader)
    monkeypatch.setattr("verl_omni.workers.config.diffusion.model.hf_tokenizer", Mock(return_value=tokenizer))
    monkeypatch.setattr("verl.utils.hf_tokenizer", Mock(return_value=tokenizer))
    return tmp_path, tokenizer, processor, loader


@pytest.mark.parametrize("nrefs", range(5))
def test_model_config_processor_reaches_agent_tokenization(processor_checkpoint, nrefs):
    root, tokenizer, processor, loader = processor_checkpoint
    QwenImage21DiffusionNFT.prepare_processor_files(str(root))
    config = DiffusionModelConfig(
        path=str(root),
        tokenizer_path=str(root / "processor"),
        algorithm="diffusion_nft",
        attn_backend="native",
    )
    assert config.processor is processor
    loader.assert_called_once_with(str(root / "processor"), trust_remote_code=False)
    assert (root / "processor/config.json").read_bytes() == (root / "text_encoder/config.json").read_bytes()
    assert DiffusionModelBase.get_class(config) is QwenImage21DiffusionNFT

    async def tokenize():
        rollout = SimpleNamespace(prompt_length=32768)
        trainer = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)))
        agent = QwenImage21DiffusionSingleTurnAgentLoop(
            trainer,
            server_manager=object(),
            tokenizer=tokenizer,
            processor=config.processor,
            dataset_cls=None,
            data_config=SimpleNamespace(config={}),
        )
        images = [Image.new("RGB", (64, 64), (40 * index, 80, 120)) for index in range(nrefs)]
        media = prepare_reference_media({"images": images})
        ids = await agent.ct_build_initial_tokens([{"role": "user", "content": "draw a sign"}], images=media["images"])
        return ids, media

    ids, media = asyncio.run(tokenize())
    assert ids == [1, 2, 3]
    assert len(media["images"]) == nrefs
    processor.assert_called_once()
    assert processor.call_args.kwargs["text"][0].count("<|image_pad|>") == nrefs


def test_v1_driver_uses_existing_processor_preparation(processor_checkpoint):
    root, _, processor, loader = processor_checkpoint
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": str(root),
                    "tokenizer_path": str(root / "processor"),
                    "algorithm": "diffusion_nft",
                }
            },
            "data": {"trust_remote_code": False},
        }
    )
    trainer = SimpleNamespace(config=config)
    PolicyGradientDiffusionTrainerV1._init_tokenizer(trainer)
    assert trainer.processor is processor
    loader.assert_called_once_with(
        str(root / "processor"),
        trust_remote_code=False,
        use_fast=True,
    )
    assert (root / "processor/config.json").read_bytes() == (root / "text_encoder/config.json").read_bytes()


def test_processor_preparation_preserves_existing_config(processor_checkpoint):
    root, *_ = processor_checkpoint
    config = root / "processor/config.json"
    content = '{"model_type": "qwen3_vl", "checkpoint_metadata": true}'
    config.write_text(content)
    assert QwenImage21DiffusionNFT.prepare_processor_files(str(root)) == str(config.parent)
    assert config.read_text() == content


def test_processor_preparation_rejects_wrong_encoder(processor_checkpoint):
    root, *_ = processor_checkpoint
    (root / "text_encoder/config.json").write_text('{"model_type": "qwen2_vl"}')
    with pytest.raises(ValueError, match="Qwen3-VL"):
        QwenImage21DiffusionNFT.prepare_processor_files(str(root))
    assert not (root / "processor/config.json").exists()


@pytest.mark.parametrize("nrefs", [0, 1, 4])
def test_qwen21_agent_reuses_base_run_with_same_condition_images(monkeypatch, nrefs):
    from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop

    async def run():
        source = [Image.new("RGBA", (96, 64), (10, 20, 30, 80)) for _ in range(nrefs)]
        monkeypatch.setattr(
            DiffusionSingleTurnAgentLoop, "process_multi_modal_info", AsyncMock(return_value={"images": source})
        )
        agent = object.__new__(QwenImage21DiffusionSingleTurnAgentLoop)
        agent.loop = asyncio.get_running_loop()
        agent.rollout_config = SimpleNamespace(
            prompt_length=32768, enable_prompt_embed_cache=False, enable_prompt_embed_cache_routing_affinity=False
        )
        agent.processor = Mock(
            return_value=SimpleNamespace(
                input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long)
            )
        )
        agent.extra_tokenizer_map = {}
        agent._assert_mm_supported = Mock()
        agent._get_mm_processor_kwargs = Mock(return_value={})
        output = SimpleNamespace(
            num_preempted=0, diffusion_output=torch.zeros(3, 8, 8), log_probs=None, extra_fields={}
        )
        agent.server_manager = SimpleNamespace(generate=AsyncMock(return_value=output))
        assert QwenImage21DiffusionSingleTurnAgentLoop.run is DiffusionSingleTurnAgentLoop.run
        result = await agent.run(
            {},
            raw_prompt=[{"role": "user", "content": "edit"}],
            raw_negative_prompt=[{"role": "user", "content": " "}],
        )
        sent = agent.server_manager.generate.call_args.kwargs
        assert result.prompt_ids == sent["prompt_ids"] == sent["negative_prompt_ids"] == [1, 2, 3]
        assert agent.processor.call_count == 2
        assert len(sent["image_data"]) == nrefs
        for call in agent.processor.call_args_list:
            assert call.kwargs["text"][0].count("<|image_pad|>") == nrefs
            assert [image.size for image in call.kwargs.get("images", [])] == [
                image.size for image in sent["image_data"]
            ]
        assert all(image.mode == "RGBA" for image in sent["image_data"])
        assert all(image.size == (96, 64) for image in source)

    asyncio.run(run())


def test_qwen21_processor_uses_standard_hf_loader(tmp_path):
    from transformers import Qwen2VLImageProcessor, Qwen3VLConfig, Qwen3VLProcessor, Qwen3VLVideoProcessor
    from verl.utils import hf_processor

    from tests.special_e2e.build_qwen_image_tiny_random import _build_tiny_chatml_tokenizer

    tokenizer = _build_tiny_chatml_tokenizer()
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(tmp_path / "processor")
    Qwen3VLConfig().save_pretrained(tmp_path / "text_encoder")
    path = QwenImage21DiffusionNFT.prepare_processor_files(str(tmp_path))
    loaded = hf_processor(path, local_files_only=True)
    assert isinstance(loaded, Qwen3VLProcessor)
    assert loaded.config.model_type == "qwen3_vl"
    assert callable(loaded.get_rope_index)
    assert loaded.tokenizer.encode("draw a sign") == tokenizer.encode("draw a sign")
