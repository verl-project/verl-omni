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
"""2.1 的条件布局、目标裁剪与 CFG 合约。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines import QwenImage21DiffusionNFT, QwenImage21DiffusionNFTPipeline
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_21_diffusion_nft.common import align_prompt_tags, image_shapes, joint_image_mask
from verl_omni.pipelines.qwen_image_21_diffusion_nft.diffusers_training_adapter import (
    QwenImage21DiffusionNFT as Adapter,
)
from verl_omni.pipelines.qwen_image_21_diffusion_nft.prompt import prepare_reference_media, tokenize_prompt


def batch(nrefs, condition_shape=(2, 2), target_shape=(2, 2)):
    """使用不同形状的固定条件和目标值，暴露错误拼接或加噪。"""
    condition_tokens = condition_shape[0] * condition_shape[1]
    target_tokens = target_shape[0] * target_shape[1]
    conditions = torch.zeros(1, 4, 3, dtype=torch.long)
    conditions[:, :nrefs] = torch.tensor([1, *condition_shape])
    image_slots = nrefs * condition_tokens // 4
    length = 3 + image_slots
    tags = torch.zeros(1, length, dtype=torch.bool)
    tags[:, :image_slots] = True
    data = TensorDict(
        {
            "condition_image_shapes": conditions,
            "target_image_shape": torch.tensor([[1, *target_shape]]),
            "condition_image_rows": torch.full((1, nrefs * condition_tokens, 64), 17.0),
            "prompt_token_tags": tags,
            "negative_prompt_token_tags": tags,
            "latents_clean": torch.full((1, target_tokens, 64), 2.0),
            "forward_noise": torch.full((1, target_tokens, 64), 4.0),
            "train_timesteps": torch.tensor([[500.0]]),
        },
        batch_size=[1],
    )
    return data, torch.zeros(1, length, 8), torch.ones(1, length, dtype=torch.bool)


def test_public_package_registers_both_adapters():
    """通过公共包和 registry 查找，防止漏掉 __init__ 导入。"""
    assert DiffusionModelBase.get_class_by_name("QwenImage21Pipeline", "diffusion_nft") is QwenImage21DiffusionNFT
    assert VllmOmniPipelineBase.get_class("QwenImage21Pipeline", "diffusion_nft") is QwenImage21DiffusionNFTPipeline


@pytest.mark.parametrize("nrefs", range(5))
@pytest.mark.parametrize("condition_shape,target_shape", [((2, 2), (2, 2)), ((2, 4), (4, 2))])
def test_target_only_noising_and_tail_prediction(nrefs, condition_shape, target_shape):
    data, embeds, mask = batch(nrefs, condition_shape, target_shape)
    target_tokens = data["latents_clean"].shape[1]
    condition_tokens = data["condition_image_rows"].shape[1]
    scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    noisy, time, context = Adapter.prepare_training_inputs(data, 0, scheduler)
    torch.testing.assert_close(noisy, torch.full_like(noisy, 3.0))
    module = Mock(config=SimpleNamespace(in_channels=64))
    cfg = SimpleNamespace(pipeline=SimpleNamespace(true_cfg_scale=1.0))
    positive, negative = Adapter.prepare_model_inputs(module, cfg, noisy, time, embeds, mask, None, None, data, 0)
    assert negative is None and positive["kv_cache"] is None and positive["kv_cache_mode"] is None
    torch.testing.assert_close(positive["hidden_states"][:, :condition_tokens], data["condition_image_rows"])
    torch.testing.assert_close(positive["hidden_states"][:, -target_tokens:], noisy)
    torch.testing.assert_close(positive["timestep"], torch.tensor([0.5]))
    assert positive["img_shapes"] == [[(1, *condition_shape)] * nrefs + [(1, *target_shape)]]
    assert int(positive["img_mask"].sum()) == (condition_tokens + target_tokens) // 4
    module.return_value = (torch.cat([torch.full((1, condition_tokens, 64), -10.0), noisy], dim=1),)
    torch.testing.assert_close(Adapter.forward(module, cfg, positive), noisy)
    assert "_target_seq_len" in positive
    assert "_target_seq_len" not in module.call_args.kwargs


@pytest.mark.parametrize("nrefs", [0, 1, 4])
def test_cfg_preserves_conditions_and_has_no_norm_rescale(nrefs):
    """正负分支只改变文本条件，均先裁剪目标再应用标准 CFG。"""
    data, embeds, mask = batch(nrefs)
    cfg = SimpleNamespace(pipeline=SimpleNamespace(true_cfg_scale=2.0))
    module = Mock(config=SimpleNamespace(in_channels=64))
    positive, negative = Adapter.prepare_model_inputs(
        module, cfg, data["latents_clean"], data["train_timesteps"][:, 0], embeds, mask, embeds + 1, mask, data, 0
    )
    for key in ("hidden_states", "timestep", "img_mask"):
        torch.testing.assert_close(positive[key], negative[key], rtol=0, atol=0)
    assert positive["img_shapes"] == negative["img_shapes"]
    torch.testing.assert_close(negative["encoder_hidden_states"], embeds + 1)
    module.side_effect = [(torch.full((1, 4 * (nrefs + 1), 64), 3.0),), (torch.full((1, 4 * (nrefs + 1), 64), 1.0),)]
    result = Adapter.forward(module, cfg, positive, negative)
    torch.testing.assert_close(result, torch.full((1, 4, 64), 5.0))


@pytest.mark.parametrize("missing", ["embeddings", "mask"])
def test_cfg_requires_both_negative_inputs(missing):
    data, embeds, mask = batch(1)
    with pytest.raises(ValueError, match="negative prompt embeddings and masks"):
        Adapter.prepare_model_inputs(
            Mock(config=SimpleNamespace(in_channels=64)),
            SimpleNamespace(pipeline=SimpleNamespace(true_cfg_scale=2.0)),
            data["latents_clean"],
            data["train_timesteps"][:, 0],
            embeds,
            mask,
            None if missing == "embeddings" else embeds,
            None if missing == "mask" else mask,
            data,
            0,
        )


@pytest.mark.parametrize("rows", [0, 3, 8])
def test_condition_latents_must_match_spatial_metadata(rows):
    data, embeds, mask = batch(1)
    data["condition_image_rows"] = torch.zeros(1, rows, 64)
    with pytest.raises(ValueError, match="Condition latent length"):
        Adapter.prepare_model_inputs(
            Mock(config=SimpleNamespace(in_channels=64)),
            SimpleNamespace(pipeline=SimpleNamespace(true_cfg_scale=1.0)),
            data["latents_clean"],
            data["train_timesteps"][:, 0],
            embeds,
            mask,
            None,
            None,
            data,
            0,
        )


@pytest.mark.parametrize("strategy,sp", [("fsdp", 1), ("fsdp2", 2)])
def test_unsupported_engine_or_sequence_parallel_fails_early(strategy, sp):
    with pytest.raises(ValueError, match="requires FSDP2 with ulysses_sequence_parallel_size=1"):
        Adapter.validate_engine_config(SimpleNamespace(strategy=strategy, ulysses_sequence_parallel_size=sp))
    Adapter.validate_engine_config(SimpleNamespace(strategy="fsdp2", ulysses_sequence_parallel_size=1))


def test_prompt_rejects_more_than_four_images_before_processing():
    processor = Mock()
    with pytest.raises(ValueError, match="at most four"):
        tokenize_prompt([{"role": "user", "content": "edit"}], {"images": [object()] * 5}, processor)
    processor.assert_not_called()


def test_rollout_rejects_image_slots_without_reference_images():
    pipeline = object.__new__(QwenImage21DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline._drop_idx = 2
    pipeline._img_token_id = 42
    pipeline.processor = Mock(tokenizer=SimpleNamespace(pad_token_id=0))
    pipeline.text_encoder = Mock()
    with pytest.raises(ValueError, match="image slots but no condition images"):
        pipeline.encode_prompt([{"ids": [1, 2, 42, 3]}], [[]])
    pipeline.text_encoder.assert_not_called()


def test_missing_slots_and_padding_slots_are_rejected():
    data, _, mask = batch(1)
    shapes = image_shapes(data["condition_image_shapes"], data["target_image_shape"])
    tags = torch.zeros_like(data["prompt_token_tags"])
    with pytest.raises(ValueError, match="slots do not match"):
        joint_image_mask(tags, mask, shapes, 4)
    tags[:, -1] = True
    mask[:, -1] = False
    with pytest.raises(ValueError, match="padding"):
        joint_image_mask(tags, mask, shapes, 4)


def test_noncontiguous_reference_slots_rejected():
    data, _, _ = batch(2)
    data["condition_image_shapes"][:, 0] = 0
    with pytest.raises(ValueError, match="contiguous"):
        image_shapes(data["condition_image_shapes"], data["target_image_shape"])


def test_unpadding_preserves_image_slots():
    tags = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool)
    mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(align_prompt_tags(tags, mask), tags[:, :3])
    tags[:, -1] = True
    with pytest.raises(ValueError, match="removed"):
        align_prompt_tags(tags, mask)


@pytest.mark.parametrize("nrefs", range(5))
def test_agent_formats_upstream_prompt_and_tokenizes_once(nrefs):
    from PIL import Image

    images = [Image.new("RGBA", (96 + 32 * i, 64), (40, 80, 120, 90)) for i in range(nrefs)]
    processor = Mock(
        return_value=SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long)
        )
    )
    media = prepare_reference_media({"images": images})
    ids = tokenize_prompt([{"role": "user", "content": "draw a cat"}], media, processor)
    assert ids == [1, 2, 3] and processor.call_count == 1
    text = processor.call_args.kwargs["text"][0]
    assert text.startswith("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n")
    assert text.endswith("draw a cat<|im_end|>\n<|im_start|>assistant\n")
    assert text.count("<|image_pad|>") == nrefs
    assert len(media["images"]) == nrefs
    if nrefs:
        assert [image.size for image in processor.call_args.kwargs["images"]] == [
            image.size for image in media["images"]
        ]
        assert all(image.mode == "RGB" for image in processor.call_args.kwargs["images"])
        assert all(image.mode == "RGBA" for image in media["images"])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rollout_uses_token_ids_without_tokenizing_again(batch_size):
    from PIL import Image

    from verl_omni.pipelines.qwen_image_21_diffusion_nft.vllm_omni_rollout_adapter import (
        QwenImage21DiffusionNFTPipeline,
    )

    class Encoder(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.LayerNorm(4)))
            self.seen = None

        def forward(self, **kwargs):
            self.seen = kwargs
            raw = kwargs["input_ids"].float().unsqueeze(-1).expand(-1, -1, 4)
            return SimpleNamespace(hidden_states=[self.model.language_model.norm(raw)])

    pipeline = object.__new__(QwenImage21DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 16
    pipeline._drop_idx = 2
    pipeline._img_token_id = 42
    pipeline.text_encoder = Encoder()
    pipeline.processor = Mock(tokenizer=SimpleNamespace(pad_token_id=0))
    pipeline.processor.image_processor.merge_size = 2
    pipeline.processor.image_processor.return_value = {
        "pixel_values": torch.ones(4 * batch_size, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]] * (2 * batch_size)),
    }
    prompts = [{"ids": [1, 2, 42, 3, 42, 4]}]
    images = [[Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]]
    expected_embeds = [[42.0, 3.0, 42.0, 4.0]]
    expected_tags = [[True, False, True, False]]
    expected_masks = [[1, 1, 1, 1]]
    expected_ids = [[1, 2, 42, 3, 42, 4]]
    expected_token_types = [[0, 0, 1, 0, 1, 0]]
    if batch_size == 2:
        prompts.append({"ids": [1, 2, 42, 3, 42]})
        images.append(images[0])
        expected_embeds.append([42.0, 3.0, 42.0, 0.0])
        expected_tags.append([True, False, True, False])
        expected_masks.append([1, 1, 1, 0])
        expected_ids.append([0, 1, 2, 42, 3, 42])
        expected_token_types.append([0, 0, 0, 1, 0, 1])
    embeds, mask, tags = pipeline.encode_prompt(prompts, images, num_images_per_prompt=2)
    assert not pipeline.processor.called
    pipeline.processor.image_processor.assert_called_once()
    torch.testing.assert_close(embeds[:, :, 0], torch.tensor(expected_embeds).repeat_interleave(2, dim=0))
    torch.testing.assert_close(tags, torch.tensor(expected_tags).repeat_interleave(2, dim=0))
    torch.testing.assert_close(mask, torch.tensor(expected_masks).repeat_interleave(2, dim=0))
    assert not pipeline.text_encoder.model.language_model.norm._forward_hooks
    torch.testing.assert_close(pipeline.text_encoder.seen["input_ids"], torch.tensor(expected_ids))
    torch.testing.assert_close(pipeline.text_encoder.seen["mm_token_type_ids"], torch.tensor(expected_token_types))


@pytest.mark.parametrize("reference_counts", [(1, 1), (0, 2), (2, 0)])
def test_rollout_rejects_cross_request_image_slot_mismatch(reference_counts):
    """整批 slot 总数正确也不能把第二条请求的图像特征填进第一条。"""
    from PIL import Image

    pipeline = object.__new__(QwenImage21DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline._drop_idx = 2
    pipeline._img_token_id = 42
    pipeline.text_encoder = Mock(dtype=torch.float32)
    pipeline.processor = Mock(tokenizer=SimpleNamespace(pad_token_id=0))
    pipeline.processor.image_processor.merge_size = 2
    pipeline.processor.image_processor.return_value = {
        "pixel_values": torch.ones(8, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
    }
    slot_counts = (2, 0) if reference_counts == (1, 1) else reference_counts[::-1]
    prompts = [{"ids": [1, 2, *([42] * count), 3]} for count in slot_counts]
    images = [[Image.new("RGB", (32, 32)) for _ in range(count)] for count in reference_counts]
    with pytest.raises(ValueError, match="processor grid for each request"):
        pipeline.encode_prompt(prompts, images)
    pipeline.text_encoder.assert_not_called()


@pytest.mark.parametrize("rows", [0, 16, 64])
def test_condition_rows_survive_padding_and_nested_transport(rows):
    from verl_omni.agent_loop.diffusion_agent_loop import _pad_reference_rows
    from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
    from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

    original = torch.arange(rows * 64, dtype=torch.float32).reshape(1, rows, 64)
    padded, masks = _pad_reference_rows("condition_image_rows", [original], [torch.tensor([[rows]])], target_length=128)
    data = TensorDict({"condition_image_rows": padded[0], "condition_image_rows_mask": masks[0]}, batch_size=[1])
    data = embeds_padding_2_no_padding(data)
    restored, mask = DiffusersFSDPEngine._unpad_nested_embeds(
        data["condition_image_rows"], data["condition_image_rows_mask"]
    )
    torch.testing.assert_close(restored, original, atol=0, rtol=0)
    assert mask.shape == (1, rows) and mask.all()


@pytest.mark.parametrize("nrefs", [0, 1, 4])
def test_rollout_splits_training_metadata_with_its_request(nrefs):
    """两个请求各生成两个样本时，latent、embedding 和条件不能串到另一请求。"""
    from vllm_omni.diffusion.data import DiffusionOutput

    from verl_omni.pipelines.qwen_image_21_diffusion_nft.vllm_omni_rollout_adapter import (
        QwenImage21DiffusionNFTPipeline,
    )

    pipeline = object.__new__(QwenImage21DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 16
    pipeline._attention_kwargs = {"attn_path": "mask"}
    values = torch.arange(4, dtype=torch.float32).view(4, 1, 1)
    latents = values.expand(4, 4, 64).clone()
    context = {
        "prompt_embeds": values.expand(4, 3 + nrefs, 8).clone(),
        "prompt_embeds_mask": torch.ones(4, 3 + nrefs, dtype=torch.bool),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        "image_pad_mask": torch.zeros(4, 3 + nrefs, dtype=torch.bool),
        "negative_image_pad_mask": None,
        "img_mask": torch.ones(4, 4 + nrefs, dtype=torch.bool),
        "negative_img_mask": None,
        "latents": latents,
        "image_latents": values.expand(4, 4 * nrefs, 64).clone() if nrefs else None,
        "img_shapes": [[(1, 2, 2)] * (1 + nrefs) for _ in range(2)],
        "timesteps": torch.tensor([500.0]),
        "do_true_cfg": False,
    }
    pipeline._prepare_generation_context = Mock(return_value=context)
    pipeline.diffuse = Mock(return_value=latents)
    pipeline._decode_latents = Mock(return_value=DiffusionOutput(output=latents))
    sampling = SimpleNamespace(
        true_cfg_scale=1.0,
        height=32,
        width=32,
        num_outputs_per_prompt=2,
        num_inference_steps=1,
        sigmas=None,
        max_sequence_length=8,
        output_type="latent",
        extra_args={},
    )
    req = SimpleNamespace(
        prompts=[{"prompt_ids": [1, 2]}, {"prompt_ids": [3, 4]}],
        sampling_params_list=[sampling, sampling],
        num_reqs=2,
        request_ids=["first", "second"],
        is_dummy_run=lambda: False,
        collate_request_generators=lambda *args: None,
        collate_request_tensors=lambda *args: None,
    )
    outputs = pipeline.forward(req)
    assert len(outputs) == 2
    assert pipeline.diffuse.call_args.kwargs["additional_transformer_kwargs"]["kv_cache"] is None
    for index, output in enumerate(outputs):
        metadata = output.output["metadata"]
        rows = slice(index * 2, index * 2 + 2)
        torch.testing.assert_close(metadata["rl"]["latents_clean"], latents[rows], atol=0, rtol=0)
        torch.testing.assert_close(metadata["prompt_embeddings"]["prompt_embeds"], context["prompt_embeds"][rows])
        assert metadata["rl"]["condition_image_rows"].shape == (2, nrefs * 4, 64)
        assert metadata["rl"]["condition_image_row_count"].tolist() == [[nrefs * 4], [nrefs * 4]]
        assert metadata["rl"]["condition_image_shapes"][:, :nrefs].tolist() == [[[1, 2, 2]] * nrefs] * 2
