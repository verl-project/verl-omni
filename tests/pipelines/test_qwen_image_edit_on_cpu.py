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
import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    VAE_IMAGE_SIZE,
    calculate_dimensions,
)

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_edit_flow_grpo.diffusers_training_adapter import QwenImageEditPlusFlowGRPO
from verl_omni.pipelines.qwen_image_edit_flow_grpo.vllm_omni_rollout_adapter import (
    QwenImageEditPlusPipelineWithLogProb,
    _condition_images_for_prompt_encoding,
    _select_qwen_image_edit_sde_window,
    _use_true_cfg,
    _validate_condition_image_sizes,
)
from verl_omni.pipelines.request_batch import shared_sde_window_generator
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


def _model_config() -> DiffusionModelConfig:
    config = object.__new__(DiffusionModelConfig)
    object.__setattr__(config, "architecture", "QwenImageEditPlusPipeline")
    object.__setattr__(config, "external_lib", None)
    object.__setattr__(config, "algorithm", "flow_grpo")
    return config


def _non_tensor_stack(values):
    return NonTensorStack.from_list([NonTensorData(value) for value in values])


def test_processor_hook_creates_missing_config(tmp_path):
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()

    prepared_dir = QwenImageEditPlusFlowGRPO.prepare_processor_files(str(tmp_path))

    assert prepared_dir == str(processor_dir)
    assert json.loads((Path(prepared_dir) / "config.json").read_text(encoding="utf-8")) == {"model_type": "qwen2_vl"}


def test_processor_hook_preserves_existing_config(tmp_path):
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()
    config_path = processor_dir / "config.json"
    config_path.write_text('{"model_type": "custom"}', encoding="utf-8")

    prepared_dir = QwenImageEditPlusFlowGRPO.prepare_processor_files(str(tmp_path))

    assert prepared_dir == str(processor_dir)
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"model_type": "custom"}


def test_get_class_resolves_qwen_image_adapter():
    assert DiffusionModelBase.get_class(_model_config()) is QwenImageEditPlusFlowGRPO


def test_prepare_condition_unwraps_metadata():
    image_shapes = [[(1, 16, 16), (1, 8, 8)], [(1, 16, 16), (1, 8, 8)]]
    micro_batch = TensorDict(
        {
            "condition_image_latents": torch.zeros(2, 64, 8),
            "img_shapes": _non_tensor_stack(image_shapes),
            "sp_size": NonTensorData(1),
        },
        batch_size=[2],
    )

    condition = QwenImageEditPlusFlowGRPO.prepare_condition(
        micro_batch,
        latents=torch.zeros(2, 1, 64, 8),
        step=0,
    )

    assert condition["image_latents"].shape == (2, 64, 8)
    assert condition["img_shapes"] == image_shapes
    assert condition["sp_size"] == 1


def test_forward_crops_condition_predictions_through_qwen_mro():
    class _EchoModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_kwargs = None

        def forward(self, **kwargs):
            self.last_kwargs = kwargs
            return (kwargs["hidden_states"],)

    module = _EchoModule()
    prediction = QwenImageEditPlusFlowGRPO.forward(
        module,
        _model_config(),
        {
            "hidden_states": torch.zeros(1, 5, 4),
            "_target_seq_len": 2,
        },
    )

    assert prediction.shape == (1, 2, 4)
    assert "_target_seq_len" not in module.last_kwargs


def test_prepare_condition_rejects_reserved_rollout_key():
    micro_batch = TensorDict(
        {"image_latents": torch.zeros(1, 64, 8)},
        batch_size=[1],
    )

    with pytest.raises(ValueError, match="wrong key"):
        QwenImageEditPlusFlowGRPO.prepare_condition(
            micro_batch,
            latents=torch.zeros(1, 1, 64, 8),
            step=0,
        )


def test_inject_condition_updates_qwen_image_shapes():
    model_inputs = {"hidden_states": torch.zeros(1, 2, 4), "img_shapes": [[(1, 1, 2)]]}
    negative_inputs = {"hidden_states": torch.zeros(1, 2, 4), "img_shapes": [[(1, 1, 2)]]}
    image_shapes = [[(1, 1, 2), (1, 1, 3)]]

    output, negative_output = QwenImageEditPlusFlowGRPO.inject_condition(
        model_inputs,
        negative_inputs,
        {"image_latents": torch.ones(1, 3, 4), "img_shapes": image_shapes},
    )

    assert output["img_shapes"] == image_shapes
    assert negative_output["img_shapes"] == image_shapes
    assert output["hidden_states"].shape == (1, 5, 4)
    assert output["hidden_states"].shape[1] == sum(math.prod(shape) for shape in output["img_shapes"][0])


def test_inject_condition_validates_qwen_sequence_parallel_alignment():
    model_inputs = {"hidden_states": torch.zeros(1, 2, 4)}
    condition = {
        "image_latents": torch.zeros(1, 3, 4),
        "sp_size": 2,
    }

    with pytest.raises(ValueError, match="sequence-parallel size"):
        QwenImageEditPlusFlowGRPO.inject_condition(model_inputs, None, condition)


def test_non_tensor_sp_size_preserves_sequence_parallel_validation():
    micro_batch = TensorDict(
        {
            "condition_image_latents": torch.zeros(1, 3, 4),
            "sp_size": NonTensorData(2),
        },
        batch_size=[1],
    )
    condition = QwenImageEditPlusFlowGRPO.prepare_condition(
        micro_batch,
        latents=torch.zeros(1, 1, 4, 4),
        step=0,
    )

    with pytest.raises(ValueError, match="sequence-parallel size"):
        QwenImageEditPlusFlowGRPO.inject_condition({"hidden_states": torch.zeros(1, 2, 4)}, None, condition)


def test_true_cfg_requires_negative_prompt_inputs():
    assert not _use_true_cfg(1.0, None, None, None)
    assert _use_true_cfg(4.0, [1], None, None)
    with pytest.raises(ValueError, match="requires negative_prompt_ids"):
        _use_true_cfg(4.0, None, None, None)


def test_prompt_encoding_requires_condition_images():
    with pytest.raises(AssertionError, match="requires condition images"):
        QwenImageEditPlusPipelineWithLogProb._get_qwen_prompt_embeds(
            None,
            torch.tensor([1]),
            condition_images=None,
        )


def test_prompt_encoding_prefers_raw_image_over_preprocessed_alias():
    raw_image = torch.zeros(3, 8, 8)
    resized_image = torch.ones(3, 4, 4)
    payload = {
        "multi_modal_data": {"image": [raw_image]},
        "additional_information": {"condition_images": [resized_image]},
    }

    result = _condition_images_for_prompt_encoding(payload)
    assert len(result) == 1
    assert result[0] is raw_image


def test_prompt_encoding_falls_back_to_preprocessed_image():
    resized_image = torch.ones(3, 4, 4)
    payload = {"additional_information": {"condition_images": [resized_image]}}

    result = _condition_images_for_prompt_encoding(payload)
    assert len(result) == 1
    assert result[0] is resized_image


@pytest.mark.parametrize("as_list", [False, True])
def test_raw_and_engine_preprocessed_images_are_distinct_named_views(monkeypatch, as_list):
    from types import SimpleNamespace

    from PIL import Image

    from verl_omni.pipelines.qwen_image_edit_flow_grpo import vllm_omni_rollout_adapter as adapter

    class Captured(Exception):
        pass

    raw = Image.new("RGB", (32, 32), "blue")
    processed = raw.resize((384, 384))
    prompt = {
        "prompt_ids": [1, 2],
        "multi_modal_data": {"image": [raw] if as_list else raw},
        "additional_information": {"condition_images": [processed], "vae_image_sizes": [(1024, 1024)]},
    }
    request = SimpleNamespace(
        prompts=[prompt], sampling_params=SimpleNamespace(height=64, width=64, output_type="image")
    )

    def capture(images, sizes, **kwargs):
        assert images == [raw]
        assert images[0].size == (32, 32)
        assert sizes == [(1024, 1024)]
        raise Captured

    monkeypatch.setattr(adapter, "_validate_condition_image_sizes", capture)
    with pytest.raises(Captured):
        QwenImageEditPlusPipelineWithLogProb.forward(SimpleNamespace(), request)


def test_condition_images_require_fixed_square_latents():
    _validate_condition_image_sizes(["image"], [(1024, 1024)])
    with pytest.raises(ValueError, match="vae_image_sizes"):
        _validate_condition_image_sizes(["image"], None)
    with pytest.raises(ValueError, match="exactly one condition image"):
        _validate_condition_image_sizes(["image", "image"], [(1024, 1024), (1024, 1024)])
    with pytest.raises(ValueError, match="vae_image_sizes"):
        _validate_condition_image_sizes(["image"], [(1024, 1024), (1024, 1024)])
    with pytest.raises(ValueError, match="square condition images"):
        _validate_condition_image_sizes(["image"], [(1344, 768)])


def test_condition_images_allow_fixed_nonsquare_target():
    # (height, width) targets for 16:9 and 4:3 aspect ratios.
    for target in [(720, 1280), (768, 1024)]:
        expected = calculate_dimensions(VAE_IMAGE_SIZE, target[1] / target[0])
        # A condition image whose VAE size matches the target aspect ratio is accepted.
        _validate_condition_image_sizes(["image"], [expected], target_size=target)

    # A square condition under a non-square target is rejected early with a
    # clear message instead of failing later in the cross-sample concat.
    with pytest.raises(ValueError, match="target aspect ratio"):
        _validate_condition_image_sizes(["image"], [(1024, 1024)], target_size=(720, 1280))

    # A square target still requires square conditions (backward compatible).
    _validate_condition_image_sizes(["image"], [(1024, 1024)], target_size=(512, 512))


def _data_parallel_rank(global_rank: int, groups: list[list[int]]) -> int:
    for group in groups:
        if global_rank in group:
            return group.index(global_rank)
    raise AssertionError(global_rank)


def test_qwen_image_edit_window_matches_across_tp_ranks(monkeypatch):
    from vllm_omni.diffusion.distributed.parallel_state import RankGenerator

    import verl_omni.pipelines.request_batch as request_batch

    # TP=2, DP=1 is one replica: both process ranks share data-parallel rank 0.
    single = RankGenerator(2, 1, 1, 1, 1)
    assert single.get_ranks("tp") == [[0, 1]]
    assert [_data_parallel_rank(rank, single.get_ranks("dp")) for rank in (0, 1)] == [0, 0]
    # TP=2, DP=2: each tensor-parallel pair shares one data-parallel rank.
    paired = RankGenerator(2, 1, 1, 1, 2)
    assert paired.get_ranks("tp") == [[0, 1], [2, 3]]
    paired_dp = paired.get_ranks("dp")
    assert [_data_parallel_rank(rank, paired_dp) for rank in (0, 1, 2, 3)] == [0, 0, 1, 1]

    extra = {"sde_window_seed": 0}
    kwargs = {"device": "cpu", "sde_window_size": 3, "sde_window_range": (0, 6), "num_timesteps": 6}

    def process_rank_window(process_rank: str) -> tuple[int, int]:
        monkeypatch.setenv("RANK", process_rank)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        generator = shared_sde_window_generator(extra, torch.Generator(), device="cpu", policy="rank")
        start = int(torch.randint(0, 4, (1,), generator=generator).item())
        return (start, start + 3)

    assert process_rank_window("0") == (0, 3)
    assert process_rank_window("1") == (1, 4)

    def replica_window(process_rank: str, data_parallel_rank: int) -> tuple[int, int]:
        monkeypatch.setenv("RANK", process_rank)
        monkeypatch.setattr(request_batch, "_data_parallel_rank", lambda: data_parallel_rank)
        return _select_qwen_image_edit_sde_window(extra, torch.Generator(), **kwargs)

    assert replica_window("0", 0) == replica_window("1", 0) == (0, 3)
    assert replica_window("2", 1) == replica_window("3", 1) == (1, 4)

    monkeypatch.undo()
    monkeypatch.setenv("RANK", "1")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert _select_qwen_image_edit_sde_window(extra, torch.Generator(), **kwargs) == (0, 3)
    pinned_start = int(torch.randint(0, 4, (1,), generator=torch.Generator().manual_seed(99)).item())
    pinned = _select_qwen_image_edit_sde_window({"sde_window_seed": 99}, torch.Generator(), **kwargs)
    assert pinned == (pinned_start, pinned_start + 3)
    assert _select_qwen_image_edit_sde_window(
        extra, torch.Generator(), device="cpu", sde_window_size=None, sde_window_range=(0, 6), num_timesteps=10
    ) == (0, 9)
    with pytest.raises(ValueError, match="target aspect ratio"):
        _validate_condition_image_sizes(["image"], [(1344, 768)], target_size=(512, 512))
