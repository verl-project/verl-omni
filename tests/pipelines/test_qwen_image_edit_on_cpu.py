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
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    VAE_IMAGE_SIZE,
    calculate_dimensions,
)

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_edit_flow_grpo.diffusers_training_adapter import (
    QwenImageEditPlusFlowGRPO,
    _processor_cache_key,
)
from verl_omni.pipelines.qwen_image_edit_flow_grpo.vllm_omni_rollout_adapter import (
    _use_true_cfg,
    _validate_condition_image_sizes,
)
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
    cache_dir = tmp_path / "cache"

    with patch.dict("os.environ", {"VERL_OMNI_PROCESSOR_CACHE": str(cache_dir)}):
        prepared_dir = QwenImageEditPlusFlowGRPO.prepare_processor_files(str(tmp_path))

    assert not (processor_dir / "config.json").exists()
    assert json.loads((Path(prepared_dir) / "config.json").read_text()) == {"model_type": "qwen2_vl"}


def test_processor_hook_accepts_read_only_model_directory(tmp_path):
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()
    (processor_dir / "preprocessor_config.json").write_text("{}")
    processor_dir.chmod(0o555)
    cache_dir = tmp_path / "cache"
    try:
        with patch.dict("os.environ", {"VERL_OMNI_PROCESSOR_CACHE": str(cache_dir)}):
            prepared_dir = QwenImageEditPlusFlowGRPO.prepare_processor_files(str(tmp_path))
        assert (Path(prepared_dir) / "config.json").is_file()
        assert stat.S_IMODE(Path(prepared_dir).stat().st_mode) & stat.S_IWUSR
    finally:
        processor_dir.chmod(0o755)


def test_processor_cache_key_changes_when_local_files_change(tmp_path):
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()
    config_path = processor_dir / "preprocessor_config.json"
    config_path.write_text("{}")
    original_key = _processor_cache_key(processor_dir)

    config_path.write_text('{"updated": true}')

    assert _processor_cache_key(processor_dir) != original_key


def test_get_class_applies_qwen_ulysses_patch():
    with patch("verl_omni.models.diffusers.qwen_image.apply_qwen_image_ulysses_mask_fix") as apply_patch:
        assert DiffusionModelBase.get_class(_model_config()) is QwenImageEditPlusFlowGRPO

    apply_patch.assert_called_once_with()


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


def test_condition_images_require_fixed_square_latents():
    _validate_condition_image_sizes(["image"], [(1024, 1024)])
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
    with pytest.raises(ValueError, match="target aspect ratio"):
        _validate_condition_image_sizes(["image"], [(1344, 768)], target_size=(512, 512))
