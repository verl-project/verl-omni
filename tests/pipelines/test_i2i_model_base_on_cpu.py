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
"""CPU tests for the generic I2I diffusion model hooks."""

import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase
from verl_omni.pipelines.utils import prepare_model_inputs
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


def _model_config(architecture: str) -> DiffusionModelConfig:
    config = object.__new__(DiffusionModelConfig)
    object.__setattr__(config, "architecture", architecture)
    object.__setattr__(config, "external_lib", None)
    object.__setattr__(config, "algorithm", "flow_grpo")
    return config


def _non_tensor_stack(values):
    return NonTensorStack.from_list([NonTensorData(value) for value in values])


class _EchoModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return (kwargs["hidden_states"],)


class TestDiffusionI2IModelBase:
    def test_forward_slices_condition_predictions(self):
        module = _EchoModule()
        hidden_states = torch.arange(12, dtype=torch.float32).view(1, 3, 4)

        prediction = DiffusionModelBase.forward(
            module,
            model_config=None,
            model_inputs={"hidden_states": hidden_states, "_target_seq_len": 2},
        )

        assert prediction.shape == (1, 2, 4)
        assert "_target_seq_len" not in module.last_kwargs

    def test_inject_condition_is_noop_without_condition(self):
        model_inputs = {"hidden_states": torch.zeros(1, 2, 4)}

        output, negative_output = DiffusionI2IModelBase.inject_condition(model_inputs, None, None)

        assert output is model_inputs
        assert negative_output is None

    def test_inject_condition_concatenates_positive_and_negative_inputs(self):
        model_inputs = {"hidden_states": torch.zeros(1, 2, 4)}
        negative_inputs = {"hidden_states": torch.ones(1, 2, 4)}
        condition = {"image_latents": torch.full((1, 3, 4), 2.0)}

        output, negative_output = DiffusionI2IModelBase.inject_condition(model_inputs, negative_inputs, condition)

        assert output["hidden_states"].shape == (1, 5, 4)
        assert negative_output["hidden_states"].shape == (1, 5, 4)
        assert output["_target_seq_len"] == 2
        assert negative_output["_target_seq_len"] == 2

    def test_inject_condition_rejects_mismatched_batch_size(self):
        model_inputs = {"hidden_states": torch.zeros(2, 2, 4)}
        condition = {"image_latents": torch.zeros(1, 3, 4)}

        with pytest.raises(ValueError, match="batch size"):
            DiffusionI2IModelBase.inject_condition(model_inputs, None, condition)

    def test_inject_condition_rejects_non_tensor_sequence(self):
        model_inputs = {"hidden_states": torch.zeros(1, 2, 4)}
        condition = {"image_latents": torch.zeros(1, 2, 2, 4)}

        with pytest.raises(ValueError, match="must be 3-D"):
            DiffusionI2IModelBase.inject_condition(model_inputs, None, condition)

    def test_inject_condition_rejects_missing_latents(self):
        model_inputs = {"hidden_states": torch.zeros(1, 2, 4)}

        with pytest.raises(ValueError, match=r"requires condition\['image_latents'\]"):
            DiffusionI2IModelBase.inject_condition(model_inputs, None, {"img_shapes": [[(1, 2, 2)]]})

    def test_inject_condition_validates_sequence_parallel_alignment(self):
        model_inputs = {"hidden_states": torch.zeros(1, 2, 4)}

        with pytest.raises(ValueError, match="sequence-parallel size"):
            DiffusionI2IModelBase.inject_condition(
                model_inputs,
                None,
                {"image_latents": torch.zeros(1, 3, 4), "sp_size": 2},
            )

    def test_metadata_helpers_unwrap_non_tensor_values(self):
        image_shapes = [[(1, 16, 16)], [(1, 16, 16)]]
        micro_batch = TensorDict(
            {
                "img_shapes": _non_tensor_stack(image_shapes),
                "sp_size": _non_tensor_stack([2, 2]),
            },
            batch_size=[2],
        )

        assert DiffusionI2IModelBase.get_i2i_metadata(micro_batch, "img_shapes") == image_shapes
        assert DiffusionI2IModelBase.get_i2i_scalar_metadata(micro_batch, "sp_size") == 2

    def test_scalar_metadata_rejects_mixed_micro_batch_values(self):
        micro_batch = TensorDict(
            {"sp_size": _non_tensor_stack([1, 2])},
            batch_size=[2],
        )

        with pytest.raises(ValueError, match="differs across the micro-batch"):
            DiffusionI2IModelBase.get_i2i_scalar_metadata(micro_batch, "sp_size")


class TestPrepareModelInputsConditionDispatch:
    def test_t2i_adapter_keeps_existing_dispatch_behavior(self):
        @DiffusionModelBase.register("_GeneralInterfaceT2I", algorithm="flow_grpo")
        class _T2I(DiffusionModelBase):
            @classmethod
            def build_scheduler(cls, model_config):
                pass

            @classmethod
            def set_timesteps(cls, scheduler, model_config, device):
                pass

            @classmethod
            def prepare_model_inputs(cls, *args, **kwargs):
                return {"hidden_states": torch.zeros(1, 2, 4)}, None

            @classmethod
            def forward_and_sample_previous_step(cls, *args, **kwargs):
                pass

        model_inputs, negative_model_inputs = prepare_model_inputs(
            module=None,
            model_config=_model_config("_GeneralInterfaceT2I"),
            latents=torch.zeros(1, 2, 4),
            timesteps=torch.zeros(1),
            prompt_embeds=torch.zeros(1, 1, 4),
            prompt_embeds_mask=None,
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch={},
            step=0,
        )

        assert model_inputs["hidden_states"].shape == (1, 2, 4)
        assert negative_model_inputs is None

    def test_i2i_adapter_injects_condition_after_base_inputs(self):
        @DiffusionModelBase.register("_GeneralInterfaceI2I", algorithm="flow_grpo")
        class _I2I(DiffusionI2IModelBase):
            @classmethod
            def build_scheduler(cls, model_config):
                pass

            @classmethod
            def set_timesteps(cls, scheduler, model_config, device):
                pass

            @classmethod
            def prepare_model_inputs(cls, *args, **kwargs):
                return {"hidden_states": torch.zeros(1, 2, 4)}, None

            @classmethod
            def forward_and_sample_previous_step(cls, *args, **kwargs):
                pass

            @classmethod
            def prepare_condition(cls, micro_batch, latents, step):
                return {"image_latents": micro_batch["condition_image_latents"]}

        model_inputs, negative_model_inputs = prepare_model_inputs(
            module=None,
            model_config=_model_config("_GeneralInterfaceI2I"),
            latents=torch.zeros(1, 2, 4),
            timesteps=torch.zeros(1),
            prompt_embeds=torch.zeros(1, 1, 4),
            prompt_embeds_mask=None,
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch={"condition_image_latents": torch.ones(1, 3, 4)},
            step=0,
        )

        assert model_inputs["hidden_states"].shape == (1, 5, 4)
        assert model_inputs["_target_seq_len"] == 2
        assert negative_model_inputs is None

    def test_i2i_adapter_fails_when_condition_is_missing(self):
        @DiffusionModelBase.register("_GeneralInterfaceMissingI2I", algorithm="flow_grpo")
        class _MissingI2I(DiffusionI2IModelBase):
            @classmethod
            def build_scheduler(cls, model_config):
                pass

            @classmethod
            def set_timesteps(cls, scheduler, model_config, device):
                pass

            @classmethod
            def prepare_model_inputs(cls, *args, **kwargs):
                return {"hidden_states": torch.zeros(1, 2, 4)}, None

            @classmethod
            def forward_and_sample_previous_step(cls, *args, **kwargs):
                pass

        with pytest.raises(ValueError, match="prepare_condition returned None"):
            prepare_model_inputs(
                module=None,
                model_config=_model_config("_GeneralInterfaceMissingI2I"),
                latents=torch.zeros(1, 2, 4),
                timesteps=torch.zeros(1),
                prompt_embeds=torch.zeros(1, 1, 4),
                prompt_embeds_mask=None,
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch={},
                step=0,
            )
