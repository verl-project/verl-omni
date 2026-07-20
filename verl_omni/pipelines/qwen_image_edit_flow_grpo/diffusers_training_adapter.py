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

"""Qwen-Image-Edit-Plus I2I training adapter for diffusion RL."""

import json
from pathlib import Path
from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage

__all__ = ["QwenImageEditPlusFlowGRPO"]


@DiffusionModelBase.register("QwenImageEditPlusPipeline", algorithm="flow_grpo")
class QwenImageEditPlusFlowGRPO(DiffusionI2IModelBase, QwenImage):
    """Training adapter for Qwen-Image-Edit-Plus image editing.

    Reuses Qwen-Image's T2I input construction and sampling logic, then
    injects image-condition latents into the transformer input.
    """

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> Optional[str]:
        """Create the processor config omitted by Qwen-Image-Edit checkpoints."""
        processor_dir = Path(model_path) / "processor"
        if not processor_dir.is_dir():
            raise FileNotFoundError(f"Qwen-Image-Edit processor directory not found: {processor_dir}")
        config_path = processor_dir / "config.json"
        if not config_path.is_file():
            config_path.write_text(json.dumps({"model_type": "qwen2_vl"}), encoding="utf-8")
        return str(processor_dir)

    @classmethod
    def prepare_condition(
        cls,
        micro_batch: TensorDict,
        latents: torch.Tensor,
        step: int,
    ) -> Optional[dict]:
        del latents, step
        image_latents = micro_batch.get("condition_image_latents", None)
        if image_latents is None:
            # Detect wrong key name: rollout output "image_latents" instead of "condition_image_latents".
            if "image_latents" in micro_batch:
                raise ValueError(
                    "QwenImageEditPlusFlowGRPO.prepare_condition: "
                    "micro_batch has 'image_latents' but not 'condition_image_latents'. "
                    "The rollout adapter likely output the wrong key. "
                    "Use 'condition_image_latents' in custom_output to avoid "
                    "colliding with the MFU FLOPs counter."
                )
            return None
        return {
            "image_latents": image_latents,
            "img_shapes": tu.get(micro_batch, "img_shapes"),
            "sp_size": tu.get(micro_batch, "sp_size"),
        }

    @classmethod
    def inject_condition(
        cls,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        condition: Optional[dict],
    ) -> tuple[dict, Optional[dict]]:
        if condition and condition.get("image_latents") is not None:
            target_seq_len = model_inputs["hidden_states"].shape[1]
            condition_seq_len = condition["image_latents"].shape[1]
            sp_size = condition.get("sp_size")
            if isinstance(sp_size, int) and sp_size > 1 and (target_seq_len + condition_seq_len) % sp_size:
                raise ValueError(
                    "Qwen-Image-Edit target and condition token lengths must be divisible by "
                    f"the sequence-parallel size: ({target_seq_len} + {condition_seq_len}) % {sp_size} != 0."
                )
        model_inputs, negative_model_inputs = super().inject_condition(model_inputs, negative_model_inputs, condition)
        if not condition:
            return model_inputs, negative_model_inputs
        # Replace img_shapes with condition-aware shapes (target + condition) for 2D RoPE.
        img_shapes = condition.get("img_shapes")
        if img_shapes is not None:
            model_inputs["img_shapes"] = img_shapes
            if negative_model_inputs is not None:
                negative_model_inputs["img_shapes"] = img_shapes
        return model_inputs, negative_model_inputs
