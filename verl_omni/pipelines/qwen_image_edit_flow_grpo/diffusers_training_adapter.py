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

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage

__all__ = ["QwenImageEditPlusFlowGRPO"]


def _processor_cache_key(processor_dir: Path) -> str:
    digest = hashlib.sha256(str(processor_dir.resolve()).encode())
    for path in sorted(processor_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(str(path.relative_to(processor_dir)).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


@DiffusionModelBase.register("QwenImageEditPlusPipeline", algorithm="flow_grpo")
class QwenImageEditPlusFlowGRPO(DiffusionI2IModelBase, QwenImage):
    """Training adapter for Qwen-Image-Edit-Plus image editing.

    Reuses Qwen-Image's T2I input construction and sampling logic, then
    injects image-condition latents into the transformer input.
    """

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> Optional[str]:
        """Return a writable processor copy with a minimal ``config.json``.

        Qwen-Image-Edit checkpoints can omit this file. The prepared copy keeps
        Hugging Face snapshots and shared model mounts unchanged.
        """
        processor_dir = Path(model_path) / "processor"
        if not processor_dir.is_dir():
            raise FileNotFoundError(f"Qwen-Image-Edit processor directory not found: {processor_dir}")
        if (processor_dir / "config.json").is_file():
            return str(processor_dir)

        cache_root = Path(
            os.environ.get(
                "VERL_OMNI_PROCESSOR_CACHE",
                Path(tempfile.gettempdir()) / "verl_omni_processors",
            )
        )
        cache_key = _processor_cache_key(processor_dir)
        prepared_dir = cache_root / f"qwen_image_edit_{cache_key}"
        if (prepared_dir / "config.json").is_file():
            return str(prepared_dir)

        cache_root.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix=f".{cache_key}-", dir=cache_root))
        temporary_processor = temporary_root / "processor"
        try:
            shutil.copytree(processor_dir, temporary_processor)
            with open(temporary_processor / "config.json", "w") as config_file:
                json.dump({"model_type": "qwen2_vl"}, config_file)
            try:
                temporary_processor.rename(prepared_dir)
            except OSError:
                if not (prepared_dir / "config.json").is_file():
                    raise
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        return str(prepared_dir)

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
