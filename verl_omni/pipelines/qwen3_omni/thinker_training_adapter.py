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
"""Qwen3-Omni Thinker training adapter.

Implements ``OmniModelBase`` for thinker-stage training of
Qwen3-Omni: sub-module stripping, forward redirection, and
processor/tokenizer configuration.
"""

import json
import logging
import os
from typing import Any

import torch
from tensordict import TensorDict

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


@OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")
class Qwen3OmniThinkerAdapter(OmniModelBase):
    """Thinker-stage training adapter for Qwen3-Omni.

    Handles model setup that is required before verl's FSDP engine
    loads and wraps the model: sub-module stripping, forward redirection
    to the thinker component, and processor/tokenizer configuration.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["talker", "code2wav", "code_predictor"]

    @classmethod
    def configure_model(cls, module, model_config):
        """Strip non-training stages and redirect forward to thinker.

        Args:
            module: The loaded Qwen3-Omni model before FSDP wrapping.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured module with talker/codec stripped and
            forward/embedding accessors redirected to thinker.
        """
        module = super().configure_model(module, model_config)
        module.forward = module.thinker.forward
        module.get_input_embeddings = module.thinker.get_input_embeddings
        module.set_input_embeddings = module.thinker.set_input_embeddings
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the Qwen3-Omni multimodal processor with RoPE helpers.

        Swaps ``processor.config`` to ``thinker_config`` (Qwen3-Omni
        nests multimodal settings under sub-configs).  Binds
        ``get_rope_index`` and ``get_llm_pos_ids_for_vision`` to the
        processor — the omni agent loop calls these on the processor,
        but they are model methods.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured processor with RoPE helpers bound.
        """
        import types

        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)

        processor.config = config.thinker_config
        processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
        processor.config.vision_start_token_id = config.talker_config.vision_start_token_id

        model_cls = Qwen3OmniMoeThinkerForConditionalGeneration
        processor.get_rope_index = types.MethodType(model_cls.get_rope_index, processor)
        processor.get_llm_pos_ids_for_vision = types.MethodType(model_cls.get_llm_pos_ids_for_vision, processor)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the tokenizer with chat template from ``chat_template.json``.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured tokenizer with ``chat_template`` loaded from
            ``chat_template.json``.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if not os.path.isfile(chat_template_path):
            raise FileNotFoundError(
                f"Qwen3-Omni chat template not found at {chat_template_path}. "
                f"Ensure the model checkpoint includes chat_template.json."
            )
        with open(chat_template_path) as f:
            tokenizer.chat_template = json.load(f)["chat_template"]
        return tokenizer

    @staticmethod
    def _drop_zero_rows(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        keep = (tensor.reshape(tensor.shape[0], -1) != 0).any(dim=-1)
        return tensor[keep]

    @classmethod
    def prepare_model_inputs(
        cls,
        model_config,
        micro_batch: TensorDict,
        *,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Build Qwen3-Omni thinker forward kwargs from a training micro-batch.

        Uses ``micro_batch`` , then applies Qwen3-Omni
        normalization so the result can be passed to
        ``model(**model_inputs, use_cache=False)``.

        ``micro_batch`` key contract
        ----------------------------

        **Required (text / log-prob)**

        - ``input_ids`` (``LongTensor``, shape ``(B, L)``): Token ids for the
          full prompt + response sequence. Multimodal placeholder positions may
          use model-specific sentinel indices before the thinker forward.
        - ``attention_mask`` (``LongTensor`` or ``BoolTensor``, shape ``(B, L)``):
          ``1``/``True`` for real tokens, ``0``/``False`` for padding.
        - ``labels`` (``LongTensor``, shape ``(B, L)``): Supervision mask for
          token log-prob computation. Prompt positions must be ``-100``;
          response positions carry the target token ids. The trainer/engine
          typically shifts these internally when gathering log-probs.
        - ``position_ids`` (``LongTensor``): mRoPE positions for Qwen3-Omni.
          Accepted layouts:

          - ``(B, 3, L)`` — preferred after dataset collation;
          - ``(B, 3, 1, L)`` — collated with an extra singleton axis; squeezed
            here to ``(B, 3, L)``;
          - other ranks are passed through unchanged.

        **Optional (image)**

        Include both keys when the sample contains images:

        - ``pixel_values`` (``FloatTensor``, shape ``(B, N_img, D)`` or
          ``(N_img, D)`` after per-sample padding): Vision patch embeddings fed
          to the thinker. Zero-padded rows (entire row is zero) are dropped.
        - ``image_grid_thw`` (``LongTensor``, shape ``(B, N_img, 3)`` or
          ``(N_img, 3)``): ``(T, H, W)`` grid metadata per image patch group.
          Zero rows are dropped to match the filtered ``pixel_values``.

        **Optional (video)**

        Include both keys when the sample contains videos:

        - ``pixel_values_videos`` (``FloatTensor``): Same role as
          ``pixel_values`` but for video patches.
        - ``video_grid_thw`` (``LongTensor``): Same role as ``image_grid_thw``
          but for video patch groups.

        **Optional (audio)**

        Include when the sample contains audio:

        - ``input_features`` (``FloatTensor``, shape ``(B, N_audio, D)`` or
          ``(N_audio, D)``): Audio features for the thinker. Zero-padded rows
          are dropped.
        - ``feature_attention_mask`` (``LongTensor`` or ``BoolTensor``):
          Per-audio-frame validity mask from the processor. Passed through to
          the model when present; not rewritten here.
        - ``audio_feature_lengths`` (``LongTensor``, shape ``(B,)`` or
          ``(B, 1)``): Effective audio feature length per sample. Entries equal
          to ``0`` are removed; 1-D inputs are flattened first.



        Normalization applied here
        --------------------------

        - Squeeze mRoPE ``position_ids`` when an extra singleton axis is present.
        - Drop all-zero rows from padded image/video/audio tensors and grids.
        - Cast floating-point multimodal tensors to ``dtype`` when provided.

        Args:
            model_config: ``OmniModelConfig`` (or compatible object with
                ``architecture`` and ``model_stage`` for registry lookup).
            micro_batch: ``TensorDict`` produced by the dataloader/collate path.
            dtype: Optional parameter dtype for ``pixel_values``,
                ``pixel_values_videos``, and ``input_features``.

        Returns:
            dict[str, Any]: Keyword arguments ready for the thinker
            ``forward()`` call.
        """
        model_inputs = dict(micro_batch)

        position_ids = model_inputs.get("position_ids")
        if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 4 and position_ids.shape[2] == 1:
            model_inputs["position_ids"] = position_ids.squeeze(2).contiguous()

        for pixel_key, grid_key in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
            pixel_values = model_inputs.get(pixel_key)
            grid = model_inputs.get(grid_key)
            if isinstance(grid, torch.Tensor) and grid.ndim == 3:
                model_inputs[grid_key] = cls._drop_zero_rows(grid.reshape(-1, grid.shape[-1]))
            if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
                model_inputs[pixel_key] = cls._drop_zero_rows(pixel_values.reshape(-1, pixel_values.shape[-1]))

        input_features = model_inputs.get("input_features")
        if isinstance(input_features, torch.Tensor) and input_features.ndim == 3:
            model_inputs["input_features"] = cls._drop_zero_rows(input_features.reshape(-1, input_features.shape[-1]))

        audio_feature_lengths = model_inputs.get("audio_feature_lengths")
        if isinstance(audio_feature_lengths, torch.Tensor):
            if audio_feature_lengths.ndim > 1:
                audio_feature_lengths = audio_feature_lengths.reshape(-1)
            model_inputs["audio_feature_lengths"] = audio_feature_lengths[audio_feature_lengths != 0]

        if dtype is not None:
            for key in ("pixel_values", "pixel_values_videos", "input_features"):
                value = model_inputs.get(key)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    model_inputs[key] = value.to(dtype=dtype)
        return model_inputs
