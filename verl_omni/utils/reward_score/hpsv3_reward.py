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

import logging
import math
import os
import threading

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_lock = threading.Lock()
_inferencer = None

_INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text 
Alignment and give a overall score to estimate the human preference. Please provide 
a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Visual Quality:**
Evaluate the overall visual quality of the image. The following sub-dimensions 
should be considered:
- **Reasonableness:** The image should not contain any significant biological or 
  logical errors, such as abnormal body structures or nonsensical environmental 
  setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image 
  should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, 
  lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, 
  including the color scheme, composition, atmosphere, depth of field, and the 
  overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, 
  such as political, violent, or adult material. If such content is present, the 
  image quality and satisfaction score should be the lowest possible.

**Text Alignment:**
Assess how well the image matches the textual prompt across the following 
sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image 
  (e.g., person, animal, object) align with the textual description. The subject 
  should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic 
  style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and 
  surrounding elements in the image logically fit the scenario described in the 
  prompt. The environment should support and enhance the subject without 
  contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt 
  (e.g., colors, clothing, accessories, expressions, actions) are faithfully 
  represented in the image. Minor deviations may be acceptable, but critical 
  attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the 
  prompt are captured in the image. The generated content should not introduce 
  elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

_PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

_PROMPT_WITHOUT_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: 
"""

_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

_IMAGE_FACTOR = 28
_MIN_PIXELS = 4 * 28 * 28
_MAX_PIXELS = 16384 * 28 * 28
_MAX_RATIO = 200


def _round_by_factor(number, factor):
    return round(number / factor) * factor


def _ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def _floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def _smart_resize(height, width, factor=_IMAGE_FACTOR, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS):
    if max(height, width) / min(height, width) > _MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {_MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _fetch_image(ele):
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            from io import BytesIO

            import requests

            image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
        elif image.startswith("file://"):
            image = Image.open(image[7:]).convert("RGB")
        elif image.startswith("data:image"):
            import base64
            from io import BytesIO

            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(data)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
    else:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = _smart_resize(ele["resized_height"], ele["resized_width"])
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", _MIN_PIXELS)
        max_pixels = ele.get("max_pixels", _MAX_PIXELS)
        resized_height, resized_width = _smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    image = image.resize((resized_width, resized_height), Image.BICUBIC)
    return image


def _extract_vision_info(conversations):
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if "image" in ele or "image_url" in ele or ele.get("type") in ("image", "image_url"):
                        vision_infos.append(ele)
    return vision_infos


def _process_vision_info(conversations):
    vision_infos = _extract_vision_info(conversations)
    image_inputs = [_fetch_image(v) for v in vision_infos if "image" in v or "image_url" in v]
    return image_inputs if image_inputs else None


class _Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    __module__ = Qwen2VLForConditionalGeneration.__module__

    def __init__(
        self,
        config,
        output_dim=2,
        reward_token="special",
        special_token_ids=None,
        rm_head_type="ranknet",
    ):
        super().__init__(config)
        self.output_dim = output_dim
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids

        if rm_head_type == "ranknet":
            self.rm_head = nn.Sequential(
                nn.Linear(config.text_config.hidden_size, 1024),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(1024, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim),
            )
        else:
            self.rm_head = nn.Linear(config.text_config.hidden_size, output_dim, bias=False)

        self.rm_head.to(torch.float32)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = self.rm_head(hidden_states)  # [B, L, N]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for stid in self.special_token_ids:
            special_token_mask = special_token_mask | (input_ids == stid)
        pooled_logits = logits[special_token_mask, ...]
        pooled_logits = pooled_logits.view(batch_size, 1, -1)
        pooled_logits = pooled_logits.view(batch_size, -1)

        return {"logits": pooled_logits}


def _remap_state_dict(state_dict, model_keys):
    if any("model.language_model" in k for k in model_keys) and not any("language_model" in k for k in state_dict):
        remapped = {}
        for key, value in state_dict.items():
            if "visual" in key:
                remapped[key.replace("visual", "model.visual")] = value
            elif "model" in key:
                remapped[key.replace("model", "model.language_model")] = value
            else:
                remapped[key] = value
        return remapped
    return state_dict


class _HPSv3Inferencer:
    def __init__(self, checkpoint_path: str, base_config: str = _BASE_MODEL, device: str = "npu"):
        logger.info("Creating HPSv3 model")
        config = AutoConfig.from_pretrained(base_config, trust_remote_code=True)

        processor = AutoProcessor.from_pretrained(base_config, padding_side="right")
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

        model = _Qwen2VLRewardModelBT(
            config,
            output_dim=2,
            reward_token="special",
            special_token_ids=special_token_ids,
            rm_head_type="ranknet",
        )
        model.resize_token_embeddings(len(processor.tokenizer))
        model.to(torch.bfloat16)
        model.rm_head.to(torch.float32)
        model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        logger.info("Loading HPSv3 checkpoint from %s", checkpoint_path)
        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch

            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]

        state_dict = _remap_state_dict(state_dict, model.state_dict().keys())
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        model.to(device)
        self.model = model
        self.processor = processor
        self.device = device
        self.use_special_tokens = True

    def _prepare_input(self, data):
        from collections.abc import Mapping

        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, tuple | list):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def prepare_batch(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths, strict=False):
            message_list.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image, "min_pixels": max_pixels, "max_pixels": max_pixels},
                            {
                                "type": "text",
                                "text": _INSTRUCTION.format(text_prompt=text) + _PROMPT_WITH_SPECIAL_TOKEN,
                            },
                        ],
                    }
                ]
            )

        image_inputs = _process_vision_info(message_list)
        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        return self._prepare_input(batch)

    @torch.no_grad()
    def reward(self, image_paths, prompts):
        batch = self.prepare_batch(image_paths, prompts)
        return self.model(return_dict=True, **batch)["logits"]


def _get_inferencer(checkpoint_path: str, device: str):
    global _inferencer

    with _lock:
        if _inferencer is None:
            _inferencer = _HPSv3Inferencer(checkpoint_path=checkpoint_path, device=device)

    return _inferencer


def _to_pil_hwc(image) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.float().cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)
    return image


def _extract_frames(solution_image, frame_interval: int = 1) -> list[Image.Image]:
    is_channels_last = solution_image.shape[-1] in (1, 3) if solution_image.ndim >= 3 else False

    if solution_image.ndim == 3:
        if is_channels_last:
            solution_image = solution_image.permute(2, 0, 1)
        solution_image = solution_image.unsqueeze(0)

    elif solution_image.ndim == 4:
        if is_channels_last:
            solution_image = solution_image.permute(3, 0, 1, 2)
        solution_image = solution_image[:, ::frame_interval]
        solution_image = solution_image.permute(1, 0, 2, 3)

    elif solution_image.ndim == 5:
        if is_channels_last:
            solution_image = solution_image.permute(0, 4, 1, 2, 3)
        solution_image = solution_image[:, :, ::frame_interval]
        solution_image = solution_image.permute(0, 2, 1, 3, 4)
        solution_image = solution_image.reshape(-1, *solution_image.shape[2:])

    return [_to_pil_hwc(frame) for frame in solution_image]


def compute_score_hpsv3(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    model_name: str = None,
    reward_scale: float = 0.1,
    device: str = "npu",
    **kwargs,
) -> dict:
    checkpoint_path = os.getenv("custom_reward_model_path", model_name)
    assert checkpoint_path is not None, "HPSv3 checkpoint path must be provided via reward.reward_model.model_path"

    inferencer = _get_inferencer(checkpoint_path, device)

    frame_interval = extra_info.get("frame_interval", 4)
    pil_images = _extract_frames(solution_image, frame_interval=frame_interval)

    prompt = ground_truth if ground_truth else ""
    with _lock:
        raw_rewards = inferencer.reward(pil_images, [prompt] * len(pil_images))
    scores = [raw_rewards[i][0].item() * reward_scale for i in range(len(pil_images))]
    raw_reward_values = [raw_rewards[i][0].item() for i in range(len(pil_images))]

    score = sum(scores) / len(scores)
    avg_raw = sum(raw_reward_values) / len(raw_reward_values)

    return {"score": score, "hpsv3_raw": avg_raw}
