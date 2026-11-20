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
"""在 agent 中执行一次上游 2.1 提示词格式化和分词。"""

from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import calculate_dimensions


def prepare_reference_media(media):
    """先缩放一次参考图，使 agent 的 image slots 与 rollout 的 VAE 条件使用相同尺寸。"""
    if media.get("videos") or media.get("audios"):
        raise ValueError("Qwen-Image 2.1 accepts text and up to four reference images.")
    images = media.get("images") or []
    if len(images) > 4:
        raise ValueError("Qwen-Image 2.1 accepts at most four reference images.")
    resizer = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=True)
    resized = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError("Agent image decoding must produce PIL images.")
        width, height = calculate_dimensions(1024 * 1024, image.width / image.height)
        resized.append(resizer.resize(image, height=height, width=width))
    return {**media, "images": resized} if images else media


def tokenize_prompt(messages, media, processor):
    """对已经缩放的参考图执行一次原生模板分词，保持 image slots 与媒体顺序。"""
    if processor is None:
        raise ValueError("Qwen-Image 2.1 requires the checkpoint's processor subfolder.")
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Qwen-Image 2.1 NFT expects one user message; its system prompt is model-defined.")
    if media.get("videos") or media.get("audios"):
        raise ValueError("Qwen-Image 2.1 accepts text and up to four reference images.")
    content = messages[0]["content"]
    if isinstance(content, str):
        text = content
    else:
        if any(part["type"] not in ("text", "image", "image_url") for part in content):
            raise ValueError("Unsupported Qwen-Image 2.1 prompt content.")
        text = "".join(part["text"] for part in content if part["type"] == "text")
    images = media.get("images") or []
    if len(images) > 4:
        raise ValueError("Qwen-Image 2.1 accepts at most four reference images.")
    encoder_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError("Agent image decoding must produce PIL images.")
        if image.mode == "RGBA":
            white = Image.new("RGB", image.size, (255, 255, 255))
            white.paste(image, mask=image.getchannel("A"))
            image = white
        encoder_images.append(image)
    prefix = " ".join(f"<image{index + 1}><|vision_start|><|image_pad|><|vision_end|>" for index in range(len(images)))
    formatted = (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        f"<|im_start|>user\n{prefix}{text or ' '}<|im_end|>\n<|im_start|>assistant\n"
    )
    kwargs = {"text": [formatted], "padding": True, "padding_side": "left", "return_tensors": "pt"}
    if encoder_images:
        kwargs["images"] = encoder_images
    encoded = processor(**kwargs)
    ids = encoded.input_ids[0][encoded.attention_mask[0].bool()].tolist()
    return ids
