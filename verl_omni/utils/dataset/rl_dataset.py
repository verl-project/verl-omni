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
"""RLHF Dataset for diffusion model training."""

import logging
import os
import re
from io import BytesIO

from omegaconf import DictConfig
from PIL import Image
from verl.trainer.ppo.utils import create_rl_dataset as _upstream_create_rl_dataset
from verl.trainer.ppo.utils import create_rl_sampler as _upstream_create_rl_sampler
from verl.utils.dataset.rl_dataset import RLHFDataset as _UpstreamRLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn as _upstream_collate_fn
from verl.utils.dataset.rl_dataset import get_dataset_class as _upstream_get_dataset_class
from verl.utils.import_utils import load_extern_object

logger = logging.getLogger(__name__)


__all__ = [
    "RLHFDataset",
    "get_collate_fn",
    "get_dataset_class",
    "create_rl_dataset",
    "create_rl_sampler",
]


class RLHFDataset(_UpstreamRLHFDataset):
    """Upstream :class:`RLHFDataset` extended with ``negative_prompt`` support.

    Diffusion models trained with classifier-free guidance need a paired
    negative prompt for every sample. We surface the raw negative prompt
    messages under ``raw_negative_prompt`` so the diffusion agent loop can
    encode them alongside the positive prompt.
    """

    def __init__(self, *args, config: DictConfig, **kwargs):
        # _build_messages also runs during upstream prompt filtering.
        self.negative_prompt_key = config.get("negative_prompt_key", "negative_prompt")
        super().__init__(*args, config=config, **kwargs)

    def _build_messages(self, example: dict, key: str):
        """Build structured messages without requiring a processor for media transport.

        Adapted from verl.utils.dataset.rl_dataset.RLHFDataset._build_messages;
        retain its media formats and count checks without its processor guard.
        """
        messages = example[key]
        # Text-only negative prompts do not consume the positive condition media.
        if (
            self.processor is None
            and key == self.negative_prompt_key
            and all(
                isinstance(message["content"], str)
                and not any(token in message["content"] for token in ("<image>", "<video>", "<audio>"))
                for message in messages
            )
        ):
            return messages

        images = example.get(self.image_key) or []
        videos = example.get(self.video_key) or []
        audios = example.get(self.audio_key) or []
        image_offset, video_offset, audio_offset = 0, 0, 0
        for message in messages:
            if not images and not videos and not audios:
                continue
            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>|<audio>)", content)
            for segment in (item for item in segments if item != ""):
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        content_list.append({"type": "image", "image": image.convert("RGB")})
                    elif isinstance(image, dict):
                        if "bytes" in image:
                            image["image"] = Image.open(BytesIO(image["bytes"]))
                        content_list.append({"type": "image", **image})
                    elif isinstance(image, str | os.PathLike):
                        content_list.append({"type": "image", "image": os.fspath(image)})
                    else:
                        raise TypeError(
                            f"image must be dict, PIL.Image, or path-like, unsupported image type: {type(image)}"
                        )
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    video = videos[video_offset]
                    if isinstance(video, dict):
                        content_list.append({"type": "video", **video})
                    elif isinstance(video, str | os.PathLike):
                        content_list.append({"type": "video", "video": os.fspath(video)})
                    elif isinstance(video, list):
                        video = [os.fspath(frame) if isinstance(frame, os.PathLike) else frame for frame in video]
                        content_list.append({"type": "video", "video": video})
                    else:
                        raise TypeError(
                            f"video must be dict, list, or path-like, unsupported video type: {type(video)}"
                        )
                    video_offset += 1
                elif segment == "<audio>":
                    assert audio_offset < len(audios), f"audio_offset {audio_offset} >= len(audios) {len(audios)}"
                    audio = audios[audio_offset]
                    if isinstance(audio, dict):
                        payload = dict(audio)
                        payload["type"] = "audio"
                        if "audio" not in payload and "audio_url" not in payload:
                            payload = {"type": "audio", "audio": audio}
                        content_list.append(payload)
                    else:
                        content_list.append({"type": "audio", "audio": audio})
                    audio_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        assert audio_offset == len(audios), f"audio_offset {audio_offset} != len(audios) {len(audios)}"
        return messages

    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        raw = self.dataframe[item]
        negative_messages = None
        if self.negative_prompt_key in raw:
            negative_messages = self._build_messages(dict(raw), key=self.negative_prompt_key)

        row_dict = super().__getitem__(item)
        if negative_messages is not None:
            row_dict["raw_negative_prompt"] = negative_messages
        return row_dict


def get_collate_fn(data_config: DictConfig):
    """Get a custom collate function from data config, falling back to upstream default."""
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        collate_fn_name = data_config.custom_cls.get("collate_fn", None)
        if collate_fn_name is not None:
            custom_collate_fn = load_extern_object(data_config.custom_cls.path, collate_fn_name)
            if not callable(custom_collate_fn):
                raise TypeError(
                    f"The custom collate function '{collate_fn_name}' from "
                    f"'{data_config.custom_cls.path}' must be callable"
                )
            logger.info("Using custom collate function: %s", collate_fn_name)
            return custom_collate_fn
    logger.info("Using default collate function")
    return _upstream_collate_fn


def get_dataset_class(data_config: DictConfig):
    """Get RLHF dataset class.

    Args:
        data_config: The data config.

    Returns:
        dataset_cls: The dataset class.
    """

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        return _upstream_get_dataset_class(data_config)
    logger.info("Using dataset class: %s", RLHFDataset.__name__)
    return RLHFDataset


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        return _upstream_create_rl_dataset(
            data_paths, data_config, tokenizer, processor, is_train=is_train, max_samples=max_samples
        )
    return RLHFDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )


def create_rl_sampler(data_config: DictConfig, dataset, sampler_config: DictConfig | dict | None = None):
    """Create an RL dataset sampler with optional verl-omni custom classes.

    When a sampler config provides both ``class_path`` and ``class_name``, the
    sampler class is loaded dynamically and initialized with the dataset, data
    config, and optional ``sampler_kwargs``. If ``sampler_config`` is omitted,
    this falls back to the legacy ``data_config.sampler`` field.

    Args:
        data_config: Data config that may contain a custom sampler definition.
        dataset: Dataset instance to sample from.
        sampler_config: Optional explicit sampler config, e.g.
            ``data.train_sampler`` or ``data.val_sampler``.

    Returns:
        Sampler instance used by the RL DataLoader.
    """
    if sampler_config is None:
        sampler_config = data_config.get("sampler", None)
    if sampler_config is not None:
        class_path = sampler_config.get("class_path", None)
        class_name = sampler_config.get("class_name", None)
        if class_path is not None and class_name is not None:
            sampler_cls = load_extern_object(class_path, class_name)
            sampler_kwargs = sampler_config.get("sampler_kwargs", {}) or {}
            logger.info("Using custom sampler: %s from %s", class_name, class_path)
            return sampler_cls(data_source=dataset, data_config=data_config, **sampler_kwargs)

    return _upstream_create_rl_sampler(data_config, dataset)
