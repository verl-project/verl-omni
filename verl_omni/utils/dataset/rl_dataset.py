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
import math
import random
from collections import defaultdict

from omegaconf import DictConfig
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
    "create_rl_batch_sampler",
    "create_rl_sampler",
]


class ModalityBatchSampler:
    """Yield same-modality batches with configurable modality sampling weights."""

    is_batch_sampler = True

    def __init__(
        self,
        dataset,
        batch_size: int,
        modality_ratios: dict[str, float] | None = None,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = seed

        groups: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            if hasattr(dataset, "get_modality"):
                modality = dataset.get_modality(index)
            else:
                modality = getattr(dataset, "modalities", ["unknown"] * len(dataset))[index]
            groups[str(modality)].append(index)
        self.groups = {modality: indices for modality, indices in groups.items() if indices}
        if not self.groups:
            raise ValueError("ModalityBatchSampler requires at least one non-empty modality group.")

        ratios = modality_ratios or {}
        self.modalities = sorted(self.groups)
        self.weights = [float(ratios.get(modality, 1.0)) for modality in self.modalities]
        if any(weight < 0 for weight in self.weights):
            raise ValueError(
                "modality_ratios must be non-negative for present modalities, "
                f"got {dict(zip(self.modalities, self.weights, strict=True))}"
            )

        dataset_size = len(dataset)
        self.num_batches = (
            dataset_size // self.batch_size if self.drop_last else math.ceil(dataset_size / self.batch_size)
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = {modality: list(indices) for modality, indices in self.groups.items()}
        cursors = {modality: 0 for modality in groups}
        if self.shuffle:
            for indices in groups.values():
                rng.shuffle(indices)

        for _ in range(self.num_batches):
            modality = rng.choices(self.modalities, weights=self.weights, k=1)[0]
            indices = groups[modality]
            batch = []
            while len(batch) < self.batch_size:
                cursor = cursors[modality]
                remaining = len(indices) - cursor
                take = min(self.batch_size - len(batch), remaining)
                batch.extend(indices[cursor : cursor + take])
                cursors[modality] += take
                if len(batch) == self.batch_size:
                    break
                cursors[modality] = 0
                if self.shuffle:
                    rng.shuffle(indices)
                if not self.drop_last and len(indices) < self.batch_size and len(batch) == len(indices):
                    break
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch


class RLHFDataset(_UpstreamRLHFDataset):
    """Upstream :class:`RLHFDataset` extended with ``negative_prompt`` support.

    Diffusion models trained with classifier-free guidance need a paired
    negative prompt for every sample. We surface the raw negative prompt
    messages under ``raw_negative_prompt`` so the diffusion agent loop can
    encode them alongside the positive prompt.
    """

    def __init__(self, *args, config: DictConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        # For diffusion model training only.
        self.negative_prompt_key = config.get("negative_prompt_key", "negative_prompt")

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


def _is_modality_batch_sampler(data_config: DictConfig) -> bool:
    sampler_config = data_config.get("sampler", {})
    return sampler_config is not None and sampler_config.get("class_name", None) == "ModalityBatchSampler"


def create_rl_batch_sampler(
    data_config: DictConfig, dataset, *, batch_size: int | None = None, drop_last: bool | None = None
):
    sampler_config = data_config.get("sampler", {})
    if not _is_modality_batch_sampler(data_config):
        return None
    if batch_size is None:
        batch_size = sampler_config.get("batch_size", None) or data_config.train_batch_size
        batch_size = int(batch_size)
    if drop_last is None:
        drop_last = bool(sampler_config.get("drop_last", True))
    return ModalityBatchSampler(
        dataset,
        batch_size=batch_size,
        modality_ratios=dict(sampler_config.get("modality_ratios", {}) or {}),
        drop_last=drop_last,
        shuffle=bool(data_config.get("shuffle", True)),
        seed=data_config.get("seed"),
    )


def create_rl_sampler(data_config, dataset):
    batch_sampler = create_rl_batch_sampler(data_config, dataset)
    if batch_sampler is not None:
        return batch_sampler
    return _upstream_create_rl_sampler(data_config, dataset)
