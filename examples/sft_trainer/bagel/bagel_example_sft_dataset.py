# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Adapter for training on the official BAGEL example dataset.

The native BAGEL reader stores example paths in DATASET_INFO.  This adapter
patches those paths from the launch config, then delegates packing to
``verl_omni.utils.dataset.bagel_sft_dataset.PackedDataset``.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

from verl_omni.utils.dataset.bagel_sft_dataset import DATASET_INFO, DataConfig, PackedDataset, add_special_tokens
from verl_omni.utils.dataset.bagel_sft_dataset import collate_wrapper


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _custom_config(config: Any) -> Any:
    return _config_get(config, "custom_cls", {}) or {}


def _read_yaml(path: str) -> dict[str, Any]:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _patch_bagel_example_paths(bagel_example_dir: str) -> None:
    bagel_example_dir = os.path.abspath(os.path.expanduser(bagel_example_dir))
    DATASET_INFO["t2i_pretrain"]["t2i"]["data_dir"] = os.path.join(bagel_example_dir, "t2i")
    DATASET_INFO["unified_edit"]["seedxedit_multi"]["data_dir"] = os.path.join(
        bagel_example_dir, "editing", "seedxedit_multi"
    )
    DATASET_INFO["unified_edit"]["seedxedit_multi"]["parquet_info_path"] = os.path.join(
        bagel_example_dir, "editing", "parquet_info", "seedxedit_multi_nas.json"
    )
    DATASET_INFO["vlm_sft"]["llava_ov"]["data_dir"] = os.path.join(bagel_example_dir, "vlm", "images")
    DATASET_INFO["vlm_sft"]["llava_ov"]["jsonl_path"] = os.path.join(bagel_example_dir, "vlm", "llava_ov_si.jsonl")


class BagelExampleSFTDataset(PackedDataset):
    """Instantiate ``PackedDataset`` through verl's custom dataset interface."""

    def __init__(
        self,
        data_files=None,
        tokenizer=None,
        processor=None,
        config=None,
        is_train: bool = True,
        max_samples: int = -1,
        **kwargs,
    ):
        del data_files, processor, is_train, max_samples, kwargs
        custom_config = _custom_config(config)

        bagel_example_dir = _config_get(custom_config, "bagel_example_dir", os.environ.get("BAGEL_EXAMPLE_DIR"))
        if not bagel_example_dir:
            raise ValueError("Set data.custom_cls.bagel_example_dir or BAGEL_EXAMPLE_DIR.")
        _patch_bagel_example_paths(bagel_example_dir)

        dataset_config_file = _config_get(custom_config, "dataset_config_file", None)
        if dataset_config_file is None:
            raise ValueError("Set data.custom_cls.dataset_config_file.")
        grouped_datasets = _read_yaml(dataset_config_file)

        tokenizer, special_tokens, _ = add_special_tokens(tokenizer)
        data_config = DataConfig(
            grouped_datasets=copy.deepcopy(grouped_datasets),
            text_cond_dropout_prob=float(_config_get(custom_config, "text_cond_dropout_prob", 0.1)),
            vit_cond_dropout_prob=float(_config_get(custom_config, "vit_cond_dropout_prob", 0.3)),
            vae_cond_dropout_prob=float(_config_get(custom_config, "vae_cond_dropout_prob", 0.3)),
            vae_image_downsample=int(_config_get(custom_config, "vae_image_downsample", 16)),
            max_latent_size=int(_config_get(custom_config, "max_latent_size", 64)),
            vit_patch_size=int(_config_get(custom_config, "vit_patch_size", 14)),
            max_num_patch_per_side=int(_config_get(custom_config, "max_num_patch_per_side", 70)),
        )

        super().__init__(
            data_config=data_config,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            local_rank=int(_config_get(custom_config, "data_rank", 0)),
            world_size=int(_config_get(custom_config, "data_world_size", 1)),
            num_workers=int(_config_get(config, "dataloader_num_workers", 1)),
            expected_num_tokens=int(_config_get(custom_config, "expected_num_tokens", 10240)),
            max_num_tokens_per_sample=int(_config_get(custom_config, "max_num_tokens_per_sample", 10240)),
            max_num_tokens=int(_config_get(custom_config, "max_num_tokens", 11520)),
            prefer_buffer_before=int(_config_get(custom_config, "prefer_buffer_before", 10240)),
            max_buffer_size=int(_config_get(custom_config, "max_buffer_size", 50)),
            interpolate_pos=bool(_config_get(custom_config, "interpolate_pos", False)),
            use_flex=bool(_config_get(custom_config, "use_flex", True)),
        )
        self.num_packed_batches = int(_config_get(custom_config, "num_packed_batches", 1000))

    def __len__(self) -> int:
        return self.num_packed_batches


bagel_example_sft_collate_fn = collate_wrapper()
