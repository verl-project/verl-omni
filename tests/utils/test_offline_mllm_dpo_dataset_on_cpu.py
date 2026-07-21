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
"""CPU unit tests for offline MLLM DPO dataset helpers and Omni-Preference preprocessing."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dataset_module():
    # Avoid importing verl_omni.__init__, which may pull optional rollout deps.
    sys.modules.setdefault("verl_omni", types.ModuleType("verl_omni"))
    sys.modules.setdefault("verl_omni.utils", types.ModuleType("verl_omni.utils"))
    sys.modules.setdefault("verl_omni.utils.dataset", types.ModuleType("verl_omni.utils.dataset"))

    dataset_dir = REPO_ROOT / "verl_omni" / "utils" / "dataset"
    transform_spec = importlib.util.spec_from_file_location(
        "verl_omni.utils.dataset.qwen3_omni_transform",
        dataset_dir / "qwen3_omni_transform.py",
    )
    if transform_spec is None or transform_spec.loader is None:
        raise ImportError("Cannot load qwen3_omni_transform module.")
    transform_module = importlib.util.module_from_spec(transform_spec)
    sys.modules[transform_spec.name] = transform_module
    transform_spec.loader.exec_module(transform_module)

    dataset_spec = importlib.util.spec_from_file_location(
        "verl_omni.utils.dataset.offline_mllm_dpo_dataset",
        dataset_dir / "offline_mllm_dpo_dataset.py",
    )
    if dataset_spec is None or dataset_spec.loader is None:
        raise ImportError("Cannot load offline_mllm_dpo_dataset module.")
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    sys.modules[dataset_spec.name] = dataset_module
    dataset_spec.loader.exec_module(dataset_module)
    return dataset_module


dataset_mod = _load_dataset_module()


def _load_multisource_module():
    module_path = REPO_ROOT / "examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py"
    spec = importlib.util.spec_from_file_location("omni_preference_dpo_multisource", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _content_item(item_type: str, *, text: str | None = None, image="", video="", audio="") -> dict:
    return {"type": item_type, "text": text, "image": image, "video": video, "audio": audio}


def _sample_prompt(modality: str = "image", media_path: str = "/tmp/dummy.png") -> list[dict]:
    media_item = _content_item(modality, **{modality: media_path})
    return [
        {"role": "system", "content": [_content_item("text", text="You are a helpful assistant.")]},
        {
            "role": "user",
            "content": [
                media_item,
                _content_item("text", text="What is shown?"),
            ],
        },
    ]


def _parquet_row(modality: str, index: int = 0) -> dict:
    return {
        "data_source": f"omni_preference/{modality}",
        "prompt": _sample_prompt(modality, f"/tmp/{modality}_{index}.bin"),
        "chosen": "preferred answer",
        "rejected": "rejected answer",
        "win_score": 8.0,
        "lose_score": 4.0,
        "ability": f"{modality}_qa",
        "reward_model": {"style": "preference"},
        "extra_info": {
            "split": "train",
            "index": index,
            "modality": modality,
            "question": "What is shown?",
        },
    }


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.get_rope_index = MagicMock(
        return_value={"position_ids": torch.zeros(1, 3, 4), "mrope_position_deltas": torch.zeros(1)}
    )
    return processor


@pytest.fixture
def mixed_parquet_path(tmp_path):
    rows = [
        _parquet_row("image", 0),
        _parquet_row("image", 1),
        _parquet_row("video", 0),
        _parquet_row("audio", 0),
    ]
    parquet_path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return str(parquet_path)


class FakeModalityDataset:
    def __init__(self, modalities: list[str]):
        self._modalities = modalities

    def __len__(self) -> int:
        return len(self._modalities)

    def get_modality(self, index: int) -> str:
        return self._modalities[index]


def test_as_python_parses_json_string():
    assert dataset_mod._as_python('{"a": 1}') == {"a": 1}
    assert dataset_mod._as_python("[1, 2]") == [1, 2]
    assert dataset_mod._as_python(b'{"k": "v"}') == {"k": "v"}


def test_build_preference_branch_extracts_media_and_answer():
    sample = {
        "prompt": _sample_prompt("video", "/tmp/clip.mp4"),
        "chosen": "answer A",
        "data_source": "omni_preference/video",
    }
    branch = dataset_mod._build_preference_branch(sample, sample["chosen"])

    assert branch["source_name"] == "omni_preference/video"
    assert branch["videos"] == ["/tmp/clip.mp4"]
    assert branch["conversations"][-1] == ["assistant", ("text", "answer A")]
    assert branch["conversations"][0][0] == "user"


def test_merge_chosen_rejected_concatenates_sequence_tensors():
    chosen = {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([3, 4])}
    rejected = {"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([7, 8])}

    merged = dataset_mod._merge_chosen_rejected(chosen, rejected)

    torch.testing.assert_close(merged["input_ids"], torch.tensor([1, 2, 5, 6]))
    torch.testing.assert_close(merged["labels"], torch.tensor([3, 4, 7, 8]))


def test_collate_tensor_values_pads_variable_length_sequences():
    values = [
        torch.tensor([1, 2]),
        torch.tensor([3, 4, 5]),
    ]
    collated = dataset_mod._collate_tensor_values("input_ids", values)

    assert collated.shape == (2, 3)
    torch.testing.assert_close(collated[0], torch.tensor([1, 2, 0]))
    torch.testing.assert_close(collated[1], torch.tensor([3, 4, 5]))


def test_offline_mllm_dpo_collate_fn_rejects_mixed_modalities():
    features = [
        {"modality": "image", "input_ids": torch.tensor([1])},
        {"modality": "video", "input_ids": torch.tensor([2])},
    ]
    with pytest.raises(ValueError, match="single modality"):
        dataset_mod.offline_mllm_dpo_collate_fn(features)


def test_offline_mllm_dpo_collate_fn_batches_same_modality():
    features = [
        {
            "modality": "image",
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([3, 4]),
            "extra_info": {"index": 0},
        },
        {
            "modality": "image",
            "input_ids": torch.tensor([5]),
            "labels": torch.tensor([6]),
            "extra_info": {"index": 1},
        },
    ]
    batch = dataset_mod.offline_mllm_dpo_collate_fn(features)

    assert batch["input_ids"].shape == (2, 2)
    assert batch["labels"].shape == (2, 2)
    assert batch["modality"].tolist() == ["image", "image"]
    assert batch["extra_info"].tolist() == [{"index": 0}, {"index": 1}]


def test_modality_grouped_batch_sampler_yields_same_modality_chunks():
    dataset = FakeModalityDataset(["image", "image", "video", "audio", "video", "audio"])
    sampler = dataset_mod.ModalityGroupedBatchSampler(
        data_source=dataset,
        batch_size=2,
        seed=0,
        num_batches=3,
    )

    batches: list[list[str]] = []
    current: list[str] = []
    for index in sampler:
        current.append(dataset.get_modality(index))
        if len(current) == sampler.batch_size:
            batches.append(current)
            current = []

    assert len(batches) == 3
    for batch_modalities in batches:
        assert len(set(batch_modalities)) == 1


def test_modality_grouped_batch_sampler_respects_weights(monkeypatch):
    dataset = FakeModalityDataset(["image", "video", "audio"])
    sampler = dataset_mod.ModalityGroupedBatchSampler(
        data_source=dataset,
        batch_size=1,
        seed=0,
        num_batches=6,
        modality_sample_weights={"video": 100.0, "image": 0.0, "audio": 0.0},
    )

    sampled = [dataset.get_modality(index) for index in sampler]
    assert sampled == ["video"] * 6


def test_offline_mllm_dpo_dataset_init_and_get_modality(mock_processor, mixed_parquet_path):
    config = OmegaConf.create({"train_batch_size": 2})
    dataset = dataset_mod.OfflineMLLMDPODataset(
        data_files=mixed_parquet_path,
        tokenizer=None,
        processor=mock_processor,
        config=config,
    )

    assert len(dataset) == 4
    assert dataset.get_modality(0) == "image"
    assert dataset.get_modality(2) == "video"
    assert dataset.get_modality(3) == "audio"


def test_offline_mllm_dpo_dataset_getitem(monkeypatch, mock_processor, mixed_parquet_path):
    def fake_transform(sample, **kwargs):
        answer = sample["conversations"][-1][1][1]
        token = 1 if answer == "preferred answer" else 2
        return [{"input_ids": torch.tensor([token]), "labels": torch.tensor([token])}]

    monkeypatch.setattr(dataset_mod, "process_qwen3_omni_sample", fake_transform)

    config = OmegaConf.create({"train_batch_size": 2})
    dataset = dataset_mod.OfflineMLLMDPODataset(
        data_files=mixed_parquet_path,
        tokenizer=None,
        processor=mock_processor,
        config=config,
    )
    item = dataset[0]

    torch.testing.assert_close(item["input_ids"], torch.tensor([1, 2]))
    torch.testing.assert_close(item["sample_level_scores"], torch.tensor([8.0, 4.0]))
    assert item["modality"] == "image"
    assert item["data_source"] == "omni_preference/image"
    assert item["extra_info"]["modality"] == "image"


def test_offline_mllm_dpo_dataset_requires_processor():
    with pytest.raises(ValueError, match="requires a multimodal processor"):
        dataset_mod.OfflineMLLMDPODataset(
            data_files=[],
            tokenizer=None,
            processor=None,
            config=OmegaConf.create({}),
        )


def test_multisource_parse_context_and_normalize_record():
    multisource = _load_multisource_module()
    content = (
        "prefix ### Context\n"
        "Image file: /data/rlaif-v-dataset/foo.jpg\n"
        "Question: What color is the object?\n"
        "Candidate A: red\n"
        "Candidate B: blue"
    )
    parsed = multisource._parse_context(content)
    assert parsed == {
        "media": "/data/rlaif-v-dataset/foo.jpg",
        "question": "What color is the object?",
        "candidate_a": "red",
        "candidate_b": "blue",
    }

    record = {
        "images": ["/data/rlaif-v-dataset/foo.jpg"],
        "messages": [{"role": "user", "content": content}],
        "solution": json.dumps({"score_A": 5, "score_B": 8, "better": "B"}),
    }
    normalized = multisource._normalize_record(record, "image", 3)

    assert normalized is not None
    assert normalized["chosen"] == "blue"
    assert normalized["rejected"] == "red"
    assert normalized["win_score"] == 8.0
    assert normalized["lose_score"] == 5.0
    assert normalized["dataset_media_rel"] == "rlaif-v-dataset/foo.jpg"


def test_multisource_skips_equal_verdict():
    multisource = _load_multisource_module()
    record = {
        "images": ["/data/foo.jpg"],
        "messages": [{"role": "user", "content": "no context"}],
        "solution": json.dumps({"score_A": 5, "score_B": 5, "better": "equal"}),
    }
    assert multisource._normalize_record(record, "image", 0) is None


def test_multisource_split_media_keys_keeps_train_and_test_disjoint():
    multisource = _load_multisource_module()
    records = [
        {"dataset_media_rel": "a/img1.jpg"},
        {"dataset_media_rel": "a/img1.jpg"},
        {"dataset_media_rel": "b/img2.jpg"},
        {"dataset_media_rel": "c/img3.jpg"},
    ]
    test_keys = multisource._split_media_keys(records, test_ratio=0.5, seed=42)
    train_keys = {multisource._media_key(record["dataset_media_rel"]) for record in records} - test_keys

    assert test_keys & train_keys == set()
    assert test_keys


def test_read_dataframe_supports_multiple_parquet_files(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    pd.DataFrame([_parquet_row("image", 0)]).to_parquet(first, index=False)
    pd.DataFrame([_parquet_row("video", 0)]).to_parquet(second, index=False)

    frame = dataset_mod._read_dataframe([str(first), str(second)])

    assert len(frame) == 2
    assert set(frame["data_source"]) == {"omni_preference/image", "omni_preference/video"}


def test_row_modality_prefers_extra_info():
    row = {
        "data_source": "omni_preference/image",
        "extra_info": {"modality": "video"},
    }
    assert dataset_mod._row_modality(row, "data_source") == "video"


def test_offline_mllm_dpo_collate_fn_squeezes_position_ids():
    features = [
        {
            "modality": "image",
            "position_ids": torch.ones(3, 1, 4),
            "input_ids": torch.tensor([1]),
        },
        {
            "modality": "image",
            "position_ids": torch.ones(3, 1, 4),
            "input_ids": torch.tensor([2]),
        },
    ]
    batch = dataset_mod.offline_mllm_dpo_collate_fn(features)

    assert batch["position_ids"].shape == (2, 3, 4)
