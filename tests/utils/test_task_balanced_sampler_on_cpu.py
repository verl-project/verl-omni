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
import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


def _load_module():
    path = Path(__file__).parents[2] / "examples/diffusionopd_trainer/sd35/task_balanced_sampler.py"
    spec = importlib.util.spec_from_file_location("task_balanced_sampler", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TaskBalancedSampler = _load_module().TaskBalancedSampler


def _dataset(counts: dict[str, int]):
    sources = [task for task, count in counts.items() for _ in range(count)]
    return SimpleNamespace(dataframe={"data_source": sources})


def _chunks(order, size):
    return [order[i : i + size] for i in range(0, len(order), size)]


def test_every_batch_holds_equal_rows_per_task():
    dataset = _dataset({"ocr": 10, "pickscore": 7, "geneval": 20})
    sampler = TaskBalancedSampler(dataset, OmegaConf.create({"train_batch_size": 6, "seed": 3}))
    order = list(sampler)
    sources = dataset.dataframe["data_source"]

    assert len(order) == len(sampler) == 20 // 2 * 6
    for chunk in _chunks(order, 6):
        assert Counter(sources[i] for i in chunk) == {"ocr": 2, "pickscore": 2, "geneval": 2}


def test_largest_task_is_visited_once_and_short_tasks_cycle():
    dataset = _dataset({"a": 4, "b": 12})
    sampler = TaskBalancedSampler(dataset, OmegaConf.create({"train_batch_size": 4, "seed": 0}))
    order = list(sampler)

    b_rows = [i for i in order if dataset.dataframe["data_source"][i] == "b"]
    a_rows = [i for i in order if dataset.dataframe["data_source"][i] == "a"]
    assert sorted(b_rows) == list(range(4, 16))
    assert Counter(a_rows) == {0: 3, 1: 3, 2: 3, 3: 3}


def test_largest_task_remainder_fills_a_last_batch():
    dataset = _dataset({"a": 4, "b": 7})
    sampler = TaskBalancedSampler(dataset, OmegaConf.create({"train_batch_size": 4, "seed": 0}))
    order = list(sampler)

    assert len(order) == len(sampler) == 4 * 4
    b_rows = [i for i in order if dataset.dataframe["data_source"][i] == "b"]
    assert set(b_rows) == set(range(4, 11))
    assert len(b_rows) == 8


def test_largest_task_smaller_than_rows_per_task_raises():
    with pytest.raises(ValueError, match="rows per task"):
        TaskBalancedSampler(_dataset({"a": 1, "b": 1}), OmegaConf.create({"train_batch_size": 4}))


def test_gen_batch_size_takes_precedence_and_epochs_reshuffle():
    dataset = _dataset({"a": 30, "b": 30})
    sampler = TaskBalancedSampler(dataset, OmegaConf.create({"train_batch_size": 7, "gen_batch_size": 10, "seed": 1}))

    first, second = list(sampler), list(sampler)
    assert len(first) == 60
    assert first != second
    assert sorted(first) == sorted(second)


def test_batch_size_not_divisible_by_tasks_raises():
    with pytest.raises(ValueError, match="multiple"):
        TaskBalancedSampler(_dataset({"a": 5, "b": 5, "c": 5}), OmegaConf.create({"train_batch_size": 8}))
