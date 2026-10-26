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
"""Task-balanced sampler for multi-teacher distillation on mixed datasets.

Loaded through ``data.sampler.class_path`` / ``data.sampler.class_name``.
"""

import torch
from torch.utils.data import Sampler


class TaskBalancedSampler(Sampler[int]):
    """Order rows so every generation batch holds the same number of rows per ``data_source``.

    One optimizer step over such a batch matches feeding one single-task batch per
    teacher under gradient accumulation. Every task is reshuffled each epoch, and
    shorter tasks are cycled so an epoch covers every row of the largest task,
    whose own last batch is topped up the same way.

    Args:
        data_source: ``RLHFDataset`` whose ``dataframe`` has a ``data_source`` column.
        data_config: Data config providing ``gen_batch_size`` or ``train_batch_size`` and ``seed``.
        seed: Overrides ``data_config.seed``.
    """

    def __init__(self, data_source, data_config, seed: int | None = None):
        self.task_indices: dict[str, torch.Tensor] = {}
        for index, source in enumerate(data_source.dataframe["data_source"]):
            self.task_indices.setdefault(source, []).append(index)
        self.task_indices = {task: torch.tensor(rows) for task, rows in self.task_indices.items()}

        batch_size = int(data_config.get("gen_batch_size", None) or data_config.train_batch_size)
        if batch_size % len(self.task_indices) != 0:
            raise ValueError(
                f"Generation batch size {batch_size} is not a multiple of the {len(self.task_indices)} tasks "
                f"{sorted(self.task_indices)}."
            )
        self.rows_per_task = batch_size // len(self.task_indices)
        largest_task = max(len(rows) for rows in self.task_indices.values())
        if largest_task < self.rows_per_task:
            raise ValueError(
                f"Every task has fewer rows than the {self.rows_per_task} rows per task a generation batch needs."
            )
        self.num_batches = -(-largest_task // self.rows_per_task)
        self.seed = int(data_config.get("seed", 0) if seed is None else seed)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        rows_needed = self.num_batches * self.rows_per_task
        columns = []
        for rows in self.task_indices.values():
            repeats = -(-rows_needed // len(rows))
            shuffled = torch.cat([rows[torch.randperm(len(rows), generator=generator)] for _ in range(repeats)])
            columns.append(shuffled[:rows_needed].view(self.num_batches, self.rows_per_task))
        yield from torch.cat(columns, dim=1).flatten().tolist()

    def __len__(self) -> int:
        return self.num_batches * self.rows_per_task * len(self.task_indices)
