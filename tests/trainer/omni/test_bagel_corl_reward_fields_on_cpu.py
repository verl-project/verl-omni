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

"""The colocated RM must receive ``data_source`` and ``reward_model`` for every episode.

Without them ``VisualRewardManager`` scores a structural zero: 85 steps of
``bagel_corl_prfix_t07_20261015_190942_a1`` trained on all-zero rewards and advantages.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl_omni.trainer.omni import bagel_corl_trainer as trainer_module
from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync


class _FakeTQ:
    def __init__(self, rows: dict):
        self.rows = rows
        self.selected: list[str] = []
        self.put: dict | None = None

    def kv_batch_get(self, keys, partition_id, select_fields):
        self.selected = list(select_fields)
        return {k: v for k, v in self.rows.items() if k in select_fields}

    def kv_batch_put(self, keys, partition_id, fields):
        self.put = fields


class _FakeRewardLoop:
    def __init__(self):
        self.seen = None

    def compute_rm_score(self, data):
        self.seen = data
        return SimpleNamespace(batch={"rm_scores": torch.zeros(len(data), 4)}, meta_info={}, non_tensor_batch={})


def _jagged(rows):
    return torch.nested.as_nested_tensor([torch.tensor(r) for r in rows], layout=torch.jagged)


def test_colocated_reward_hands_the_scorer_unwrapped_row_fields(monkeypatch):
    import transfer_queue

    gts = [{"ground_truth": {"task_type": "plan"}}, {"ground_truth": {"task_type": "reflect"}}]
    fake = _FakeTQ(
        {
            "prompts": _jagged([[1, 2], [3]]),
            "responses": _jagged([[4, 5, 6], [7]]),
            "raw_prompt": [[{"role": "user"}], [{"role": "user"}]],
            # TQ hands non-tensor fields back wrapped; the scorer must see plain values.
            "data_source": NonTensorStack(NonTensorData("unicot_breakdown"), NonTensorData("unicot_reflection")),
            "reward_model": NonTensorStack(*(NonTensorData(g) for g in gts)),
        }
    )
    for name in ("kv_batch_get", "kv_batch_put"):
        monkeypatch.setattr(transfer_queue, name, getattr(fake, name))

    trainer = object.__new__(OmniBagelCoRLTrainerSync)
    trainer.reward_loop_manager = _FakeRewardLoop()
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    batch = type("B", (), {"keys": ["a", "b"], "partition_id": "train", "__len__": lambda self: 2})()

    trainer._compute_reward_colocate(batch)

    assert {"data_source", "reward_model"} <= set(fake.selected)
    seen = trainer.reward_loop_manager.seen.non_tensor_batch
    assert list(seen["data_source"]) == ["unicot_breakdown", "unicot_reflection"]
    assert [row["ground_truth"]["task_type"] for row in seen["reward_model"]] == ["plan", "reflect"]
    assert isinstance(seen["reward_model"], np.ndarray)
    assert fake.put is not None


def test_override_is_bound_on_the_trainer_not_inherited():
    assert "_compute_reward_colocate" in vars(OmniBagelCoRLTrainerSync)
    # ``extra_info`` rides along: ``VisualRewardManager._reward_extra_info`` reads it
    # to rebuild the media/provenance metadata the episode scorer needs.
    assert trainer_module.OmniBagelCoRLTrainerSync.REWARD_ROW_FIELDS == (
        "data_source",
        "reward_model",
        "extra_info",
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
