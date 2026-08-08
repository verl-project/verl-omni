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
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class _NestedTokens:
    def __init__(self, rows):
        self.rows = [torch.tensor(row, dtype=torch.int64) for row in rows]

    def offsets(self):
        lengths = torch.tensor([len(row) for row in self.rows], dtype=torch.int64)
        return torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])

    def to_padded_tensor(self, padding):
        return torch.nn.utils.rnn.pad_sequence(self.rows, batch_first=True, padding_value=padding)


def _load_trainer_module(monkeypatch):
    def register_trainer(_name):
        return lambda cls: cls

    trainer_base = types.ModuleType("verl.trainer.ppo.v1.trainer_base")
    trainer_base.register_trainer = register_trainer
    trainer_sync = types.ModuleType("verl.trainer.ppo.v1.trainer_sync")
    trainer_sync.PPOTrainerSync = object
    config_module = types.ModuleType("verl.utils.config")
    config_module.omega_conf_to_dataclass = lambda *args, **kwargs: None
    worker_config = types.ModuleType("verl_omni.workers.config")
    worker_config.OmniModelConfig = object

    for package_name in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1", "verl.utils"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.trainer_base", trainer_base)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.trainer_sync", trainer_sync)
    monkeypatch.setitem(sys.modules, "verl.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "verl_omni.workers.config", worker_config)

    module_path = Path(__file__).parents[3] / "verl_omni" / "trainer" / "omni" / "ray_omni_trainer.py"
    spec = importlib.util.spec_from_file_location("ray_omni_trainer_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colocated_reward_preserves_qi_metadata_and_writes_scores(monkeypatch):
    module = _load_trainer_module(monkeypatch)
    fetched = {
        "prompts": _NestedTokens([[1, 2], [3, 4, 5]]),
        "responses": _NestedTokens([[6, 7], [8]]),
        "raw_prompt": [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]],
        "data_source": ["omnivideo", "omnivideo"],
        "reward_model": [{"ground_truth": "A"}, {"ground_truth": "B"}],
        "extra_info": [
            {"video_path": "/data/one.mp4", "question": "one?"},
            {"video_path": "/data/two.mp4", "question": "two?"},
        ],
    }
    writes = {}

    transfer_queue = types.ModuleType("transfer_queue")

    def kv_batch_get(*, keys, partition_id, select_fields):
        assert keys == ["one", "two"]
        assert partition_id == "train"
        assert set(select_fields) == set(fetched)
        return fetched

    def kv_batch_put(*, keys, partition_id, fields):
        writes.update({"keys": keys, "partition_id": partition_id, "fields": fields})

    transfer_queue.kv_batch_get = kv_batch_get
    transfer_queue.kv_batch_put = kv_batch_put
    monkeypatch.setitem(sys.modules, "transfer_queue", transfer_queue)

    class FakeTensorDict(dict):
        def __init__(self, values, batch_size):
            super().__init__(values)
            self.batch_size = batch_size

    tensordict = types.ModuleType("tensordict")
    tensordict.TensorDict = FakeTensorDict
    monkeypatch.setitem(sys.modules, "tensordict", tensordict)

    class FakeDataProto:
        def __init__(self, batch, non_tensor_batch=None, meta_info=None):
            self.batch = batch
            self.non_tensor_batch = non_tensor_batch or {}
            self.meta_info = meta_info or {}

    protocol = types.ModuleType("verl.protocol")
    protocol.DataProto = FakeDataProto
    monkeypatch.setitem(sys.modules, "verl.protocol", protocol)
    tensordict_utils = types.ModuleType("verl.utils.tensordict_utils")
    tensordict_utils.get_tensordict = lambda values: values
    monkeypatch.setitem(sys.modules, "verl.utils.tensordict_utils", tensordict_utils)
    sys.modules["verl.utils"].tensordict_utils = tensordict_utils

    captured = {}

    class RewardLoopManager:
        def compute_rm_score(self, data):
            captured["input"] = data
            return FakeDataProto(
                batch={"rm_scores": torch.tensor([[0.7, 0.7], [0.4, 0.0]])},
                non_tensor_batch={"format": np.array([1.0, 0.0])},
                meta_info={"reward_extra_keys": ["format"]},
            )

    trainer = object.__new__(module.OmniPPOTrainerSync)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = RewardLoopManager()
    batch = type(
        "Batch",
        (),
        {"keys": ["one", "two"], "partition_id": "train", "__len__": lambda self: 2},
    )()
    result = trainer._compute_reward_colocate(batch)

    assert result is batch
    reward_input = captured["input"]
    assert reward_input.batch["attention_mask"].tolist() == [[1, 1, 0, 1, 1], [1, 1, 1, 1, 0]]
    assert reward_input.non_tensor_batch["reward_model"].shape == (2,)
    assert reward_input.non_tensor_batch["extra_info"][1]["video_path"] == "/data/two.mp4"
    assert writes["keys"] == ["one", "two"]
    assert writes["partition_id"] == "train"
    assert writes["fields"]["format"].tolist() == [1.0, 0.0]
    assert writes["fields"]["rm_scores"].offsets().diff().tolist() == [2, 1]
