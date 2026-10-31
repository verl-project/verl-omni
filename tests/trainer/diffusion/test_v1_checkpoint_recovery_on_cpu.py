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
"""CPU regression tests for diffusion V1 async checkpoint recovery."""

import logging
import uuid
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.v1 import trainer_base as trainer_base_module
from verl_omni.trainer.diffusion.v1 import trainer_separate_async as separate_async_module
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync

_REAL_TQ_SUPPORTS_CHECKPOINT = trainer_base_module._tq_supports_checkpoint
requires_tq_checkpoint = pytest.mark.skipif(
    not _REAL_TQ_SUPPORTS_CHECKPOINT(),
    reason="TransferQueue >= 0.1.9 with checkpoint APIs is required",
)


@pytest.fixture(scope="module")
def initialized_tq():
    tq.init()
    yield
    tq.close()


def test_transfer_queue_checkpoint_guard(monkeypatch):
    checkpoint_api = lambda *args, **kwargs: None
    monkeypatch.setattr(trainer_base_module.tq, "save_checkpoint", checkpoint_api, raising=False)
    monkeypatch.setattr(trainer_base_module.tq, "load_checkpoint", checkpoint_api, raising=False)

    monkeypatch.setattr(trainer_base_module.tq, "__version__", "0.1.8", raising=False)
    assert trainer_base_module._tq_supports_checkpoint() is False

    monkeypatch.setattr(trainer_base_module.tq, "__version__", "0.1.9", raising=False)
    assert trainer_base_module._tq_supports_checkpoint() is True

    monkeypatch.setattr(trainer_base_module.tq, "load_checkpoint", None)
    assert trainer_base_module._tq_supports_checkpoint() is False


def test_async_checkpoint_saves_and_loads_transfer_queue(monkeypatch, tmp_path, caplog):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    checkpoint_dir = tmp_path / "global_step_4"
    trainer.global_steps = 4
    trainer.trainer_mode = "separate_async"
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "max_actor_ckpt_to_keep": None,
                "resume_mode": "resume_path",
                "resume_from_path": str(checkpoint_dir),
                "del_local_ckpt_after_load": False,
            }
        }
    )
    trainer.actor_rollout_wg = SimpleNamespace(
        save_checkpoint=lambda *args, **kwargs: None,
        load_checkpoint=lambda *args, **kwargs: None,
    )
    loaded_dataloader_states = []
    trainer.train_dataloader = SimpleNamespace(
        state_dict=lambda: {"cursor": 2},
        load_state_dict=loaded_dataloader_states.append,
    )

    saved = []
    loaded = []

    def save_transfer_queue(path, **kwargs):
        Path(path).mkdir(parents=True)
        saved.append((path, kwargs))

    monkeypatch.setattr(trainer_base_module, "_tq_supports_checkpoint", lambda: True)
    monkeypatch.setattr(trainer_base_module.tq, "save_checkpoint", save_transfer_queue, raising=False)
    monkeypatch.setattr(trainer_base_module.tq, "load_checkpoint", loaded.append, raising=False)

    trainer._save_checkpoint()
    trainer._load_checkpoint()

    tq_checkpoint = str(checkpoint_dir / "transfer_queue")
    assert saved == [(tq_checkpoint, {"metadata": {"global_steps": 4}})]
    assert loaded == [tq_checkpoint]
    assert loaded_dataloader_states == [{"cursor": 2}]

    Path(tq_checkpoint).rmdir()
    with caplog.at_level(logging.WARNING, logger=trainer_base_module.logger.name):
        trainer._load_checkpoint()
    assert "No TransferQueue state at" in caplog.text

    caplog.clear()
    monkeypatch.setattr(trainer_base_module, "_tq_supports_checkpoint", lambda: False)
    with caplog.at_level(logging.WARNING, logger=trainer_base_module.logger.name):
        trainer._load_checkpoint()
    assert "TransferQueue checkpoint recovery is unavailable" in caplog.text


def test_reissue_restarts_only_inflight_prompt_groups(monkeypatch):
    pending = "pending"
    running = "running"
    finished = "finished"
    items = {
        pending: {"is_prompt": True, "status": "pending", "global_steps": 2},
        running: {"is_prompt": True, "status": "running", "global_steps": 2},
        finished: {"is_prompt": True, "status": "finished", "global_steps": 2},
        f"{pending}_0_0": {"is_prompt": False, "global_steps": 2},
        f"{running}_0_0": {"is_prompt": False, "global_steps": 2},
        f"{finished}_0_0": {"is_prompt": False, "global_steps": 2},
    }
    prompt_batch = tu.get_tensordict(
        {
            "uid": [pending, running],
            "raw_prompt": ["prompt-pending", "prompt-running"],
            "index": torch.tensor([0, 1]),
        }
    )
    tu.assign_non_tensor_data(prompt_batch, "global_steps", 2)
    cleared = []
    updated = []
    submitted = []

    monkeypatch.setattr(trainer_base_module, "_tq_supports_checkpoint", lambda: True)
    monkeypatch.setattr(
        trainer_base_module.tq,
        "kv_list",
        lambda partition_id: {partition_id: deepcopy(items)},
    )
    monkeypatch.setattr(
        trainer_base_module.tq,
        "kv_batch_get",
        lambda *, keys, partition_id: prompt_batch,
    )
    monkeypatch.setattr(
        trainer_base_module.tq,
        "kv_clear",
        lambda *, keys, partition_id: cleared.extend(keys),
    )
    monkeypatch.setattr(
        trainer_base_module.tq,
        "kv_batch_put",
        lambda **kwargs: updated.append(kwargs),
    )

    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.global_steps = 8
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"seed": None}}})
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=submitted.append)

    assert trainer._reissue_inflight_prompts() == 2
    assert set(cleared) == {f"{pending}_0_0", f"{running}_0_0"}
    assert f"{finished}_0_0" not in cleared
    assert len(updated) == 1
    assert updated[0]["keys"] == [pending, running]
    assert updated[0]["tags"] == [
        {"is_prompt": True, "status": "pending", "global_steps": 8},
        {"is_prompt": True, "status": "pending", "global_steps": 8},
    ]
    assert set(updated[0]["fields"].keys()) == {"uid", "raw_prompt", "index"}
    assert submitted == [prompt_batch]


def test_reissue_restamps_rollout_seed_for_the_current_step(monkeypatch):
    # The TQ round trip keeps tensor fields only: ``rollout_seed`` is gone and
    # ``rollout_seed_index`` still holds the positions from the original step.
    inflight = ["pending", "running"]
    items = {uid: {"is_prompt": True, "status": "pending", "global_steps": 2} for uid in inflight}
    prompt_batch = tu.get_tensordict(
        {
            "uid": inflight,
            "raw_prompt": ["prompt-pending", "prompt-running"],
            "index": torch.tensor([0, 0]),
            "rollout_seed_index": torch.tensor([4, 5]),
        }
    )
    updated = []
    submitted = []

    monkeypatch.setattr(trainer_base_module, "_tq_supports_checkpoint", lambda: True)
    monkeypatch.setattr(trainer_base_module.tq, "kv_list", lambda partition_id: {partition_id: deepcopy(items)})
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_get", lambda *, keys, partition_id: prompt_batch)
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_put", lambda **kwargs: updated.append(kwargs))

    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.global_steps = 8
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 2}, "actor_rollout_ref": {"rollout": {"seed": 7}}})
    trainer._rollout_seed_step = None
    trainer._rollout_seed_next_index = 0
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=submitted.append)

    assert trainer._reissue_inflight_prompts() == 2
    assert int(submitted[0]["rollout_seed"]) == 7 + 8 - 1
    assert [int(v) for v in submitted[0]["rollout_seed_index"]] == [0, 1]
    assert [int(v) for v in updated[0]["fields"]["rollout_seed_index"]] == [0, 1]

    # Warmup prompts submitted after the reissue in the same step take the next positions.
    trainer.train_dataloader_it = iter([{"raw_prompt": np.array(["p0", "p1"], dtype=object)}])
    trainer._gen_batch_seq = 0
    trainer._effective_generation_batch_size = 2
    warmup = trainer._next_train_batch()
    assert int(warmup["rollout_seed"]) == 7 + 8 - 1
    assert [int(v) for v in warmup["rollout_seed_index"]] == [2, 3]


@requires_tq_checkpoint
def test_transfer_queue_checkpoint_roundtrip_and_reissue(initialized_tq, tmp_path):
    partition_id = f"test-{uuid.uuid4().hex}"
    pending, finished = uuid.uuid4().hex, uuid.uuid4().hex
    prompt_batch = tu.get_tensordict(
        {
            "uid": [pending, finished],
            "raw_prompt": ["pending prompt", "finished prompt"],
            "index": torch.tensor([0, 1]),
            "rollout_seed_index": torch.tensor([4, 5]),
        }
    )
    tq.kv_batch_put(
        keys=[pending, finished],
        partition_id=partition_id,
        fields=prompt_batch,
        tags=[
            {"is_prompt": True, "status": "running", "global_steps": 3},
            {"is_prompt": True, "status": "finished", "global_steps": 3},
        ],
    )
    pending_trajectory = f"{pending}_0_0"
    finished_trajectory = f"{finished}_0_0"
    for key in (pending_trajectory, finished_trajectory):
        tq.kv_put(
            key=key,
            partition_id=partition_id,
            fields={"input_ids": torch.tensor([1, 2, 3])},
            tag={"is_prompt": False, "global_steps": 3},
        )

    checkpoint_dir = tmp_path / "transfer_queue"
    tq.save_checkpoint(checkpoint_dir, metadata={"global_steps": 3})
    tq.kv_clear(keys=list(tq.kv_list(partition_id)[partition_id]), partition_id=partition_id)
    tq.load_checkpoint(checkpoint_dir)

    submitted = []
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = "separate_async"
    trainer.global_steps = 4
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"seed": 7}}})
    trainer._rollout_seed_step = None
    trainer._rollout_seed_next_index = 0
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=submitted.append)
    try:
        assert trainer._reissue_inflight_prompts(partition_id) == 1
        items = tq.kv_list(partition_id)[partition_id]
        assert items[pending]["status"] == "pending"
        assert items[finished]["status"] == "finished"
        assert pending_trajectory not in items
        assert finished_trajectory in items
        assert list(submitted[0]["uid"]) == [pending]
        assert list(submitted[0]["raw_prompt"]) == ["pending prompt"]
        assert int(submitted[0]["global_steps"]) == 4
        assert int(submitted[0]["rollout_seed"]) == 7 + 4 - 1
        assert [int(v) for v in submitted[0]["rollout_seed_index"]] == [0]
        persisted_prompt = tq.kv_batch_get(keys=[pending], partition_id=partition_id)
        assert list(persisted_prompt["uid"]) == [pending]
        assert list(persisted_prompt["raw_prompt"]) == ["pending prompt"]
        assert int(persisted_prompt["index"][0]) == 0
        assert [int(v) for v in persisted_prompt["rollout_seed_index"]] == [0]
    finally:
        tq.kv_clear(keys=list(tq.kv_list(partition_id)[partition_id]), partition_id=partition_id)


@pytest.mark.parametrize(
    ("statuses", "expected_submissions"),
    [
        (["pending"] * 4, []),
        (["finished"], [("batch", 2), ("batch", 2)]),
        (["running"] * 3, [("prompts", 1)]),
    ],
)
def test_warmup_tops_up_inflight_prompts(monkeypatch, statuses, expected_submissions):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "trainer": {"v1": {"separate_async": {"num_warmup_batches": 2}}},
        }
    )
    # Skip the hybrid reclaim branch; this test only covers warmup top-up math.
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=True)
    submitted = []
    trainer._add_batch_to_generate = lambda: submitted.append(("batch", 2))
    trainer._add_prompts_to_generate = lambda count: submitted.append(("prompts", count))

    queue_items = {
        f"prompt-{idx}": {"is_prompt": True, "status": status, "global_steps": 3} for idx, status in enumerate(statuses)
    }
    monkeypatch.setattr(separate_async_module.tq, "kv_list", lambda partition_id: {partition_id: queue_items})
    trainer.on_train_begin()
    assert submitted == expected_submissions
