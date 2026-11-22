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
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.v1 import trainer_base as trainer_base_module
from verl_omni.trainer.diffusion.v1 import trainer_colocate_async as colocate_async_module
from verl_omni.trainer.diffusion.v1 import trainer_separate_async as separate_async_module
from verl_omni.trainer.diffusion.v1.trainer_colocate_async import PolicyGradientDiffusionTrainerV1ColocateAsync
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync


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


def test_colocate_checkpoint_saves_rollout_before_publishing_step(monkeypatch, tmp_path):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1ColocateAsync)
    trainer.global_steps = 4
    trainer.trainer_mode = "colocate_async"
    trainer._resume_denoising = True
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "v1": {"colocate_async": {"request_timeout_s": 9}},
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    checkpoint = tmp_path / "global_step_4"
    latest = tmp_path / "latest_checkpointed_iteration.txt"
    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=lambda path, *a, **k: Path(path).mkdir(parents=True))
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": 2})
    handoff = {
        "policy_ref": "rollout-step-3",
        "model_fingerprint": "a" * 64,
        "requests": {"open_1_0": "engine-id"},
        "snapshots": [{"engine-id": b"denoise"}],
        "adapters": [b"old-lora"],
    }

    def export(session_keys, timeout, max_bytes):
        assert session_keys == {"open_0_0", "open_1_0"} and timeout == 9
        assert (checkpoint / "actor").is_dir() and (checkpoint / "transfer_queue").is_dir()
        assert not latest.exists()
        return handoff

    trainer.llm_server_manager = SimpleNamespace(
        server_handles=[SimpleNamespace(export_diffusion_checkpoint=SimpleNamespace(remote=export))]
    )
    monkeypatch.setattr(trainer_base_module, "_tq_supports_checkpoint", lambda: True)
    monkeypatch.setattr(trainer_base_module.tq, "save_checkpoint", lambda path, **kw: Path(path).mkdir())
    monkeypatch.setattr(
        colocate_async_module.tq,
        "kv_list",
        lambda partition: {
            partition: {
                "open": {"is_prompt": True, "status": "running"},
                "done": {"is_prompt": True, "status": "finished"},
            }
        },
    )
    monkeypatch.setattr(colocate_async_module.ray, "get", lambda value, **kw: value)

    trainer._save_checkpoint()

    assert msgspec.msgpack.decode((checkpoint / "diffusion_rollout.msgpack").read_bytes()) == {
        "version": 2,
        "step": 4,
        "handoff": handoff,
    }
    assert latest.read_text() == "4"
    trainer._load_rollout_checkpoint(str(checkpoint))
    assert trainer._checkpoint_handoff == handoff
    trainer._checkpoint_handoff = None
    handoff.pop("model_fingerprint")
    (checkpoint / "diffusion_rollout.msgpack").write_bytes(
        msgspec.msgpack.encode({"version": 1, "step": 4, "handoff": handoff})
    )
    trainer._load_rollout_checkpoint(str(checkpoint))
    assert trainer._checkpoint_handoff is None


def test_checkpoint_reissue_keeps_completed_member_and_transports_restored_output(monkeypatch):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1ColocateAsync)
    trainer._resume_denoising = True
    trainer.global_steps = 5
    trainer.trainer_mode = "colocate_async"
    trainer._gen_batch_seq = 0
    trainer.config = OmegaConf.create(
        {
            "trainer": {"v1": {"sampler": {"max_off_policy_threshold": 2}}},
            "actor_rollout_ref": {"rollout": {"n": 3, "seed": None}},
        }
    )
    tag = {"status": "success", "global_steps": 3, "min_global_steps": 3, "max_global_steps": 3}
    items = {
        "open": {"is_prompt": True, "status": "running"},
        "open_0_0": tag,
        "open_2_0": {**tag, "global_steps": 1, "min_global_steps": 1, "max_global_steps": 1},
        "consumed_0_0": tag,
    }
    result = {"version": 3, "log_probs": torch.tensor([0.25])}
    trainer._checkpoint_outputs = {"open_1_0": result}
    batch = tu.get_tensordict({"uid": ["open"], "raw_prompt": ["prompt"], "index": torch.tensor([0])})
    cleared, written, submitted = [], [], []
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=submitted.append)
    monkeypatch.setattr(trainer_base_module, "_tq_supports_checkpoint", lambda: True)
    monkeypatch.setattr(trainer_base_module.tq, "kv_list", lambda partition: {partition: items})
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_get", lambda **kw: batch)
    monkeypatch.setattr(trainer_base_module.tq, "kv_clear", lambda *, keys, partition_id: cleared.extend(keys))
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_put", lambda **kw: written.append(kw))

    assert trainer._reissue_inflight_prompts() == 1
    assert set(cleared) == {"open_2_0", "consumed_0_0"}
    assert "checkpoint_members" not in written[0]["fields"].keys()
    assert submitted[0].get("checkpoint_members")[0].data == {0: 3, 1: result}
    assert trainer._checkpoint_outputs is None


@pytest.mark.parametrize("trainer_mode", ["separate_async", "colocate_async"])
def test_async_checkpoint_saves_and_loads_transfer_queue(monkeypatch, tmp_path, caplog, trainer_mode):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    checkpoint_dir = tmp_path / "global_step_4"
    trainer.global_steps = 4
    trainer.trainer_mode = trainer_mode
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


@pytest.mark.parametrize("trainer_mode", ["separate_async", "colocate_async"])
def test_reissue_restarts_only_inflight_prompt_groups(monkeypatch, trainer_mode):
    pending = "pending"
    running = "running"
    finished = "finished"
    items = {
        pending: {"is_prompt": True, "status": "pending", "global_steps": 2},
        running: {"is_prompt": True, "status": "running", "global_steps": 2},
        finished: {"is_prompt": True, "status": "finished", "global_steps": 2},
        f"{pending}_0_0": {"is_prompt": False, "global_steps": 2},
        f"{running}_0_0": {"is_prompt": False, "global_steps": 2},
        f"{finished}_0_0": {"is_prompt": False, "global_steps": 2, "gen_batch_seq": 51},
        "consumed_0_0": {"is_prompt": False, "global_steps": 2},
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
    trainer.trainer_mode = trainer_mode
    trainer.global_steps = 8
    trainer._gen_batch_seq = 0
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"seed": None}}})
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=submitted.append)

    assert trainer._reissue_inflight_prompts() == 2
    assert set(cleared) == {f"{pending}_0_0", f"{running}_0_0", "consumed_0_0"}
    assert f"{finished}_0_0" not in cleared
    assert len(updated) == 1
    assert updated[0]["keys"] == [pending, running]
    assert updated[0]["tags"] == [
        {"is_prompt": True, "status": "pending", "global_steps": 8},
        {"is_prompt": True, "status": "pending", "global_steps": 8},
    ]
    assert set(updated[0]["fields"].keys()) == {"uid", "raw_prompt", "index", "rollout_seed", "rollout_seed_index"}
    assert updated[0]["fields"]["rollout_seed_index"].tolist() == [0, 1]
    assert submitted == [prompt_batch]
    assert trainer._gen_batch_seq == 52


@pytest.mark.parametrize("trainer_mode", ["separate_async", "colocate_async"])
def test_reissue_restamps_legacy_rollout_seed_for_the_current_step(monkeypatch, trainer_mode):
    # The TQ round trip keeps tensor fields only: ``rollout_seed`` is gone and
    # ``rollout_seed_index`` still holds the positions from the original step.
    """Legacy prompts gain current seeds without colliding with subsequent warmup prompts."""
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
    trainer.trainer_mode = trainer_mode
    trainer.global_steps = 8
    trainer._gen_batch_seq = 0
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 2}, "actor_rollout_ref": {"rollout": {"seed": 7}}})
    trainer._rollout_seed_step = None
    trainer._rollout_seed_next_index = 0
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=submitted.append)

    assert trainer._reissue_inflight_prompts() == 2
    assert submitted[0]["rollout_seed"].tolist() == [7 + 8 - 1] * 2
    assert [int(v) for v in submitted[0]["rollout_seed_index"]] == [0, 1]
    assert [int(v) for v in updated[0]["fields"]["rollout_seed_index"]] == [0, 1]

    # Warmup prompts submitted after the reissue in the same step take the next positions.
    trainer.train_dataloader_it = iter([{"raw_prompt": np.array(["p0", "p1"], dtype=object)}])
    trainer._gen_batch_seq = 0
    trainer._effective_generation_batch_size = 2
    warmup = trainer._next_train_batch()
    assert warmup["rollout_seed"].tolist() == [7 + 8 - 1] * 2
    assert [int(v) for v in warmup["rollout_seed_index"]] == [2, 3]


@pytest.mark.parametrize(
    ("statuses", "expected_submissions"),
    [
        ([], {"separate_async": [("batch", 2), ("batch", 2)], "colocate_async": [("prompts", 4)]}),
        (["pending"] * 4, {"separate_async": [], "colocate_async": []}),
        (["finished"], {"separate_async": [("batch", 2), ("prompts", 1)], "colocate_async": [("prompts", 3)]}),
        (["running"] * 3, {"separate_async": [("prompts", 1)], "colocate_async": [("prompts", 1)]}),
        (["running"] * 5, {"separate_async": [], "colocate_async": []}),
    ],
)
@pytest.mark.parametrize("trainer_mode", ["separate_async", "colocate_async"])
def test_warmup_tops_up_restored_prompt_groups(monkeypatch, statuses, expected_submissions, trainer_mode):
    """Finished groups count toward warmup in both async modes."""
    cls = (
        PolicyGradientDiffusionTrainerV1SeparateAsync
        if trainer_mode == "separate_async"
        else PolicyGradientDiffusionTrainerV1ColocateAsync
    )
    trainer = object.__new__(cls)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "trainer": {"v1": {trainer_mode: {"num_warmup_batches": 2}}},
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
    assert submitted == expected_submissions[trainer_mode]
