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
"""Colocated diffusion lifecycle and configuration contracts without GPUs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

import verl_omni
from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls
from verl_omni.trainer.diffusion.v1 import trainer_base as base_module
from verl_omni.trainer.diffusion.v1 import trainer_colocate_async as module
from verl_omni.trainer.diffusion.v1.trainer_colocate_async import PolicyGradientDiffusionTrainerV1ColocateAsync


def _config(*overrides):
    directory = Path(verl_omni.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(
            config_name="diffusion_trainer",
            overrides=[
                "trainer.v1.trainer_mode=colocate_async",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                *overrides,
            ],
        )


def test_registered_trainer_uses_async_groups_and_exact_refill():
    trainer = get_diffusion_trainer_cls("colocate_async")(_config("+data.gen_batch_size=4"))
    assert isinstance(trainer, PolicyGradientDiffusionTrainerV1ColocateAsync)
    assert isinstance(trainer.replay_buffer, ReplayBufferAsync)
    assert trainer.parameter_sync_step == 1
    assert trainer._generation_batch_size() == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("actor_rollout_ref.rollout.nnodes=1", "colocated rollout"),
        ("actor_rollout_ref.rollout.checkpoint_engine.backend=nccl", "backend=naive"),
        ("actor_rollout_ref.rollout.free_cache_engine=false", "reclaim rollout GPU memory"),
        ("trainer.v1.colocate_async.num_warmup_batches=-1", "nonnegative integer"),
        ("trainer.v1.colocate_async.num_warmup_batches=1.5", "nonnegative integer"),
        ("+trainer.v1.colocate_async.parameter_sync_step=2", "one parameter sync"),
        ("actor_rollout_ref.model.algorithm=diffusion_nft", "FlowGRPO"),
        ("data.train_batch_size=0", "positive integer"),
        ("data.train_batch_size=-2", "positive integer"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size=0", "positive integer"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size=1.5", "positive integer"),
        ("data.train_batch_size=3", "divisible by ppo_mini_batch_size"),
    ],
)
def test_invalid_config_fails_before_workers_start(override, message):
    with pytest.raises(ValueError, match=message):
        PolicyGradientDiffusionTrainerV1ColocateAsync(_config(override))


def _remote(fn):
    return SimpleNamespace(remote=fn)


@pytest.mark.parametrize("profile", [False, True])
def test_routing_stays_closed_from_abort_through_weight_commit(monkeypatch, profile):
    trainer = PolicyGradientDiffusionTrainerV1ColocateAsync(_config())
    events = []
    trainer.global_steps = 3
    trainer.curr_step_profile = profile
    trainer._stop_rollout_profiling = lambda: events.append("stop profile")
    trainer.timing_raw = {}
    trainer.llm_server_manager = SimpleNamespace(
        server_addresses=["server"],
        server_handles=["handle"],
        global_load_balancer=SimpleNamespace(
            remove_servers=_remote(lambda servers: events.append(("remove", servers))),
            add_servers=_remote(lambda servers: events.append(("add", servers))),
        ),
    )
    trainer.checkpoint_manager = SimpleNamespace(
        abort_replicas=lambda: events.append("abort"),
        sleep_replicas=lambda: events.append("sleep"),
        update_weights=lambda step: events.append(("commit", step)),
        resume_generation_replicas=lambda: events.append("resume"),
    )
    monkeypatch.setattr(module.ray, "get", lambda result: result)
    trainer.on_sample_end()
    events.append("actor update")
    trainer.on_step_end()
    assert events == [
        ("remove", ["server"]),
        "abort",
        "sleep",
        *(["stop profile"] if profile else []),
        "actor update",
        ("commit", 3),
        "resume",
        ("add", {"server": "handle"}),
    ]

    # A failed sync never reopens routing.
    events.clear()
    trainer.checkpoint_manager.update_weights = lambda step: (_ for _ in ()).throw(RuntimeError("sync failed"))
    with pytest.raises(RuntimeError, match="sync failed"):
        trainer.on_step_end()
    assert events == []


@pytest.mark.parametrize(("queued", "expected"), [(0, [8]), (3, [5]), (8, []), (10, [])])
def test_warmup_tops_up_restored_groups(monkeypatch, queued, expected):
    trainer = PolicyGradientDiffusionTrainerV1ColocateAsync(_config("trainer.v1.colocate_async.num_warmup_batches=2"))
    items = {f"p{i}": {"is_prompt": True, "status": "finished" if i % 2 else "running"} for i in range(queued)}
    items["p0_0_0"] = {"status": "success"}
    monkeypatch.setattr(module.tq, "kv_list", lambda partition: {partition: items})
    added = []
    trainer._add_prompts_to_generate = added.append
    trainer.on_train_begin()
    assert added == expected


def test_validation_reclaims_rollout_before_colocated_reward(monkeypatch):
    trainer = PolicyGradientDiffusionTrainerV1ColocateAsync(_config("reward.reward_model.enable=true"))
    trainer.global_steps = 0
    trainer._gen_batch_seq = 0
    trainer.val_dataloader = [{"raw_prompt": [[{"role": "user", "content": "prompt"}]]}]
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=lambda batch: None)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    batch = KVBatchMeta(partition_id="val", keys=["sample_0_0"], tags=[{}])
    trainer.replay_buffer.sample = lambda **kwargs: (batch, {})
    events = []
    trainer.on_sample_end = lambda: events.append("abort/sleep")
    trainer.on_step_end = lambda: events.append("commit/resume")
    data = DataProto.from_dict(tensors={"prompts": torch.ones(1, 2), "responses": torch.ones(1, 3, 2, 2)})
    trainer._compute_reward_colocate = lambda value: (events.append("reward"), value)[1]
    monkeypatch.setattr(base_module, "canonicalize_diffusion_tq_meta", lambda value: value)
    monkeypatch.setattr(base_module, "diffusion_tq_batch_to_dataproto", lambda *args, **kwargs: data)
    monkeypatch.setattr(base_module.tq, "kv_batch_put", lambda **kwargs: None)

    class StopAfterReward(Exception):
        pass

    def decode(*args, **kwargs):
        raise StopAfterReward

    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=decode)
    with pytest.raises(StopAfterReward):
        trainer._validate()
    assert events == ["abort/sleep", "reward", "commit/resume"]
