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
"""NFT V1 wiring: preserve sample associations and refresh the rollout adapter."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from transfer_queue import KVBatchMeta
from verl import DataProto

import verl_omni


def make_nft_trainer():
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    config_dir = Path(verl_omni.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "trainer.use_v1=true",
                "trainer.v1.trainer_mode=sync",
                "algorithm.trainer_type=direct_preference",
                "algorithm.sample_source=online",
                "algorithm.old_policy_update_interval=2",
                "algorithm.old_policy_decay_schedule=delayed_linear_to_0_999",
                "algorithm.timestep_fraction=1.0",
                "actor_rollout_ref.model.algorithm=diffusion_nft",
                "actor_rollout_ref.model.model_type=diffusion_nft_model",
                'actor_rollout_ref.model.policy_state_adapters=["default","old"]',
                "actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft",
                "actor_rollout_ref.rollout.rollout_adapter=old",
                "actor_rollout_ref.rollout.calculate_log_probs=false",
            ],
        )
    return PolicyGradientDiffusionTrainerV1Sync(config)


def test_nft_initializes_old_adapter_before_first_weight_sync(monkeypatch):
    trainer = make_nft_trainer()
    events = []
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.copy_adapter.side_effect = lambda **kwargs: events.append("copy")
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append("sync"))
    monkeypatch.setattr(trainer, "_setup", lambda: None)

    trainer.init()

    trainer.actor_rollout_wg.copy_adapter.assert_called_once_with(source="default", target="old")
    assert events == ["copy", "sync"]


@pytest.mark.parametrize("step,refresh", [(1, None), (2, "copy"), (76, "ema")])
def test_nft_tq_rows_and_old_policy_schedule(monkeypatch, step, refresh):
    trainer = make_nft_trainer()
    trainer.global_steps = step
    trainer.timing_raw = {}
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
    trainer.actor_rollout_wg = MagicMock()
    events = []
    trainer.actor_rollout_wg.copy_adapter.side_effect = lambda **kwargs: events.append("copy")
    trainer.actor_rollout_wg.ema_update_adapter.side_effect = lambda **kwargs: events.append("ema")
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append("sync"))

    # Interleave prompt groups so grouping cannot accidentally rely on row adjacency.
    keys = ["p0_0_0", "p1_0_0", "p0_1_0", "p1_1_0"]
    uids = ["p0", "p1", "p0", "p1"]
    latents = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    timesteps = torch.tensor([[900, 600, 300]]).repeat(4, 1)
    rewards = torch.tensor([[1.0], [0.2], [0.0], [0.8]])
    rollout = {
        "uid": uids,
        "sample_id": keys,
        "latents_clean": latents,
        "train_timesteps": timesteps,
        "rm_scores": rewards,
    }
    read = MagicMock(return_value=rollout)
    write = MagicMock()
    monkeypatch.setattr("verl_omni.trainer.diffusion.v1.tq_utils.tq.kv_batch_get", read)
    monkeypatch.setattr("verl_omni.trainer.diffusion.v1.tq_utils.tq.kv_batch_put", write)

    def forbidden(*args, **kwargs):
        pytest.fail("NFT must use final latents without PPO log-probs or trainer-side DPO reference inference")

    monkeypatch.setattr(trainer, "_compute_old_log_prob", forbidden)
    monkeypatch.setattr(trainer, "_compute_ref_noise_pred", forbidden)
    captured = {}

    def update(data):
        events.append("actor")
        captured["data"] = data
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/nft_loss": 0.1}})

    monkeypatch.setattr(trainer, "_update_actor", update)
    meta = KVBatchMeta(partition_id="train", keys=keys, tags=[{"is_padding": False}] * 4)
    metrics = {}

    assert trainer._train_sampled_batch(metrics, {}, meta) is meta
    trainer.on_step_end()

    read.assert_called_once_with(keys=keys, partition_id="train")
    sent = captured["data"]
    assert list(sent.non_tensor_batch["uid"]) == uids
    assert list(sent.non_tensor_batch["sample_id"]) == keys
    torch.testing.assert_close(sent.batch["latents_clean"], latents)
    torch.testing.assert_close(sent.batch["train_timesteps"].sort(dim=1).values, timesteps.sort(dim=1).values)
    torch.testing.assert_close(sent.batch["sample_level_scores"], rewards)
    assert sent.batch["reward_prob"].shape == (4, 3)
    assert (sent.batch["reward_prob"][[0, 3]] > 0.5).all()
    assert (sent.batch["reward_prob"][[1, 2]] < 0.5).all()
    assert "old_log_probs" not in sent.batch
    assert "ref_noise_pred" not in sent.batch
    assert write.call_args.kwargs["keys"] == keys
    torch.testing.assert_close(write.call_args.kwargs["fields"]["sample_level_scores"], rewards)
    assert metrics["old_policy/update_applied"] == float(refresh is not None)
    assert events == ["actor", *([refresh] if refresh else []), "sync"]
    if refresh == "copy":
        trainer.actor_rollout_wg.copy_adapter.assert_called_once_with(source="default", target="old")
        trainer.actor_rollout_wg.ema_update_adapter.assert_not_called()
    elif refresh == "ema":
        trainer.actor_rollout_wg.ema_update_adapter.assert_called_once_with(
            source="default", target="old", decay=pytest.approx(0.0075)
        )
        trainer.actor_rollout_wg.copy_adapter.assert_not_called()
    else:
        trainer.actor_rollout_wg.copy_adapter.assert_not_called()
        trainer.actor_rollout_wg.ema_update_adapter.assert_not_called()
