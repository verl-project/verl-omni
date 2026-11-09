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
"""Independent policy snapshots and resume ordering, using tiny CPU parameters."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

from tests.workers.crd_fsdp_checkpoint import Engine
from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.trainer.diffusion.diffusion_trainer_utils import old_policy_decay
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync


def make_trainer(engine):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1Sync)
    trainer.actor_rollout_wg = engine
    trainer._has_old_adapter = True
    trainer._loss_fn = get_diffusion_loss_fn("crd")
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"lora_rank": 2, "policy_state_adapters": ["default", "old", "moving_reference"]}
            },
            "algorithm": {
                "old_policy_decay": 0.5,
                "old_policy_decay_schedule": "copy",
                "old_policy_update_interval": 2,
                "reference_policy_decay": 0.75,
                "reference_policy_decay_schedule": "copy",
                "reference_policy_update_interval": 1,
            },
        }
    )
    return trainer


def test_ema_snapshots_are_independent_and_restore_trainability():
    engine = Engine()
    trainer = make_trainer(engine)
    trainer.global_steps = 1
    metrics = trainer._update_policy_adapters()
    assert metrics["old_policy/update_applied"] == 0
    assert metrics["reference_policy/update_applied"] == 1
    torch.testing.assert_close(engine.module.lora_A["old"], torch.tensor([0.0]))
    torch.testing.assert_close(engine.module.lora_A["moving_reference"], torch.tensor([2.5]))
    trainer.global_steps = 2
    trainer._update_policy_adapters()
    torch.testing.assert_close(engine.module.lora_A["old"], torch.tensor([2.0]))
    torch.testing.assert_close(engine.module.lora_A["moving_reference"], torch.tensor([2.875]))
    assert engine.module.active_adapter == "default"
    assert [k for k, v in engine.module.lora_A.items() if v.requires_grad] == ["default"]


def test_serialized_snapshots_resume_without_reinitialization(tmp_path):
    engine = Engine()
    trainer = make_trainer(engine)
    trainer.global_steps = 2
    trainer._update_policy_adapters()
    saved = {k: v.clone() for k, v in engine.module.state_dict().items()}
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    resumed = make_trainer(Engine())
    # Run the real setup/init order, stubbing only cluster and model startup.
    for method in (
        "_init_tokenizer",
        "_init_dataloader",
        "_init_dump_executor",
        "_init_resource_pool_mgr",
        "_normalize_dynamic_bsz_config",
        "_init_colocated_workers",
        "_init_online_rollout_stack",
        "on_init_end",
    ):
        setattr(resumed, method, MagicMock())
    resumed.resource_pool_manager = SimpleNamespace(create_resource_pool=lambda: None, resource_pool_dict={})
    resumed.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None)
    resumed._load_checkpoint = lambda: resumed.actor_rollout_wg.module.load_state_dict(
        torch.load(path, weights_only=True)
    )
    resumed.init()
    for key, value in saved.items():
        torch.testing.assert_close(resumed.actor_rollout_wg.module.state_dict()[key], value)
    assert not torch.equal(saved["lora_A.default"], saved["lora_A.old"])


def test_fresh_snapshots_copy_current_weights():
    engine = Engine()
    make_trainer(engine)._initialize_policy_adapters()
    for name in ("old", "moving_reference"):
        torch.testing.assert_close(engine.module.lora_A[name], engine.module.lora_A["default"])
    trainer = make_trainer(MagicMock())
    trainer._loss_fn = SimpleNamespace(extra_ema_adapters=(("other_reference", "reference_policy"),))
    trainer._initialize_policy_adapters()
    assert [call.kwargs["target"] for call in trainer.actor_rollout_wg.copy_adapter.call_args_list] == [
        "other_reference",
        "old",
    ]
    trainer.global_steps = 1
    assert trainer._update_policy_adapters()["reference_policy/update_applied"] == 1
    trainer.actor_rollout_wg.ema_update_adapter.assert_called_once_with(
        source="default", target="other_reference", decay=0.75
    )


def test_independent_decay_schedules():
    assert old_policy_decay(0, "linear_from_0_25_to_0_5") == 0.25
    assert old_policy_decay(100, "linear_from_0_25_to_0_5") == 0.35
    assert old_policy_decay(100, "linear_to_0_5") == 0.1
    assert old_policy_decay(1000, "linear_from_0_25_to_0_5") == 0.5
