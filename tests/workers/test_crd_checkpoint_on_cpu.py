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
"""CRD named-adapter checkpoint restoration and EMA verification."""

import pytest
import torch
from peft import LoraConfig
from transformers import PretrainedConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

from tests.workers.crd_fsdp_checkpoint import Engine, assert_adapter_updates


@pytest.mark.parametrize("lora_only", [False, True])
def test_real_checkpoint_manager_restores_snapshots_optimizer_scheduler_and_rng(tmp_path, lora_only):
    """Exercise verl's saver/loader, not a stubbed load_state_dict call."""
    from tests.workers.crd_fsdp_checkpoint import assert_same_state, local_state

    # A CPU process group exercises the same checkpoint manager without FSDP sharding.
    assert not torch.distributed.is_initialized()
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        engine = Engine()
        engine.module.config = PretrainedConfig()
        engine.module.can_generate = lambda: False
        engine.module.peft_config = {"default": LoraConfig(r=2, lora_alpha=4)}
        optimizer = torch.optim.AdamW(engine.module.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        def update():
            optimizer.zero_grad()
            engine.module.lora_A["default"].square().sum().backward()
            optimizer.step()
            scheduler.step()

        def state():
            return local_state((engine.module.state_dict(), optimizer.state_dict(), scheduler.state_dict()))

        update()
        engine.ema_update_adapter(target="old", decay=0.2)
        engine.ema_update_adapter(target="moving_reference", decay=0.7)
        saved = state()
        manager = FSDPCheckpointManager(
            engine.module, optimizer, scheduler, checkpoint_config=CheckpointConfig(save_lora_only=lora_only)
        )
        path = str(tmp_path / "checkpoint")
        manager.save_checkpoint(path, global_step=1)
        expected_random = torch.rand(3)
        update()
        continued = state()
        manager.load_checkpoint(path, del_local_after_load=False)
        assert_same_state(state(), saved)
        torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
        update()
        assert_same_state(state(), continued)
        assert not torch.equal(saved[0]["lora_A.old"], saved[0]["lora_A.moving_reference"])
    finally:
        torch.distributed.destroy_process_group()


def test_independent_ema_assertions_detect_wrong_reference():
    before = {f"layer.lora_A.{name}.weight": torch.zeros(2) for name in ("default", "old", "moving_reference")}
    trained = {key: value.clone() for key, value in before.items()}
    trained["layer.lora_A.default.weight"].fill_(1)
    decays = {"old": 0.2, "moving_reference": 0.7}
    refreshed = {key: value.clone() for key, value in trained.items()}
    refreshed["layer.lora_A.old.weight"].fill_(0.8)
    refreshed["layer.lora_A.moving_reference.weight"].fill_(0.3)
    assert_adapter_updates(before, trained, refreshed, decays)
    refreshed["layer.lora_A.moving_reference.weight"].fill_(0.8)
    with pytest.raises(AssertionError):
        assert_adapter_updates(before, trained, refreshed, decays)
