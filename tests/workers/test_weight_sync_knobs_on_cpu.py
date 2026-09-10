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
"""Regression pin: weight-sync knobs must be derivable without a rollout role.

With ``actor_rollout_ref.separate=true`` the actor worker's role has no
"rollout", so nothing in the rollout-engine build branch runs for it — yet
``update_weights()`` (nccl / separate-async path) reads ``self.rollout_adapter``
and, with LoRA, ``self.peft_merge``. The v0 diffusion separate e2e crashed with
``AttributeError: 'ActorRolloutRefWorker' object has no attribute
'rollout_adapter'`` when those knobs were initialized only under
``if "rollout" in self.role``. init_model now derives them for every role via
``_init_weight_sync_knobs``; this test pins that derivation, including the
omni-config defaults (no ``rollout_adapter`` field, no lora merge).
"""

from types import SimpleNamespace

from omegaconf import DictConfig

import verl_omni.workers.engine_workers as ew


def _bare_worker(rollout_cfg: dict):
    # The real worker's self.config.rollout is an OmegaConf DictConfig: the
    # knob derivation mixes attribute access (load_format) with .get(...), so
    # the fake must be a DictConfig too, not a SimpleNamespace.
    worker = object.__new__(ew.ActorRolloutRefWorker)
    worker.config = SimpleNamespace(rollout=DictConfig(rollout_cfg))
    return worker


def test_weight_sync_knobs_default_for_config_without_optional_fields():
    # Omni rollout config shape: no rollout_adapter / layered_summon keys.
    rollout_cfg = {"load_format": "safetensors"}
    model_config = SimpleNamespace(lora={})
    worker = _bare_worker(rollout_cfg)

    ew.ActorRolloutRefWorker._init_weight_sync_knobs(worker, model_config)

    assert worker.rollout_adapter == "default"
    assert worker.peft_merge is False
    assert worker.layered_summon is False
    assert worker.base_sync_done is True
    assert worker._zmq_update_seq == 0


def test_weight_sync_knobs_read_diffusion_dual_adapter_and_lora_merge():
    rollout_cfg = {"load_format": "dummy_model_dt", "rollout_adapter": "adapter_b", "layered_summon": True}
    model_config = SimpleNamespace(lora={"merge": True})
    worker = _bare_worker(rollout_cfg)

    ew.ActorRolloutRefWorker._init_weight_sync_knobs(worker, model_config)

    assert worker.rollout_adapter == "adapter_b"
    assert worker.peft_merge is True
    assert worker.layered_summon is True
    assert worker.base_sync_done is False  # dummy load_format skips base sync
