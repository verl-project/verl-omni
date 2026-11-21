# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the R3 rollout/actor router-replay coupling."""

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.omni.trainer_base import validate_rollout_routing_replay


def _config(*, strategy="megatron", actor_cfg=None):
    actor = {"strategy": strategy}
    actor.update(actor_cfg or {})
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": actor,
                "rollout": {"enable_rollout_routing_replay": True},
            }
        }
    )


def test_no_op_without_rollout_routing_replay():
    config = _config()
    config.actor_rollout_ref.rollout.enable_rollout_routing_replay = False
    validate_rollout_routing_replay(config)


def test_accepts_megatron_engine_in_r3_mode():
    validate_rollout_routing_replay(
        _config(actor_cfg={"megatron": {"router_replay": {"mode": "R3"}}}),
    )


def test_rejects_actor_strategy_without_router_replay():
    with pytest.raises(NotImplementedError, match="requires an actor strategy"):
        validate_rollout_routing_replay(_config(strategy="fsdp"))


def test_rejects_disabled_engine_router_replay():
    with pytest.raises(ValueError, match="router_replay.mode='R3'"):
        validate_rollout_routing_replay(
            _config(actor_cfg={"megatron": {"router_replay": {"mode": "disabled"}}}),
        )


def test_rejects_r2_engine_router_replay():
    with pytest.raises(ValueError, match="router_replay.mode='R3'"):
        validate_rollout_routing_replay(
            _config(actor_cfg={"megatron": {"router_replay": {"mode": "R2"}}}),
        )
