# Copyright 2026 verl-omni contributors
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

"""Fail closed for execution modes that have not received GPU qualification."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_omni.pipelines.proximal_reuse import rpc, strategy
from verl_omni.pipelines.proximal_reuse.execution import megatron_proximal_execution


def _engine():
    config = SimpleNamespace(
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        infer_micro_batch_size_per_gpu=2,
        forward_only=False,
        use_fused_kernels=False,
        dynamic_context_parallel=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=4,
        tensor_model_parallel_size=4,
    )
    model = SimpleNamespace(
        model_type="test",
        mtp=SimpleNamespace(enable=False),
        lora={"rank": 0},
        lora_rank=0,
        lora_adapter_path=None,
        hf_config=SimpleNamespace(to_dict=lambda: {}),
    )
    transformer = SimpleNamespace(attention_dropout=0.0, hidden_dropout=0.0, num_moe_experts=0)
    return SimpleNamespace(
        engine_config=config,
        model_config=model,
        tf_config=transformer,
        module=[torch.nn.Linear(2, 2)],
        enable_routing_replay=False,
        _qat_enabled=False,
        peft_cls=None,
        is_value_model=False,
    )


def test_qualified_synthetic_layout_is_inspected():
    result = megatron_proximal_execution(_engine())
    assert result["model_type"] == "test"
    assert result["checked_controls"]["expert_model_parallel_size"] == 1


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("expert_model_parallel_size", 4, "Expert parallel"),
        ("context_parallel_size", 2, "Pipeline/context"),
        ("pipeline_model_parallel_size", 2, "Pipeline/context"),
        ("use_dynamic_bsz", True, "microbatch"),
        ("use_fused_kernels", True, "fused-output"),
    ],
)
def test_unqualified_execution_is_rejected(attribute, value, message):
    engine = _engine()
    setattr(engine.engine_config, attribute, value)
    with pytest.raises(ValueError, match=message):
        megatron_proximal_execution(engine)


def test_ep4_qualification_falls_back_to_native_old_before_preparation(monkeypatch):
    engine = _engine()
    engine.engine_config.expert_model_parallel_size = 4
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine), enable_routing_replay=False)
    monkeypatch.setattr(rpc, "_check_actual_loss", lambda worker: None)
    # A fake engine cannot satisfy the production exact-type check; retain the
    # actual execution guard while bypassing that unrelated check in this test.
    monkeypatch.setattr(rpc, "_engine_digest", megatron_proximal_execution)
    report = rpc.qualify_actor(worker)
    assert report["supported"] is False and "Expert parallel" in report["reason"]

    config = SimpleNamespace(trainer=SimpleNamespace(v1={"reuse_proximal_actor_forward": True}, critic_warmup=0))
    trainer = SimpleNamespace(
        config=config, global_steps=1, actor_rollout_wg=SimpleNamespace(_query_dispatch_info=lambda _: [0])
    )
    candidate = strategy.ProximalActorReuse(trainer)
    monkeypatch.setattr(candidate, "_check_proximal_algorithm", lambda batch: None)
    monkeypatch.setattr(candidate, "_prepare_proximal_actor", lambda batch: pytest.fail("EP4 must not prepare"))
    monkeypatch.setattr(strategy.worker_calls, "control", lambda group, operation: [report])
    batch, metrics, calls = object(), {}, []
    result = candidate._compute_old_log_prob(batch, metrics, lambda data, _: calls.append("native") or data)
    assert result is batch and calls == ["native"] and metrics["training/proximal_anchor_reuse"] == 0


@pytest.mark.parametrize(
    "correction",
    [
        {"rollout_is": "token"},
        {"rollout_rs": "sequence"},
        {"bypass_mode": True},
        {"rollout_is": ""},
        {"rollout_rs": ""},
        {"rollout_is": False},
        {"rollout_rs": False},
    ],
)
def test_old_dependent_rollout_correction_falls_back_before_any_worker_call(monkeypatch, correction):
    config = OmegaConf.create(
        {
            "trainer": {"v1": {"reuse_proximal_actor_forward": True}, "critic_warmup": 0},
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "rollout_correction": correction},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_epochs": 1,
                    "shuffle": False,
                    "use_prefix_grouper": False,
                    "calculate_sum_pi_squared": False,
                    "policy_loss": {"loss_mode": "gspo"},
                    "ppo_mini_batch_size": 2,
                },
                "rollout": {"n": 1},
            },
        }
    )
    trainer = SimpleNamespace(
        config=config,
        global_steps=1,
        parameter_sync_step=1,
        use_critic=False,
        use_teacher_policy=False,
        actor_rollout_wg=SimpleNamespace(_query_dispatch_info=lambda _: [0]),
    )
    candidate = strategy.ProximalActorReuse(trainer)
    monkeypatch.setattr(strategy.worker_calls, "control", lambda *args: pytest.fail("unexpected worker call"))
    monkeypatch.setattr(candidate, "_prepare_proximal_actor", lambda batch: pytest.fail("unexpected preparation"))
    batch, metrics, calls = [1, 2], {}, []
    result = candidate._compute_old_log_prob(batch, metrics, lambda data, _: calls.append("native") or data)
    assert result is batch and calls == ["native"] and metrics["training/proximal_anchor_reuse"] == 0
