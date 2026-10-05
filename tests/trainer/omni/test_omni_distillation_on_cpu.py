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
"""Omni OPD config, vocabulary validation, and synthetic padding contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_module
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig

from verl_omni.trainer.omni import distillation as module
from verl_omni.workers.config.omni.distillation import OmniDistillationTeacherModelConfig


def test_omni_distillation_hydra_preserves_teacher_type_and_scoring_capacity():
    with initialize_config_module(config_module="verl_omni.trainer.config", version_base=None):
        config = compose(
            config_name="omni_trainer",
            overrides=[
                "distillation.enabled=true",
                "distillation.nnodes=1",
                "distillation.n_gpus_per_node=2",
                "distillation.teacher_models.teacher_model.model_path=teacher",
                "distillation.teacher_models.teacher_model.inference.name=vllm_omni",
                "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
                "distillation.distillation_loss.loss_mode=kl",
                "distillation.distillation_loss.use_policy_gradient=true",
            ],
        )
    parsed = omega_conf_to_dataclass(config.distillation)
    teacher = parsed.teacher_models["default"]
    assert isinstance(teacher, OmniDistillationTeacherModelConfig)
    assert teacher.inference.response_length == 1
    assert (
        teacher.inference.prompt_length
        == config.actor_rollout_ref.rollout.prompt_length + config.actor_rollout_ref.rollout.response_length
    )
    assert teacher.num_replicas == 2


@pytest.mark.parametrize("engine", ["vllm", "vllm_omni"])
def test_teacher_topk_extension_keeps_upstream_backends(engine):
    config = OmniDistillationTeacherModelConfig(inference=RolloutConfig(name=engine))
    config._validate_topk_logprobs(True, 8)
    assert config.inference.engine_kwargs[engine]["max_logprobs"] == 8
    with pytest.raises(ValueError, match="must be >="):
        config._validate_topk_logprobs(True, 16)


@pytest.mark.parametrize("topk", [None, 0, -1])
def test_teacher_topk_rejects_missing_or_nonpositive_budget(topk):
    config = OmniDistillationTeacherModelConfig(inference=RolloutConfig(name="vllm_omni"))
    with pytest.raises(ValueError, match="positive"):
        config._validate_topk_logprobs(True, topk)


@pytest.mark.parametrize("same_policy", [False, True])
def test_reverse_kl_updates_only_valid_student_actions(same_policy):
    from tensordict import TensorDict
    from verl.trainer.distillation.losses import distillation_loss
    from verl.workers.config import ActorConfig, DistillationLossConfig

    student = torch.tensor([-0.3, -0.5, -0.7, -0.2], requires_grad=True)
    teacher = student.detach().clone().reshape(4, 1)
    if not same_policy:
        teacher[1, 0] = -0.2
    teacher.requires_grad_()
    data = TensorDict(
        {
            "prompts": torch.tensor([[11, 12]]),
            "responses": torch.tensor([[13, 14]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "response_mask": torch.tensor([[True, False]]),
            "old_log_probs": student.detach()[1:3].reshape(1, 2),
            "teacher_logprobs": teacher,
            "dp_size": 1,
            "batch_num_tokens": 1,
            "global_batch_size": 1,
        },
        batch_size=[],
    )
    config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1, loss_agg_mode="token-mean")
    distillation = SimpleNamespace(
        distillation_loss=DistillationLossConfig(
            loss_mode="kl",
            use_policy_gradient=True,
            use_task_rewards=False,
        )
    )
    loss, _ = distillation_loss(config, distillation, {"log_probs": student}, data)
    loss.backward()
    expected = torch.zeros(4)
    if not same_policy:
        expected[1] = -0.3
    torch.testing.assert_close(student.grad, expected)
    assert teacher.grad is None


def test_teacher_padding_reuses_shared_template_and_preserves_source():
    from verl.trainer.ppo import padding_utils

    from verl_omni.workers.utils.padding import patched_padding_template

    wrapper = padding_utils.construct_minimal_padding_template
    assert wrapper is patched_padding_template
    source = {
        "input_ids": torch.arange(5),
        "teacher_ids": torch.ones(5, 2, dtype=torch.long),
        "teacher_logprobs": torch.full((5, 2), -0.7),
        "multi_modal_inputs": {"image": "data"},
    }
    sample, tag = wrapper(source, {}, 9)
    assert sample["teacher_ids"].shape == sample["teacher_logprobs"].shape == (2, 2)
    assert sample["teacher_ids"].tolist() == [[9, 9], [9, 9]]
    torch.testing.assert_close(sample["teacher_logprobs"], torch.zeros(2, 2))
    assert not sample["loss_mask"].any()
    assert sample["multi_modal_inputs"] == {}
    assert tag["is_padding"]
    assert source["teacher_ids"].shape == (5, 2)


@pytest.mark.parametrize("mismatch", [None, "vocab", "eos"])
def test_teacher_tokenizer_validation(monkeypatch, mismatch):
    def tokenizer():
        return SimpleNamespace(
            get_vocab=lambda: {"a": 1, "b": 2}, bos_token_id=0, eos_token_id=3, pad_token_id=0, all_special_ids=[0, 3]
        )

    student, teacher_tokenizer = tokenizer(), tokenizer()
    if mismatch == "vocab":
        teacher_tokenizer.get_vocab = lambda: {"a": 2, "b": 1}
    if mismatch == "eos":
        teacher_tokenizer.eos_token_id = 4
    teacher = SimpleNamespace(
        model_path="teacher",
        key="default",
        inference=SimpleNamespace(engine_kwargs={"vllm_omni": {"trust_remote_code": True}}),
    )
    monkeypatch.setattr(
        module, "omega_conf_to_dataclass", lambda config: SimpleNamespace(teacher_models={"default": teacher})
    )
    monkeypatch.setattr(module, "resolve_model_local_dir", lambda path: path)
    loader = MagicMock(return_value=teacher_tokenizer)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", loader)
    if mismatch:
        with pytest.raises(ValueError, match="tokenizer"):
            module.validate_teacher_tokenizers(student, SimpleNamespace(distillation={}))
    else:
        module.validate_teacher_tokenizers(student, SimpleNamespace(distillation={}))
    loader.assert_called_once_with("teacher", trust_remote_code=True)
