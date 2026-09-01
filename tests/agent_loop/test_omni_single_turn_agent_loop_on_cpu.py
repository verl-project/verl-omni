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
"""CPU contracts for autoregressive omni rollout-to-replay policy output."""

from types import SimpleNamespace

import pytest
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils import tensordict_utils as tu

from verl_omni.agent_loop.single_turn_agent_loop import OmniSingleTurnAgentLoop
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase


def _output(extra_fields):
    return AgentLoopOutput(
        prompt_ids=[10, 11],
        response_ids=[1],
        response_mask=[1],
        response_logprobs=[-0.1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


def _internal_output(replay_payload, *, response_token):
    prompt_ids = torch.tensor([[10, 11]])
    response_ids = torch.tensor([[response_token, 0]])
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0),
        response_mask=torch.tensor([[1, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        response_logprobs=None,
        routed_experts=None,
        teacher_logprobs=None,
        teacher_ids=None,
        multi_modal_inputs=None,
        reward_score=None,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={"test_talker_replay": replay_payload},
    )


def test_omni_single_turn_agent_resolves_registered_adapter(monkeypatch):
    class Adapter(OmniRolloutPipelineBase):
        @classmethod
        def build_stage_configs(cls, pipeline_mode="talker"):
            return []

    monkeypatch.setattr(OmniRolloutPipelineBase, "_registry", {"test_talker": Adapter})
    config = SimpleNamespace(engine_kwargs={"vllm_omni": {"pipeline_name": "test_talker"}})

    assert OmniSingleTurnAgentLoop._resolve_rollout_adapter(config) is Adapter

    with pytest.raises(ValueError, match="requires a registered"):
        OmniSingleTurnAgentLoop._resolve_rollout_adapter(
            SimpleNamespace(engine_kwargs={"vllm_omni": {"pipeline_name": "missing"}})
        )
    with pytest.raises(ValueError, match="requires engine_kwargs"):
        OmniSingleTurnAgentLoop._resolve_rollout_adapter(SimpleNamespace(engine_kwargs={}))


@pytest.mark.asyncio
async def test_talker_contract_supports_different_trajectory_and_conditioning_shapes(monkeypatch):
    codebooks = torch.arange(24, dtype=torch.long).reshape(3, 8)
    hidden_states = torch.ones(4, 6)
    upstream_outputs = iter(
        [
            _output({"rvq_replay": {"codes": codebooks, "speaker_embedding": torch.ones(5)}}),
            _output(
                {
                    "hidden_state_replay": {
                        "policy_tokens": [31, 32, 33, 34],
                        "thinker_hidden_states": hidden_states,
                    }
                }
            ),
        ]
    )

    async def upstream_run(_self, sampling_params, **kwargs):
        assert sampling_params == {"temperature": 0.8}
        assert kwargs["raw_prompt"][0]["content"] == "hello"
        return next(upstream_outputs)

    monkeypatch.setattr(SingleTurnAgentLoop, "run", upstream_run)

    class RvqAdapter:
        @classmethod
        def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
            del tokenizer
            replay = output.extra_fields["rvq_replay"]
            output.response_ids = replay["codes"][:response_length, 0].tolist()
            output.response_mask = [1] * len(output.response_ids)
            output.response_logprobs = [-0.2] * len(output.response_ids)
            return output

    class HiddenStateAdapter:
        @classmethod
        def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
            del tokenizer
            replay = output.extra_fields["hidden_state_replay"]
            output.response_ids = replay["policy_tokens"][:response_length]
            output.response_mask = [1] * len(output.response_ids)
            output.response_logprobs = None
            return output

    loop = object.__new__(OmniSingleTurnAgentLoop)
    loop.response_length = 4
    loop.tokenizer = SimpleNamespace()

    loop.rollout_adapter = RvqAdapter
    rvq_output = await loop.run({"temperature": 0.8}, raw_prompt=[{"role": "user", "content": "hello"}])
    loop.rollout_adapter = HiddenStateAdapter
    hidden_output = await loop.run({"temperature": 0.8}, raw_prompt=[{"role": "user", "content": "hello"}])

    assert rvq_output.response_ids == codebooks[:, 0].tolist()
    assert rvq_output.extra_fields["rvq_replay"]["codes"].shape == (3, 8)
    assert hidden_output.response_ids == [31, 32, 33, 34]
    assert hidden_output.extra_fields["hidden_state_replay"]["thinker_hidden_states"].shape == (4, 6)


def test_agent_loop_batches_model_defined_replay_payload_as_top_level_micro_batch_key():
    replay_payloads = [
        {"codes": torch.arange(24, dtype=torch.long).reshape(3, 8), "speaker_embedding": torch.ones(5)},
        {"codes": torch.arange(32, dtype=torch.long).reshape(4, 8), "speaker_embedding": torch.ones(5) * 2},
    ]
    worker = object.__new__(AgentLoopWorker)
    worker.reward_loop_worker_handles = None

    data = worker._postprocess(
        [
            _internal_output(replay_payloads[0], response_token=7),
            _internal_output(replay_payloads[1], response_token=8),
        ]
    )

    assert "extra_fields" not in data.non_tensor_batch
    assert "test_talker_replay" in data.non_tensor_batch
    assert data.non_tensor_batch["test_talker_replay"][0]["codes"].shape == (3, 8)
    assert data.non_tensor_batch["test_talker_replay"][1]["codes"].shape == (4, 8)

    micro_batch = data.to_tensordict()
    transported_payloads = tu.get(micro_batch, "test_talker_replay")
    assert len(transported_payloads) == 2
    assert transported_payloads[0]["speaker_embedding"].tolist() == [1.0] * 5
    assert transported_payloads[1]["speaker_embedding"].tolist() == [2.0] * 5


@pytest.mark.asyncio
async def test_omni_single_turn_agent_rejects_misaligned_policy_output(monkeypatch):
    async def upstream_run(_self, sampling_params, **kwargs):
        return _output({"trajectory": torch.ones(2, 2)})

    monkeypatch.setattr(SingleTurnAgentLoop, "run", upstream_run)

    class MisalignedAdapter:
        @classmethod
        def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
            output.response_ids = [1, 2]
            output.response_mask = [1]
            return output

    loop = object.__new__(OmniSingleTurnAgentLoop)
    loop.rollout_adapter = MisalignedAdapter
    loop.response_length = 4
    loop.tokenizer = SimpleNamespace()

    with pytest.raises(ValueError, match="response_mask must align"):
        await loop.run({"temperature": 0.8}, raw_prompt=[{"role": "user", "content": "hello"}])
