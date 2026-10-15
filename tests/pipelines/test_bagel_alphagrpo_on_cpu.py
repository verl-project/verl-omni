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

"""AlphaGRPO sampling likelihoods and unsupported configurations."""

from types import SimpleNamespace

import pytest
import torch
from verl.trainer.config.algorithm import RolloutCorrectionConfig

from verl_omni.pipelines.bagel_alphagrpo.vllm_omni_rollout_adapter import BagelAlphaGRPOPipeline
from verl_omni.trainer.diffusion.diffusion_algos import AlphaGRPOLoss
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig, FSDPDiffusionActorConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


def pipeline():
    instance = object.__new__(BagelAlphaGRPOPipeline)
    torch.nn.Module.__init__(instance)
    instance.device = torch.device("cpu")
    instance.new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}
    instance.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [3] * 64)
    instance.bagel = SimpleNamespace(config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1)))
    return instance


def request(**extra):
    config = DiffusionPipelineConfig()
    sampling = SimpleNamespace(
        extra_args=extra, max_sequence_length=config.max_sequence_length, past_key_values=None, seed=42
    )
    return SimpleNamespace(is_dummy_run=lambda: False, prompts=[{"prompt_ids": [1, 5, 2]}], sampling_params=sampling)


@pytest.mark.parametrize("temperature", [0.3, 1.0, 2.0])
def test_sampling_logp_matches_distribution(monkeypatch, temperature):
    instance = pipeline()
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    class Language:
        lm_head = staticmethod(lambda hidden: hidden)

        def __call__(self, **kwargs):
            return SimpleNamespace(past_key_values=None, packed_query_sequence=logits)

    instance.bagel.language_model = Language()
    probabilities = []

    def sample(probs, count, **kwargs):
        probabilities.append(probs)
        return torch.tensor([[3]])

    monkeypatch.setattr(torch, "multinomial", sample)
    tokens, logp = instance._sample_thinking(None, 0, 1, temperature, torch.Generator())
    expected = (logits / temperature).softmax(-1)
    assert tokens == [3]
    torch.testing.assert_close(probabilities[0], expected)
    assert logp[0] == pytest.approx(expected[0, 3].log().item())


@pytest.mark.parametrize("top_p", [0.01, 0.8])
def test_truncated_sampling_rejected(top_p):
    with pytest.raises(ValueError, match="think_top_p=1"):
        pipeline().forward(request(think_top_p=top_p))


def test_default_budget_reaches_prefill(monkeypatch):
    instance = pipeline()

    def prefill(*args, **kwargs):
        raise RuntimeError("prefill reached")

    monkeypatch.setattr(instance, "_prefill", prefill)
    with pytest.raises(RuntimeError, match="prefill reached"):
        instance.forward(request())


def test_explicit_budget_overflow_rejected():
    with pytest.raises(ValueError, match="exceed pipeline.max_sequence_length"):
        pipeline().forward(request(max_think_tokens=512))


@pytest.mark.parametrize(
    "correction",
    [
        {"bypass_mode": True},
        {"rollout_is": "sequence"},
        {"rollout_rs": "seq_mean_k1", "rollout_rs_threshold": 2.0},
    ],
)
def test_joint_correction_rejected_at_configuration(correction):
    rc = RolloutCorrectionConfig(**correction)
    with pytest.raises(ValueError, match="does not support rollout correction"):
        FSDPDiffusionActorConfig(
            rollout_n=2, diffusion_loss=DiffusionLossConfig(loss_mode="alphagrpo"), rollout_correction=rc
        )
    FSDPDiffusionActorConfig(rollout_n=2, rollout_correction=rc)


def test_joint_uncorrected_configuration():
    FSDPDiffusionActorConfig(
        rollout_n=2,
        diffusion_loss=DiffusionLossConfig(loss_mode="alphagrpo"),
        rollout_correction=RolloutCorrectionConfig(rollout_is=None, rollout_rs=None, bypass_mode=False),
    )


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_joint_loss_rejects_injected_correction_weights(weight):
    with pytest.raises(ValueError, match="does not support rollout correction"):
        AlphaGRPOLoss()(config=None, model_output={}, data={"rollout_is_weights": torch.tensor([weight])})
