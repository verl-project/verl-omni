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

"""Reflection selection, block-causal replay and gradient regression checks."""

import pytest
import torch

from verl_omni.pipelines.bagel_alphagrpo.bagel_model import BagelForAlphaGRPO
from verl_omni.pipelines.bagel_alphagrpo.reflection import penalize_failed_refinements, select_initial_images
from verl_omni.pipelines.bagel_alphagrpo.reflection_context import context_layout
from verl_omni.pipelines.bagel_flow_grpo.bagel_model import BagelTrainingConfig


def tiny_model():
    torch.manual_seed(7)
    model = BagelForAlphaGRPO(
        BagelTrainingConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            latent_channel=2,
            latent_patch_size=2,
            max_latent_size=2,
            start_of_image_id=1,
            end_of_image_id=2,
        )
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("layers."))
    return model


def text_inputs():
    ids = torch.tensor([[3, 4, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7]])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 10:] = True
    return dict(
        thinking_input_ids=ids,
        thinking_attention_mask=torch.ones_like(mask),
        thinking_labels=torch.full_like(ids, 8),
        thinking_mask=mask,
        condition_image_embeds=torch.randn(1, 7, 16),
        condition_image_layout=torch.tensor([[2, 4, 3]]),
    )


def test_group_selection_handles_interleaved_uids_and_ties():
    selected = select_initial_images(torch.tensor([0.7, 0.4, 0.1, 0.4]), ["a", "b", "a", "b"], 2)
    assert selected.tolist() == [2, 1, 2, 1]


def test_partial_groups_fail():
    with pytest.raises(ValueError, match="complete candidate"):
        select_initial_images(torch.ones(3), ["a", "a", "b"], 2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_rewards_fail(bad):
    with pytest.raises(ValueError, match="finite"):
        select_initial_images(torch.tensor([bad, 1.0]), ["a", "a"], 2)


def test_failure_penalty_uses_initial_minimum_not_refined_minimum():
    scores, skip = penalize_failed_refinements(
        torch.tensor([0.1, 0.8, 0.2, 0.7]), torch.tensor([0.3, 0.4, 0.5, 0.6]), ["a", "a", "b", "b"]
    )
    torch.testing.assert_close(scores, torch.tensor([0.3, 0.8, 0.5, 0.7]))
    assert not skip


def test_no_improvement_includes_ties():
    _, skip = penalize_failed_refinements(torch.tensor([0.2, 0.3]), torch.tensor([0.2, 0.4]), ["a", "a"])
    assert skip


def test_layout_has_two_bidirectional_image_blocks_and_causal_text():
    positions, generation, mask = context_layout(12, 2, 4, 3, "cpu", target_length=4)
    assert positions.tolist() == [0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7]
    assert generation.nonzero().flatten().tolist() == [3, 4, 13, 14]
    assert mask[2:6, :6].all()
    assert not mask[:6, 6:].any()
    assert mask[6:9, :9].all()
    assert not mask[:9, 9:].any()
    assert not mask[10, 11]
    assert mask[12:, :].all()


def test_image_conditions_affect_text_likelihood_and_both_pathway_gradients():
    model = tiny_model()
    inputs = text_inputs()
    log_probs = model(**inputs)
    changed = model(**{**inputs, "condition_image_embeds": inputs["condition_image_embeds"] + 3})
    assert not torch.allclose(log_probs, changed, atol=1e-5)
    (-log_probs.sum()).backward()
    for name in ("q_proj", "v_proj_moe_gen"):
        gradient = getattr(model.layers[0].self_attn, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_future_text_cannot_change_earlier_action_likelihood():
    model = tiny_model()
    inputs = text_inputs()
    before = model(**inputs)
    inputs["thinking_input_ids"][0, 11] = 17
    after = model(**inputs)
    torch.testing.assert_close(before[:, 10], after[:, 10], rtol=0, atol=0)


def test_text_only_explicit_layout_matches_existing_replay():
    model = tiny_model()
    inputs = text_inputs()
    inputs.pop("condition_image_embeds")
    inputs.pop("condition_image_layout")
    expected = model(**inputs)
    actual = model(
        **inputs,
        condition_image_embeds=torch.empty(1, 0, 16),
        condition_image_layout=torch.zeros(1, 3, dtype=torch.long),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_refinement_velocity_has_finite_gradients():
    model = tiny_model()
    text = text_inputs()
    inputs = dict(
        hidden_states=torch.randn(1, 4, 8),
        timestep=torch.tensor([0.5]),
        latent_pos_ids=torch.arange(4)[None],
        text_token_ids=text["thinking_input_ids"],
        text_attention_mask=text["thinking_attention_mask"],
        condition_image_embeds=text["condition_image_embeds"],
        condition_image_layout=text["condition_image_layout"],
    )
    output = model(**inputs)[0]
    assert output.shape == inputs["hidden_states"].shape
    output.square().mean().backward()
    gradient = model.layers[0].self_attn.v_proj_moe_gen.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_trainable_cached_projection_rejected():
    model = tiny_model()
    model.vae2llm.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        model(**text_inputs())
