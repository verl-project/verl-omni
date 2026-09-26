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
"""CPU contract tests for MiniMax H3 Diffusers context parallelism."""

import pytest
import torch

from tests.special_e2e.run_flowgrpo_minimax_h3_tiny import _hydra_overrides, _validate_actor_sp
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    TEXT_TAG,
    h3_ulysses_forward,
    pad_h3_layout_for_ulysses,
)

_TINY_H3 = dict(
    in_channels=4,
    audio_in_channels=4,
    num_layers=2,
    num_refiner_layers=1,
    hidden_size=32,
    num_attention_heads=2,
    attention_head_dim=16,
    ffn_dim=64,
    text_dim=16,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=1,
)


def _layout_inputs(text_rows: int = 2, video_rows: int = 3, audio_rows: int = 2) -> dict:
    """Build an H3 packed layout whose joint length is intentionally odd by default."""
    generator = torch.Generator().manual_seed(3)
    seq_len = text_rows + video_rows + audio_rows
    token_tags = torch.tensor([1] * text_rows + [0] * video_rows + [2] * audio_rows)
    row_timesteps = torch.full((seq_len,), 0.5)
    row_timesteps[text_rows] = 0.999
    timestep, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
    return {
        "hidden_states": torch.randn(1, video_rows, 16, generator=generator),
        "audio_hidden_states": torch.randn(1, audio_rows, 4, generator=generator),
        "encoder_hidden_states": torch.randn(1, text_rows, 16, generator=generator),
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": torch.randn(seq_len, 3, generator=generator),
        "video_indices": torch.arange(text_rows, text_rows + video_rows),
        "audio_indices": torch.arange(text_rows + video_rows, seq_len),
        "text_indices": torch.arange(text_rows),
        "return_dict": False,
    }


def test_minimax_h3_uses_diffusers_standard_ulysses_configuration() -> None:
    from diffusers import ContextParallelConfig

    config = ContextParallelConfig(ulysses_degree=2)

    assert config.ulysses_anything is False
    assert config.ring_anything is False


@pytest.mark.parametrize(("sequence_length", "sp_size"), [(7, 1), (8, 2), (8, 4), (12, 4)])
def test_minimax_h3_padding_is_a_no_op_for_aligned_layouts(sequence_length, sp_size) -> None:
    inputs = _layout_inputs(text_rows=sequence_length - 5)

    assert pad_h3_layout_for_ulysses(inputs, sp_size) is inputs


@pytest.mark.parametrize(("sequence_length", "sp_size", "padded_length"), [(7, 2, 8), (9, 4, 12), (10, 4, 12)])
def test_minimax_h3_padding_masks_trailing_rows(sequence_length, sp_size, padded_length) -> None:
    inputs = _layout_inputs(text_rows=sequence_length - 5)

    padded = pad_h3_layout_for_ulysses(inputs, sp_size)

    assert padded["position_ids"].shape == (padded_length, 3)
    assert padded["token_tags"].shape == padded["timestep_indices"].shape == (padded_length,)
    assert torch.equal(padded["position_ids"][:sequence_length], inputs["position_ids"])
    assert torch.all(padded["position_ids"][sequence_length:] == 0)
    assert torch.all(padded["token_tags"][sequence_length:] == TEXT_TAG)
    assert torch.all(padded["timestep_indices"][sequence_length:] == 0)
    assert padded["attention_mask"].shape == (1, 1, 1, padded_length)
    assert padded["attention_mask"].dtype == torch.bool
    assert padded["attention_mask"][..., :sequence_length].all()
    assert not padded["attention_mask"][..., sequence_length:].any()
    for key in ("video_indices", "audio_indices", "text_indices"):
        assert padded[key] is inputs[key]


def test_minimax_h3_padding_rejects_nonpositive_sp_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        pad_h3_layout_for_ulysses(_layout_inputs(), sp_size=0)


def test_minimax_h3_padding_rejects_non_h3_modules() -> None:
    with pytest.raises(TypeError, match="MiniMaxH3Transformer3DModel"):
        h3_ulysses_forward(torch.nn.Linear(1, 1), _layout_inputs(), sp_size=2)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("sp_size", [2, 4])
def test_minimax_h3_padded_forward_matches_unpadded_forward(sp_size, gradient_checkpointing) -> None:
    """Masked padding rows must not change real-row outputs or parameter gradients."""
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(0)
    reference = MiniMaxH3Transformer3DModel(**_TINY_H3).float()
    padded_model = MiniMaxH3Transformer3DModel(**_TINY_H3).float()
    padded_model.load_state_dict(reference.state_dict())
    for model in (reference, padded_model):
        model.set_attention_backend("native")
        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
    inputs = _layout_inputs()

    expected = reference(**inputs)
    actual = h3_ulysses_forward(padded_model, inputs, sp_size)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    sum(output.square().mean() for output in expected).backward()
    sum(output.square().mean() for output in actual).backward()
    compared = 0
    for (name, got), (_, want) in zip(padded_model.named_parameters(), reference.named_parameters(), strict=True):
        if want.grad is None:
            assert got.grad is None, name
            continue
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-5, atol=1e-6, msg=name)
        compared += 1
    assert compared > 0


def test_minimax_h3_masked_forward_fails_closed_on_diffusers_signature_drift(monkeypatch) -> None:
    from diffusers import MiniMaxH3Transformer3DModel

    import verl_omni.pipelines.minimax_h3_diffusion_nft.common as common

    monkeypatch.setattr(common, "_H3_FORWARD_PARAMETERS", ("self", "hidden_states"))
    with pytest.raises(RuntimeError, match="Revalidate the masked forward"):
        h3_ulysses_forward(MiniMaxH3Transformer3DModel(**_TINY_H3), _layout_inputs(), sp_size=2)


@pytest.mark.parametrize(("task", "train_batch_size"), [("t2va", 8), ("fl2va", 8), ("ref2va", 4)])
def test_minimax_h3_smoke_config_enables_sp_for_each_task(task, train_batch_size) -> None:
    overrides = _hydra_overrides(
        tiny_model_dir="/tmp/model",
        train_parquet="/tmp/train.parquet",
        val_parquet="/tmp/val.parquet",
        reward_stub_path="/tmp/reward.py",
        output_dir="/tmp/output",
        task=task,
        actor_backend="fsdp2",
        num_gpus=4,
        actor_sp=2,
        rollout_tp=2,
        text_encoder_tp=1,
        total_training_steps=1,
        ray_num_cpus=4,
        height=160,
        width=288,
        num_frames=97,
        num_inference_steps=4,
    )

    assert "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2" in overrides
    assert f"data.train_batch_size={train_batch_size}" in overrides
    assert f"actor_rollout_ref.rollout.pipeline.task={task}" in overrides


def test_minimax_h3_smoke_config_rejects_invalid_sp_partition() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _validate_actor_sp(num_gpus=4, actor_sp=3)


def test_minimax_h3_smoke_config_rejects_veomni_actor_sp() -> None:
    with pytest.raises(ValueError, match="SP=1 only"):
        _hydra_overrides(
            tiny_model_dir="/tmp/model",
            train_parquet="/tmp/train.parquet",
            val_parquet="/tmp/val.parquet",
            reward_stub_path="/tmp/reward.py",
            output_dir="/tmp/output",
            task="t2va",
            actor_backend="veomni",
            num_gpus=4,
            actor_sp=2,
            rollout_tp=2,
            text_encoder_tp=1,
            total_training_steps=1,
            ray_num_cpus=4,
            height=160,
            width=288,
            num_frames=97,
            num_inference_steps=4,
        )
