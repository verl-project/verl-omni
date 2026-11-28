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

"""Contract tests for native H3 LoRA and packed velocity replay."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.minimax_h3_flow_grpo.veomni import map_fused_lora, predict_row_velocities


@pytest.mark.parametrize(
    ("backend", "target", "error"),
    [
        (None, "to_q", None),
        ("fsdp", "attn.to_q", None),
        ("fsdp2", "to_q", None),
        ("fsdp2", "qkv_proj", "LoRA supports"),
        ("veomni", "qkv_proj", None),
        ("veomni", "to_q", "LoRA supports"),
        ("unknown", "to_q", "Unsupported H3 FlowGRPO training backend"),
    ],
)
def test_lora_targets_are_validated_for_the_selected_backend(tmp_path, backend, target, error):
    from verl_omni.workers.config import DiffusionModelConfig

    config = dict(
        path=str(tmp_path),
        architecture="MiniMaxH3Pipeline",
        algorithm="flow_grpo",
        load_tokenizer=False,
        attn_backend="native",
        training_backend=backend,
        lora_rank=4,
        target_modules=[target],
    )
    if error is None:
        DiffusionModelConfig(**config)
    else:
        with pytest.raises(ValueError, match=error):
            DiffusionModelConfig(**config)


@pytest.mark.parametrize(("leaf", "names"), [("qkv_proj", ["to_q", "to_k", "to_v"]), ("fc1", ["fc1_0", "fc1_1"])])
def test_logical_lora_updates_equal_native_update(leaf, names):
    heads, dim, width, rank = 2, 4, 8, 3
    arch = SimpleNamespace(num_attention_heads=heads, attention_head_dim=dim, ffn_hidden_size=width)
    block = "blocks.0.attn" if leaf == "qkv_proj" else "blocks.0.mlp"
    prefix = f"transformer.dit.{block}.{leaf}"
    a = torch.arange(rank * width).reshape(rank, width).float()
    b = torch.arange(len(names) * width * rank).reshape(-1, rank).float()
    weights = {f"{prefix}.lora_A.weight": a, f"{prefix}.lora_B.weight": b}
    mapped = map_fused_lora(weights, {leaf}, "transformer", arch)
    updates = [
        mapped[f"transformer.{block}.{n}.lora_B.weight"] @ mapped[f"transformer.{block}.{n}.lora_A.weight"]
        for n in names
    ]
    if leaf == "qkv_proj":
        actual = torch.stack([u.reshape(heads, dim, width) for u in updates], dim=1).reshape_as(b @ a)
    else:
        actual = torch.cat(updates)
    torch.testing.assert_close(actual, b @ a)
    with pytest.raises(ValueError, match="A/B pair"):
        map_fused_lora({f"{prefix}.lora_A.weight": a}, {leaf}, "transformer", arch)


@pytest.mark.parametrize("kind", ["lora_A", "lora_B"])
def test_lora_export_rejects_duplicate_normalized_key(kind):
    path = "blocks.0.attn.out_proj"
    weights = {
        f"{path}.lora_A.weight": torch.ones(2, 4),
        f"{path}.lora_B.weight": torch.ones(4, 2),
    }
    weights[f"transformer.dit.{path}.{kind}.weight"] = torch.zeros_like(weights[f"{path}.{kind}.weight"])
    with pytest.raises(ValueError, match=rf"Duplicate H3 LoRA tensor.*{kind}"):
        map_fused_lora(weights, {"out_proj"}, "transformer", SimpleNamespace())


def test_replay_preserves_interleaved_rows_and_velocity_sign():
    pytest.importorskip("veomni")
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import unpack_audio, unpatchify_video

    vi, ai, ti = map(torch.tensor, ([4, 0], [2, 5], [1, 3]))
    v = torch.randn(1, 2, 96, requires_grad=True)
    a = torch.randn(1, 2, 32, requires_grad=True)

    def module(**kwargs):
        torch.testing.assert_close(kwargs["x"][0, vi], v[0])
        torch.testing.assert_close(kwargs["audio_x"][0, ai], a[0])
        for key, length in (("packed_seq_params", 6), ("refiner_packed_seq_params", 2)):
            params = kwargs[key]
            assert params["cu_seqlens_host"] == (0, length)
            assert params["cu_seqlens_q"].tolist() == [0, length]
            assert params["max_seqlen_q"] == length
        return SimpleNamespace(
            predictions=[
                -unpatchify_video(2 * kwargs["x"][0, vi], 2, 2, 2),
                -unpack_audio(3 * kwargs["audio_x"][0, ai], 1, 2),
            ]
        )

    video, audio = predict_row_velocities(
        module,
        dict(
            hidden_states=v,
            audio_hidden_states=a,
            encoder_hidden_states=torch.randn(1, 2, 8),
            position_ids=torch.zeros(6, 3),
            timestep=torch.tensor([0.2, 0.75]),
            timestep_indices=torch.tensor([0, 0, 1, 0, 0, 1]),
            token_tags=torch.tensor([0, 1, 2, 1, 0, 2]),
            video_indices=vi,
            audio_indices=ai,
            text_indices=ti,
        ),
    )
    torch.testing.assert_close(video, 2 * v)
    torch.testing.assert_close(audio, 3 * a)
    (video.sum() + audio.sum()).backward()
    torch.testing.assert_close(v.grad, torch.full_like(v, 2))
    torch.testing.assert_close(a.grad, torch.full_like(a, 3))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_native_velocities_and_cps_log_prob_match_diffusers(batch_size):
    pytest.importorskip("veomni")
    compare_fixed_t2va_replay(torch.device("cpu"), batch_size)


@torch.no_grad()
def compare_fixed_t2va_replay(device, batch_size=1):
    """Compare both real DiTs at the same weights, trajectory and scheduler step."""
    from diffusers import MiniMaxH3Transformer3DModel
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
        MiniMaxH3DiTModelConfig,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    from tests.special_e2e.minimax_h3_checkpoint_parity import _diffusers_block_base, _diffusers_name
    from verl_omni.pipelines.minimax_h3_flow_grpo.common import configure_flow_scheduler, flatten_joint_latents
    from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    torch.manual_seed(42)
    shared = dict(
        hidden_size=32, num_layers=1, num_attention_heads=2, attention_head_dim=16, text_dim=32, time_embed_dim=16
    )
    native = MiniMaxH3DiTModel(
        MiniMaxH3DiTModelConfig(
            **shared,
            token_refiner_num_layers=1,
            ffn_hidden_size=64,
            timestep_input_dim=16,
            time_embed_hidden_size=32,
            adaln_out_features=576,
            final_adaln_out_features=64,
            rope_inv_freq_len=2,
            ops_implementation=OpsImplementationConfig(
                attn_implementation="eager",
                rms_norm_implementation="eager",
                rotary_pos_emb_implementation="eager",
            ),
        )
    ).eval()
    baseline = MiniMaxH3Transformer3DModel(
        **shared, num_refiner_layers=1, ffn_dim=64, freq_dim=16, time_embed_hidden_dim=32, rope_freq_dim=2
    ).eval()
    state = {}
    for name, value in native.dit.state_dict().items():
        if name.endswith(".attn.qkv_proj.weight"):
            base = _diffusers_block_base(name, ".attn.qkv_proj.weight")
            for key, part in zip(("to_q", "to_k", "to_v"), value.reshape(2, 3, 16, -1).unbind(1), strict=True):
                state[f"{base}.attn.{key}.weight"] = part.reshape(32, -1)
        elif name.endswith(".mlp.fc1.weight"):
            base = _diffusers_block_base(name, ".mlp.fc1.weight")
            gate, up = value.chunk(2)
            state[f"{base}.ff.net.0.proj.weight"] = torch.cat((up, gate))
        elif name != "rope.inv_freq":
            state[_diffusers_name(name)] = value
    baseline.load_state_dict(state, strict=True)
    native.to(device)
    baseline.to(device)
    inputs = dict(
        hidden_states=torch.randn(batch_size, 2, 96),
        audio_hidden_states=torch.randn(batch_size, 2, 32),
        encoder_hidden_states=torch.randn(batch_size, 2, 32),
        timestep=torch.tensor([0.65, 0.875]),
        timestep_indices=torch.tensor([0, 0, 1, 1, 0, 0]),
        token_tags=torch.tensor([0, 0, 2, 2, 1, 1]),
        position_ids=torch.arange(18, dtype=torch.float32).view(6, 3) / 8,
        video_indices=torch.tensor([0, 1]),
        audio_indices=torch.tensor([2, 3]),
        text_indices=torch.tensor([4, 5]),
        return_dict=False,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    expected = baseline(**inputs)
    observed = predict_row_velocities(native, inputs)
    inputs.update(
        _h3_scheduler_step=1,
        _h3_video_update_mask=torch.ones(2, dtype=torch.bool, device=device),
        _h3_audio_update_mask=torch.ones(2, dtype=torch.bool, device=device),
        _h3_target_only_trajectory=False,
    )
    sched = (FlowMatchSDEDiscreteScheduler(), FlowMatchSDEDiscreteScheduler())
    for scheduler, sigmas in zip(sched, ([1.0, 0.65, 0.25, 0.0], [1.0, 0.875, 0.35, 0.0]), strict=True):
        configure_flow_scheduler(scheduler, sigmas, device)
    next_rows = (
        flatten_joint_latents(torch.randn(batch_size, 2, 96), torch.randn(batch_size, 2, 32)).unsqueeze(1).to(device)
    )
    algo = SimpleNamespace(noise_level=0.8, sde_type="cps")
    native_config = SimpleNamespace(algo=algo, training_backend="veomni")
    diffusers_config = SimpleNamespace(algo=algo, training_backend="fsdp2")
    replay_args = (inputs, None, {"all_next_latents": next_rows}, 0)
    native_output = MiniMaxH3FlowGRPO.forward_and_sample_previous_step(native, sched, native_config, *replay_args)
    base_output = MiniMaxH3FlowGRPO.forward_and_sample_previous_step(baseline, sched, diffusers_config, *replay_args)
    differences = {
        "video_max_abs": (observed[0] - expected[0]).abs().max().item(),
        "audio_max_abs": (observed[1] - expected[1]).abs().max().item(),
        "log_prob_max_abs": (native_output[0] - base_output[0]).abs().max().item(),
    }
    for actual, reference in zip((*observed, *native_output), (*expected, *base_output), strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    return differences
