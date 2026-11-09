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

"""CPU contracts for the Lance FlowGRPO integration.

Lance reuses BAGEL's MoT transformer; what it does not share is the sigma
shift, the Wan2.2 latent geometry and the checkpoint layout. These tests also
check three-axis image positions and that right padding preserves each
sample's output and gradients. Nothing here needs a GPU or the released
checkpoint; rollout contract tests require vllm-omni.
"""

import inspect
import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import RotaryEmbedding, get_flattened_position_ids
from verl_omni.pipelines.bagel_flow_grpo.common import (
    BAGEL_TIMESTEP_SHIFT,
    bagel_time_shift,
    setup_bagel_sigmas,
)
from verl_omni.pipelines.lance_flow_grpo.common import (
    LANCE_MROPE_SECTION,
    LANCE_TEMPORAL_ROPE_SCALE,
    LANCE_TIMESTEP_SHIFT,
    setup_lance_sigmas,
)
from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion
from verl_omni.pipelines.lance_flow_grpo.lance_model import (
    LanceForTraining,
    LanceRotaryEmbedding,
    LanceTrainingConfig,
    map_lance_checkpoint_to_training,
    resolve_checkpoint_dir,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

#: Key set mirrors the released ``Lance_3B/llm_config.json`` (a Qwen2.5-VL
#: config carrying ``vision_start_token_id`` / ``vision_end_token_id`` and an
#: mrope ``rope_scaling``), with the dimensions shrunk for CPU.
TINY_LLM_CONFIG = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "model_type": "qwen2_5_vl",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 200,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": True,
    "vision_start_token_id": 198,
    "vision_end_token_id": 199,
    # head_dim here is 16, so the released [16, 24, 24] does not fit; this
    # keeps Qwen2.5-VL's 1 : 1.5 : 1.5 ratio at that size.
    "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
}


def _tiny_config() -> LanceTrainingConfig:
    return LanceTrainingConfig(
        hidden_size=TINY_LLM_CONFIG["hidden_size"],
        intermediate_size=TINY_LLM_CONFIG["intermediate_size"],
        num_hidden_layers=TINY_LLM_CONFIG["num_hidden_layers"],
        num_attention_heads=TINY_LLM_CONFIG["num_attention_heads"],
        num_key_value_heads=TINY_LLM_CONFIG["num_key_value_heads"],
        vocab_size=TINY_LLM_CONFIG["vocab_size"],
        max_latent_size=8,
        start_of_image_id=TINY_LLM_CONFIG["vision_start_token_id"],
        end_of_image_id=TINY_LLM_CONFIG["vision_end_token_id"],
        mrope_section=tuple(TINY_LLM_CONFIG["rope_scaling"]["mrope_section"]),
    )


def _write_tiny_checkpoint(
    root: str,
    *,
    subdir: str | None = "Lance_3B",
    max_latent_size: int = 8,
    num_frames: int = 1,
) -> str:
    """Write a tiny Lance-layout checkpoint and return the bundle root.

    ``max_latent_size`` and ``num_frames`` shape ``latent_pos_embed``, which is
    how a real video checkpoint differs from the image one.
    """
    ckpt_dir = os.path.join(root, subdir) if subdir else root
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "llm_config.json"), "w") as f:
        json.dump(TINY_LLM_CONFIG, f)

    config = _tiny_config()
    config.max_latent_size = max_latent_size
    config.max_num_frames = num_frames
    reference = LanceForTraining(config)
    state_dict = {}
    for name, tensor in reference.state_dict().items():
        if name.startswith(("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")):
            state_dict[name] = tensor.clone()
        else:
            state_dict[f"language_model.model.{name}"] = tensor.clone()
    # Keys the trainer must drop: the LM head and the understanding tower.
    state_dict["language_model.lm_head.weight"] = torch.zeros(
        TINY_LLM_CONFIG["vocab_size"], TINY_LLM_CONFIG["hidden_size"]
    )
    state_dict["vit_model.blocks.0.attn.qkv.weight"] = torch.zeros(3, 3)
    save_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
    return ckpt_dir


# ---------------------------------------------------------------------------
# Sigma schedule
# ---------------------------------------------------------------------------


def _rollout_denoise_grid(num_steps: int, shift: float, extra_point: bool) -> torch.Tensor:
    """The timesteps ``Bagel.generate_image`` walks, rebuilt from its own formula.

    Kept independent of the helper the implementation uses, so that a change to
    either the shift formula or the number of schedule points shows up here.
    """
    points = num_steps + 1 if extra_point else num_steps
    t = torch.linspace(1, 0, points, dtype=torch.float32)
    t = shift * t / (1 + (shift - 1) * t)
    return t[:-1]


@pytest.mark.parametrize("num_steps", [2, 15, 30])
def test_trainer_sigmas_are_the_grid_the_rollout_walks(num_steps):
    """The trainer must replay on the rollout's own grid, step for step.

    Lance departs from BAGEL twice: the 3.5 shift, and one extra schedule
    point (``LanceBagel._denoise_schedule_extra_step``) that makes the denoise
    loop run ``num_steps`` Euler steps rather than one fewer.  Taking only the
    shift leaves the trainer one sigma short and every remaining sigma on a
    different grid, which the replay cannot recover from.
    """
    scheduler = FlowMatchSDEDiscreteScheduler()
    setup_lance_sigmas(scheduler, num_steps)

    expected = _rollout_denoise_grid(num_steps, LANCE_TIMESTEP_SHIFT, extra_point=True)
    assert len(scheduler.timesteps) == num_steps
    torch.testing.assert_close(scheduler.timesteps.float(), expected, rtol=0, atol=1e-6)


def test_bagels_own_schedule_is_unchanged():
    """The hook Lance overrides must leave BAGEL on its own grid."""
    scheduler = FlowMatchSDEDiscreteScheduler()
    setup_bagel_sigmas(scheduler, 15)

    expected = _rollout_denoise_grid(15, BAGEL_TIMESTEP_SHIFT, extra_point=False)
    assert len(scheduler.timesteps) == 14
    torch.testing.assert_close(scheduler.timesteps.float(), expected, rtol=0, atol=1e-6)


def test_the_training_adapter_builds_lances_schedule():
    """The adapter the trainer actually calls, not just the helper underneath.

    ``set_timesteps`` is the only place the training side picks a schedule, so
    a Lance adapter that fell back to BAGEL's would go unnoticed if the tests
    exercised ``setup_lance_sigmas`` alone.
    """
    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

    scheduler = FlowMatchSDEDiscreteScheduler()
    model_config = SimpleNamespace(pipeline=SimpleNamespace(num_inference_steps=15))
    LanceDiffusion.set_timesteps(scheduler, model_config, device=None)

    expected = _rollout_denoise_grid(15, LANCE_TIMESTEP_SHIFT, extra_point=True)
    assert len(scheduler.timesteps) == 15
    torch.testing.assert_close(scheduler.timesteps.float(), expected, rtol=0, atol=1e-6)


def test_the_extra_schedule_point_is_the_one_lance_declares():
    """Pin the extra point to the rollout model rather than restating it."""
    from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel
    from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel

    from verl_omni.pipelines.lance_flow_grpo.common import LANCE_SCHEDULE_EXTRA_POINTS

    assert LanceBagel._denoise_schedule_extra_step is True
    assert Bagel._denoise_schedule_extra_step is False
    assert LANCE_SCHEDULE_EXTRA_POINTS == 1


def test_lance_shift_is_not_the_bagel_shift():
    """Guard against silently inheriting BAGEL's 3.0."""
    assert LANCE_TIMESTEP_SHIFT != BAGEL_TIMESTEP_SHIFT

    t = torch.linspace(1, 0, 15, dtype=torch.float32)
    lance = bagel_time_shift(LANCE_TIMESTEP_SHIFT, t)
    bagel = bagel_time_shift(BAGEL_TIMESTEP_SHIFT, t)
    assert not torch.allclose(lance, bagel)


# ---------------------------------------------------------------------------
# mRoPE: the trainer has to replay on the rotary basis the rollout used
# ---------------------------------------------------------------------------

#: A 512x512 image at Lance's 16x downsample: a 32x32 latent grid.
_IMAGE_HW = 512
_LATENT_DOWNSAMPLE = 16


def _rollout_image_block_positions(image_hw: int, anchor: int) -> torch.Tensor:
    """The positions ``LanceBagel`` feeds for the generation block.

    Calls the rollout's own builder rather than restating its layout, so a
    change upstream shows up here as a failure instead of being mirrored by a
    copy that quietly agrees with itself.
    """
    from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel

    stub = SimpleNamespace(latent_downsample=_LATENT_DOWNSAMPLE)
    return LanceBagel._per_token_mrope_for_vae_latent(stub, [(image_hw, image_hw)], [anchor])


def test_trainer_positions_are_the_ones_the_rollout_feeds():
    """Start marker, per-token latents and end marker must all line up.

    The rollout replaces BAGEL's single scalar for the image block with
    ``(P, P, P)`` for the start marker, ``(P+1, P+1+hi, P+1+wi)`` per latent
    and ``(P+max_hw+1, ...)`` for the end marker.  Replaying on BAGEL's scalar
    would evaluate the policy under a different rotary basis than the one that
    produced the trajectory.
    """
    num_text = 7
    config = _tiny_config()
    config.max_latent_size = 64  # the released grid, so the ids match the rollout
    model = LanceForTraining(config)
    latent_pos_ids = get_flattened_position_ids(_IMAGE_HW, _IMAGE_HW, _LATENT_DOWNSAMPLE, 64)

    positions = model.build_position_ids(1, num_text, latent_pos_ids.numel(), latent_pos_ids, torch.device("cpu"))
    assert positions.shape == (1, 3, num_text + 1 + latent_pos_ids.numel() + 1)

    expected = _rollout_image_block_positions(_IMAGE_HW, num_text)
    assert torch.equal(positions[0, :, num_text:], expected)
    # the text prefix keeps BAGEL's plain arange, broadcast over the three axes
    text = torch.arange(num_text)
    for axis in range(3):
        assert torch.equal(positions[0, axis, :num_text], text)


def test_trainer_positions_are_not_bagels_scalar():
    """The change has to be visible: a scalar block would not carry the grid."""
    config = _tiny_config()
    config.max_latent_size = 64
    model = LanceForTraining(config)
    latent_pos_ids = get_flattened_position_ids(_IMAGE_HW, _IMAGE_HW, _LATENT_DOWNSAMPLE, 64)

    positions = model.build_position_ids(1, 7, latent_pos_ids.numel(), latent_pos_ids, torch.device("cpu"))
    block = positions[0, :, 7:]
    assert block[1].unique().numel() > 1, "the h axis must vary across rows"
    assert block[2].unique().numel() > 1, "the w axis must vary across columns"
    assert not torch.equal(block[1], block[2]), "h and w must not collapse onto each other"


#: 25 pixel frames at the 4x temporal stride: 7 latent frames.
_VIDEO_FRAMES = 25
_VIDEO_LATENT_FRAMES = (_VIDEO_FRAMES - 1) // 4 + 1


def _rollout_video_block_positions(video_shape: tuple[int, int, int], anchor: int) -> torch.Tensor:
    """The 3-D positions ``LanceBagel`` feeds for a video generation block.

    Calls the rollout's own builder, like the image helper above, so the trainer
    cannot drift from the rotary basis the trajectory was produced under.
    """
    from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel

    stub = SimpleNamespace(
        latent_downsample=_LATENT_DOWNSAMPLE,
        config=SimpleNamespace(vae_config=SimpleNamespace(downsample_temporal=4)),
    )
    return LanceBagel._per_token_mrope_for_video_latent(stub, [video_shape], [anchor])


def test_trainer_video_positions_are_the_ones_the_rollout_feeds():
    """Video adds a temporal axis, amplified the way the rollout amplifies it.

    ``build_position_ids`` has to place ``(P + 1 + t * t_scale, P + 1 + h, P + 1 + w)``
    for the latent block so the trainer replays on the rollout's rotary basis.
    """
    num_text = 7
    config = _tiny_config()
    config.max_latent_size = 64  # the released grid, so the ids match the rollout
    model = LanceForTraining(config)
    latent_pos_ids = get_flattened_position_ids(
        _IMAGE_HW, _IMAGE_HW, _LATENT_DOWNSAMPLE, 64, num_frames=_VIDEO_LATENT_FRAMES
    )
    assert latent_pos_ids.numel() == _VIDEO_LATENT_FRAMES * 32 * 32

    positions = model.build_position_ids(1, num_text, latent_pos_ids.numel(), latent_pos_ids, torch.device("cpu"))
    expected = _rollout_video_block_positions((_VIDEO_FRAMES, _IMAGE_HW, _IMAGE_HW), num_text)
    assert torch.equal(positions[0, :, num_text:], expected)

    # The temporal axis must actually be present, and carry the amplification.
    # The block is the start marker, the latent tokens and the end marker.
    latent = positions[0, :, num_text + 1 : -1]
    assert latent[0].unique().numel() == _VIDEO_LATENT_FRAMES
    assert int(latent[0].max()) - int(latent[0].min()) == (_VIDEO_LATENT_FRAMES - 1) * LANCE_TEMPORAL_ROPE_SCALE


def test_lance_adapter_asks_for_a_temporal_axis_only_on_video():
    """The adapter derives the grid; one frame must stay BAGEL's image grid."""
    config = _tiny_config()
    config.max_latent_size = 64
    config.max_num_frames = _VIDEO_LATENT_FRAMES
    module = SimpleNamespace(config=config)
    model_config = SimpleNamespace(pipeline=SimpleNamespace(height=_IMAGE_HW, width=_IMAGE_HW, num_frames=1))

    image_ids = LanceDiffusion._get_latent_pos_ids(model_config, module, "cpu")
    assert torch.equal(image_ids, get_flattened_position_ids(_IMAGE_HW, _IMAGE_HW, _LATENT_DOWNSAMPLE, 64))

    model_config.pipeline.num_frames = _VIDEO_FRAMES
    video_ids = LanceDiffusion._get_latent_pos_ids(model_config, module, "cpu")
    assert video_ids.numel() == _VIDEO_LATENT_FRAMES * 32 * 32
    assert int(video_ids.max()) == (_VIDEO_LATENT_FRAMES - 1) * 64 * 64 + 31 * 64 + 31


@pytest.mark.parametrize("is_video", [False, True])
def test_lance_rollout_routes_the_loaded_checkpoint(is_video):
    from verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter import LancePipelineWithLogProb

    pipeline = object.__new__(LancePipelineWithLogProb)
    pipeline._is_video = is_video
    request = SimpleNamespace(prompts=[{"prompt": "A cat"}], sampling_params=SimpleNamespace(extra_args={}))
    pipeline._ensure_bagel_prompt_text(request)
    assert request.prompts[0]["modalities"] == ["video" if is_video else "image"]


@pytest.mark.parametrize("output_type", ["pil", "latent", "both"])
def test_lance_video_artifacts_preserve_frames_latents_and_replay(output_type):
    import numpy as np
    from PIL import Image
    from vllm_omni.diffusion.data import DiffusionOutput

    from verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter import LancePipelineWithLogProb

    pipeline = object.__new__(LancePipelineWithLogProb)
    pipeline._is_video = True
    request = SimpleNamespace(
        request_id="video-contract",
        sampling_params=SimpleNamespace(frame_rate=24.0, output_type=output_type, extra_args={}),
    )
    frames = [Image.fromarray(np.full((8, 12, 3), i, dtype=np.uint8)) for i in range(5)]
    native = torch.randn(6, 48)
    states, times, logps = torch.randn(1, 3, 6, 48), torch.rand(1, 2), torch.randn(1, 2)
    output = DiffusionOutput(output={"payload": {"video": frames}, "metadata": {}})
    result = pipeline._build_flowgrpo_output(request, output, native, states, times, logps)
    payload = result.output["payload"]["video"][0]
    assert payload["video_preview"].shape == (5, 3, 8, 12)
    torch.testing.assert_close(payload["video_preview"][:, 0, 0, 0], torch.arange(5, dtype=torch.uint8))
    torch.testing.assert_close(payload["video_latent"], native)
    torch.testing.assert_close(result.trajectory_latents, states, atol=0, rtol=0)
    torch.testing.assert_close(result.trajectory_timesteps, times, atol=0, rtol=0)
    torch.testing.assert_close(result.trajectory_log_probs, logps, atol=0, rtol=0)
    header = result.output["metadata"]["media_artifacts"]
    assert header["primary"] == ("video_latent" if output_type == "latent" else "video_preview")


def test_lance_video_position_table_is_the_rollouts_table():
    """The 3-D table must be built exactly as vllm-omni builds it.

    Both sides index the same ``t * side**2 + h * side + w`` rows and the
    checkpoint replaces the values at load time, so the shape and the
    ``(t, h, w)`` dimension split are what have to agree.
    """
    from vllm_omni.diffusion.models.lance.lance_transformer import LancePositionEmbedding3D as RolloutTable

    from verl_omni.pipelines.lance_flow_grpo.lance_model import LancePositionEmbedding3D as TrainerTable

    rollout_table = RolloutTable(3, 4, 48)
    trainer_table = TrainerTable(3, 4, 48)
    assert trainer_table.pos_embed.shape == rollout_table.pos_embed.shape == (3 * 4 * 4, 48)
    assert torch.equal(trainer_table.pos_embed, rollout_table.pos_embed)


def test_video_checkpoint_sizes_the_position_table_from_the_checkpoint(tmp_path):
    """A multi-frame table must load into a model built for that many frames.

    ``Lance_3B_Video`` stores ``31 * 64 * 64`` rows where the image table has
    ``64 * 64``; a trainer that always builds the image table fails the load
    with a size mismatch.
    """
    ckpt = _write_tiny_checkpoint(str(tmp_path), max_latent_size=64, num_frames=3)
    model = LanceForTraining.from_pretrained(ckpt)
    assert model.latent_pos_embed.pos_embed.shape == (3 * 64 * 64, TINY_LLM_CONFIG["hidden_size"])


def test_trainer_rotary_is_the_rollouts_rotary():
    """The assembled cos/sin must equal what the rollout's module produces."""
    from vllm_omni.diffusion.models.bagel.bagel_transformer import BagelRotaryEmbedding

    head_dim, theta, section = 128, 1_000_000.0, [16, 24, 24]
    rollout_config = SimpleNamespace(
        head_dim=head_dim,
        hidden_size=2048,
        num_attention_heads=16,
        rope_theta=theta,
        max_position_embeddings=32768,
        rope_scaling={"rope_type": "mrope", "mrope_section": section},
    )
    rollout_rotary = BagelRotaryEmbedding(rollout_config)
    trainer_rotary = LanceRotaryEmbedding(head_dim, theta=theta, mrope_section=section)

    positions = torch.stack([torch.full((40,), 8), torch.arange(40) % 7 + 8, torch.arange(40) % 5 + 8]).unsqueeze(0)
    cos_rollout, sin_rollout = rollout_rotary(torch.zeros(1, 40, head_dim), positions)
    cos_trainer, sin_trainer = trainer_rotary(positions)

    torch.testing.assert_close(cos_trainer.float(), cos_rollout.float(), rtol=0, atol=0)
    torch.testing.assert_close(sin_trainer.float(), sin_rollout.float(), rtol=0, atol=0)


def test_scalar_positions_still_take_the_1d_path():
    """The text prefix must keep BAGEL's rotary, unchanged."""
    head_dim = 16
    trainer_rotary = LanceRotaryEmbedding(head_dim, mrope_section=(2, 3, 3))
    plain = RotaryEmbedding(head_dim)

    positions = torch.arange(12).unsqueeze(0)
    for mine, theirs in zip(trainer_rotary(positions), plain(positions), strict=False):
        torch.testing.assert_close(mine.float(), theirs.float(), rtol=0, atol=0)


def test_the_mrope_section_is_the_one_the_checkpoint_declares(tmp_path):
    """Read the split from the checkpoint rather than restating the constant."""
    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    config = LanceTrainingConfig.from_model_path(ckpt_dir)
    assert config.mrope_section == tuple(TINY_LLM_CONFIG["rope_scaling"]["mrope_section"])
    assert sum(LANCE_MROPE_SECTION) * 2 == 128, "the released head_dim is 2048 / 16"


# ---------------------------------------------------------------------------
# Config, checkpoint layout and forward
# ---------------------------------------------------------------------------


def test_latent_geometry_matches_the_released_checkpoint():
    """Wan2.2 geometry: no latent patch, 48 channels, 16x downsample."""
    config = _tiny_config()
    assert config.latent_patch_size == 1
    assert config.latent_channel == 48
    assert config.vae_downsample == 16
    # ``vae2llm`` is (hidden, patch_latent_dim); the release ships (2048, 48).
    assert config.patch_latent_dim == 48


@pytest.mark.parametrize("subdir", ["Lance_3B", "Lance_3B_Video", None])
def test_resolve_checkpoint_dir_accepts_bundle_root_and_subdir(tmp_path, subdir):
    ckpt_dir = _write_tiny_checkpoint(str(tmp_path), subdir=subdir)
    assert resolve_checkpoint_dir(str(tmp_path)) == ckpt_dir
    assert resolve_checkpoint_dir(ckpt_dir) == ckpt_dir


def test_resolve_checkpoint_dir_reports_a_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="llm_config.json"):
        resolve_checkpoint_dir(str(tmp_path))


def test_config_reads_llm_dimensions_from_the_checkpoint(tmp_path):
    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    config = LanceTrainingConfig.from_model_path(ckpt_dir)

    assert config.hidden_size == TINY_LLM_CONFIG["hidden_size"]
    assert config.num_hidden_layers == TINY_LLM_CONFIG["num_hidden_layers"]
    assert config.num_key_value_heads == TINY_LLM_CONFIG["num_key_value_heads"]
    # Latent geometry is Lance's, not anything the LLM config carries.
    assert config.latent_patch_size == 1
    assert config.vae_downsample == 16


def test_config_takes_boundary_tokens_from_the_llm_config(tmp_path, monkeypatch):
    """The released config carries the vision token IDs; prefer them.

    Reading them here avoids loading the tokenizer, and keeps the trainer's
    boundary tokens identical to the ones the rollout embeds.
    """
    from unittest.mock import Mock

    from transformers import AutoTokenizer

    load_tokenizer = Mock()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    config = LanceTrainingConfig.from_model_path(ckpt_dir)

    assert config.start_of_image_id == TINY_LLM_CONFIG["vision_start_token_id"]
    assert config.end_of_image_id == TINY_LLM_CONFIG["vision_end_token_id"]
    load_tokenizer.assert_not_called()


@pytest.mark.parametrize("trust_remote_code", [False, True])
@pytest.mark.parametrize("missing_id", ["vision_start_token_id", "vision_end_token_id"])
def test_build_module_preserves_tokenizer_trust_policy(tmp_path, monkeypatch, trust_remote_code, missing_id):
    """Either missing boundary token must use the caller's tokenizer trust policy."""
    from unittest.mock import Mock

    from transformers import AutoTokenizer

    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion
    from verl_omni.workers.config import DiffusionModelConfig

    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    config = dict(TINY_LLM_CONFIG)
    config.pop(missing_id)
    with open(os.path.join(ckpt_dir, "llm_config.json"), "w") as f:
        json.dump(config, f)
    tokenizer = Mock(unk_token_id=0)
    tokenizer.convert_tokens_to_ids.side_effect = [198, 199]
    load_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "local_path", str(tmp_path))
    object.__setattr__(model_config, "trust_remote_code", trust_remote_code)

    model = LanceDiffusion.build_module(model_config, torch.float32)

    load_tokenizer.assert_called_once_with(ckpt_dir, trust_remote_code=trust_remote_code)
    assert (model.config.start_of_image_id, model.config.end_of_image_id) == (198, 199)


def test_from_pretrained_disables_remote_tokenizer_code_by_default(tmp_path, monkeypatch):
    """Direct checkpoint loading must not implicitly trust custom tokenizer code."""
    from unittest.mock import Mock

    from transformers import AutoTokenizer

    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    config = dict(TINY_LLM_CONFIG)
    config.pop("vision_start_token_id")
    with open(os.path.join(ckpt_dir, "llm_config.json"), "w") as f:
        json.dump(config, f)
    tokenizer = Mock(unk_token_id=0)
    tokenizer.convert_tokens_to_ids.side_effect = [198, 199]
    load_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)

    LanceForTraining.from_pretrained(str(tmp_path), torch_dtype=torch.float32)

    load_tokenizer.assert_called_once_with(ckpt_dir, trust_remote_code=False)


def test_checkpoint_mapping_keeps_the_transformer_and_drops_the_rest():
    state_dict = {
        "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": torch.zeros(2, 2),
        "language_model.model.embed_tokens.weight": torch.zeros(2, 2),
        "language_model.lm_head.weight": torch.zeros(2, 2),
        "vit_model.blocks.0.attn.qkv.weight": torch.zeros(2, 2),
        "vae2llm.weight": torch.zeros(2, 2),
        "llm2vae.bias": torch.zeros(2),
        "time_embedder.mlp.0.weight": torch.zeros(2, 2),
        "latent_pos_embed.pos_embed": torch.zeros(2, 2),
    }
    mapped = map_lance_checkpoint_to_training(state_dict)

    assert set(mapped) == {
        "layers.0.self_attn.q_proj_moe_gen.weight",
        "embed_tokens.weight",
        "vae2llm.weight",
        "llm2vae.bias",
        "time_embedder.mlp.0.weight",
        "latent_pos_embed.pos_embed",
    }


def test_from_pretrained_loads_every_trainable_tensor(tmp_path):
    """Every transformer tensor must arrive, by value.

    Comparing against a freshly built module of the same architecture would
    only compare shapes and would pass even if the checkpoint were never
    applied, so the checkpoint is read back and compared element by element.
    """
    from safetensors.torch import load_file

    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    model = LanceForTraining.from_pretrained(str(tmp_path), torch_dtype=torch.float32)

    on_disk = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    expected = map_lance_checkpoint_to_training(on_disk)
    assert expected, "the fixture must carry transformer tensors"
    loaded = model.state_dict()
    for name, tensor in expected.items():
        assert name in loaded, name
        torch.testing.assert_close(loaded[name].float(), tensor.float(), rtol=0, atol=0)
    # and nothing the checkpoint provides may be left at its initial value
    assert set(expected) >= {n for n, _ in model.named_parameters() if "moe_gen" in n}


def test_from_pretrained_takes_shapes_from_the_checkpoint(tmp_path):
    """The released vocabulary is extended past ``llm_config.json``."""
    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    with open(os.path.join(ckpt_dir, "llm_config.json")) as f:
        llm_config = json.load(f)
    llm_config["vocab_size"] = TINY_LLM_CONFIG["vocab_size"] - 8
    with open(os.path.join(ckpt_dir, "llm_config.json"), "w") as f:
        json.dump(llm_config, f)

    model = LanceForTraining.from_pretrained(str(tmp_path), torch_dtype=torch.float32)
    assert model.config.vocab_size == TINY_LLM_CONFIG["vocab_size"]
    assert model.embed_tokens.weight.shape[0] == TINY_LLM_CONFIG["vocab_size"]


@pytest.mark.parametrize("key", ["language_model.model.layers.0.self_attn.q_proj_moe_gen.weight", "vae2llm.weight"])
def test_from_pretrained_rejects_missing_training_weights(tmp_path, key):
    from safetensors.torch import load_file

    ckpt_dir = _write_tiny_checkpoint(str(tmp_path))
    path = os.path.join(ckpt_dir, "model.safetensors")
    weights = load_file(path)
    del weights[key]
    save_file(weights, path)

    with pytest.raises(RuntimeError, match="Missing key"):
        LanceForTraining.from_pretrained(str(tmp_path), torch_dtype=torch.float32)


@pytest.mark.parametrize("short_length", [0, 3, 7])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("shared_grid", [False, True])
def test_batched_forward_and_gradients_match_individual_samples(short_length, checkpointing, shared_grid):
    """Padding and other samples' grids must not change a sample's policy."""
    torch.manual_seed(222)
    config = _tiny_config()
    model = LanceForTraining(config).float()
    if checkpointing:
        model.enable_gradient_checkpointing()
    latents = torch.randn(2, 4, config.patch_latent_dim)
    timesteps = torch.tensor([0.6, 0.4])
    # Same token count, different spatial coordinates for the second sample.
    grids = torch.tensor([[0, 1, 8, 9], [0, 1, 2, 3]])
    if shared_grid:
        grids[1] = grids[0]
    tokens = torch.randint(1, 190, (2, 7))
    lengths = [short_length, 7]
    mask = torch.arange(7).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)

    individual = []
    for row, length in enumerate(lengths):
        output = model(
            hidden_states=latents[row : row + 1],
            timestep=timesteps[row : row + 1],
            text_token_ids=tokens[row : row + 1, :length] if length else None,
            latent_pos_ids=grids[row : row + 1],
        )[0]
        individual.append(output.detach())
        output.square().sum().backward()
    expected_grads = {name: p.grad.clone() if p.grad is not None else None for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    batched = model(
        hidden_states=latents,
        timestep=timesteps,
        text_token_ids=tokens,
        text_attention_mask=mask,
        latent_pos_ids=grids,
    )[0]
    expected_all = torch.cat(individual)
    error = (batched - expected_all).norm()
    tolerance = torch.finfo(torch.bfloat16).eps * expected_all.norm()
    assert error <= tolerance, f"forward error {error.item()} > {tolerance.item()}"
    batched.square().sum().backward()
    for name, parameter in model.named_parameters():
        expected = expected_grads[name]
        assert (parameter.grad is None) == (expected is None), name
        if expected is not None:
            # BAGEL casts attention to bf16 even for a float32 model. Masked
            # and unmasked SDPA backward reductions can round differently.
            # Bound relative gradient-norm error by one bf16 epsilon.
            error = (parameter.grad - expected).norm()
            tolerance = torch.finfo(torch.bfloat16).eps * expected.norm() + 1e-5
            assert error <= tolerance, f"{name}: gradient error {error.item()} > {tolerance.item()}"

    # The padded-batch paths above are compared under a bf16 bound because the
    # single-sample and batched calls reach different SDPA kernels.  The two
    # claims the padding has to satisfy exactly are checked without any bound:
    # a sample's text masked rather than truncated, and a different neighbour.
    alone_masked = model(
        hidden_states=latents[:1],
        timestep=timesteps[:1],
        text_token_ids=tokens[:1],
        text_attention_mask=mask[:1],
        latent_pos_ids=grids[:1],
    )[0]
    assert torch.equal(alone_masked, individual[0]), "masking padding must equal truncating it"

    neighbour_tokens = (tokens[1:2] + 1) % (config.vocab_size - 1) + 1
    neighbour_grids = torch.tensor([[0, 1, 4, 5]])
    swapped = model(
        hidden_states=latents,
        timestep=timesteps,
        text_token_ids=torch.cat([tokens[:1], neighbour_tokens]),
        text_attention_mask=mask,
        latent_pos_ids=torch.cat([grids[:1], neighbour_grids]),
    )[0]
    assert torch.equal(swapped[:1], batched[:1]), "another sample must not change this one"


def test_forward_returns_velocity_shaped_like_the_latent():
    config = _tiny_config()
    model = LanceForTraining(config).to(torch.float32)

    batch, latent_len = 2, 9
    hidden_states = torch.randn(batch, latent_len, config.patch_latent_dim)
    timestep = torch.rand(batch)
    text_token_ids = torch.randint(0, config.vocab_size, (batch, 5))
    latent_pos_ids = torch.arange(latent_len).unsqueeze(0).expand(batch, -1)

    (velocity,) = model(
        hidden_states=hidden_states,
        timestep=timestep,
        text_token_ids=text_token_ids,
        latent_pos_ids=latent_pos_ids,
        text_attention_mask=torch.ones(batch, 5, dtype=torch.bool),
    )
    assert velocity.shape == (batch, latent_len, config.patch_latent_dim)

    # The unconditional CFG branch drops the text context entirely.
    (uncond_velocity,) = model(
        hidden_states=hidden_states,
        timestep=timestep,
        text_token_ids=None,
        latent_pos_ids=latent_pos_ids,
    )
    assert uncond_velocity.shape == velocity.shape


def test_the_architecture_name_resolves_on_both_sides():
    """``+actor_rollout_ref.model.architecture=...`` has to find both adapters."""
    import verl_omni.pipelines  # noqa: F401 - registers every adapter
    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion
    from verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter import LancePipelineWithLogProb
    from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase

    name = "OmniLanceForConditionalGeneration"
    assert DiffusionModelBase.get_class_by_name(name, algorithm="flow_grpo") is LanceDiffusion
    assert VllmOmniPipelineBase.get_class(name, algorithm="flow_grpo") is LancePipelineWithLogProb


def test_rollout_wrapper_keeps_lance_construction_and_bagel_rl_forward():
    """The RL wrapper must add BAGEL's prep without losing Lance's pipeline.

    ``LancePipeline.__init__`` deliberately never calls up the MRO, so the
    composition only works if the RL class comes first: its ``forward`` runs
    the trajectory/SDE setup and its ``super().forward`` lands on Lance.
    """
    from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
    from vllm_omni.diffusion.models.lance.pipeline_lance import LANCE_DEFAULTS, LancePipeline

    from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb
    from verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter import LancePipelineWithLogProb

    mro = LancePipelineWithLogProb.__mro__
    assert mro[1] is BagelPipelineWithLogProb
    assert mro[2] is LancePipeline
    assert mro[3] is BagelPipeline
    assert LancePipelineWithLogProb.forward is BagelPipelineWithLogProb.forward
    # The shift the adapter applies must be the one the pipeline defaults to.
    assert LancePipelineWithLogProb.flowgrpo_timestep_shift == LANCE_DEFAULTS.timestep_shift
    assert LancePipelineWithLogProb.flowgrpo_timestep_shift != BagelPipelineWithLogProb.flowgrpo_timestep_shift


def test_only_the_generation_expert_trains():
    """LoRA and full fine-tuning both target the ``moe_gen`` pathway."""
    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

    model = LanceForTraining(_tiny_config())
    LanceDiffusion.configure_trainable_params(model, model_config=None)

    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert all("moe_gen" in name for name in trainable)
    assert any("q_proj_moe_gen" in name for name in trainable)


# ---------------------------------------------------------------------------
# Weight sync
# ---------------------------------------------------------------------------

#: LoRA settings small enough for CPU; only the rank matters to the scaling.
_LORA_RANK = 4
_LORA_ALPHA = 8
#: The projections the example recipe adapts.
_LORA_TARGETS = [
    "q_proj_moe_gen",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
    "o_proj_moe_gen",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
]


def _build_rollout_language_model(monkeypatch):
    """Build the MoT language model the rollout holds, on CPU.

    The rollout reaches its parameters through vllm-omni, so the packing the
    weight sync has to hit -- fused ``qkv_proj``, ``gate_up_proj``, the
    ``gen_exp`` sub-modules and ``MoTRMSNorm.gen_weight`` -- can only be
    checked against the real module.  Two things stand in the way on CPU: the
    diffusion attention backend is chosen per platform and no CPU entry
    exists, and the model needs a parallel group.  Pin the SDPA backend the
    attention layer already carries as its float32 fallback (it holds no
    parameters, so the parameter set is the one a GPU rollout would hold) and
    set up a single-process group.  Skip if either is unavailable.
    """
    pytest.importorskip("vllm_omni")
    import vllm_omni.diffusion.attention.layer as attention_layer
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, parallel_state
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.models.bagel.bagel_transformer import Qwen2MoTConfig, Qwen2MoTForCausalLM

    monkeypatch.setattr(attention_layer, "get_attn_backend_for_role", lambda **_: (SDPABackend, None))

    try:
        if not torch.distributed.is_initialized():
            for key, value in {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29577",
                "RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_RANK": "0",
            }.items():
                monkeypatch.setenv(key, value)
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29577",
                local_rank=0,
                backend="gloo",
            )
    except (RuntimeError, OSError, ValueError) as exc:  # no loopback in the sandbox
        pytest.skip(f"no single-process parallel group available: {exc}")

    config = _tiny_config()
    with set_current_vllm_config(VllmConfig()):
        if parallel_state._TP is None:
            initialize_model_parallel(tensor_model_parallel_size=1)
        rollout_config = Qwen2MoTConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            # The release ships a separate ``language_model.lm_head.weight``.
            tie_word_embeddings=False,
        )
        rollout_config.layer_module = "Qwen2MoTDecoderLayer"
        return Qwen2MoTForCausalLM(rollout_config, prefix="bagel.language_model")


def _sync_to_rollout(rollout, named_tensors):
    """Run the export through the adapter's routing and report what it skipped."""
    from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb

    skipped: list[str] = []
    import vllm_omni.diffusion.models.bagel.bagel_transformer as bagel_transformer

    original = bagel_transformer.logger.warning_once
    bagel_transformer.logger.warning_once = lambda msg, *args, **kwargs: skipped.append(args[0] if args else msg)
    try:
        # The adapter routes every ``transformer.``-prefixed tensor here; the
        # rewrite is the one BagelPipelineWithLogProb.load_weights performs.
        # The adapter rewrites ``transformer.X`` to ``model.X`` and hands the
        # result to the language model; the pipeline itself cannot be built
        # without a GPU, so the rewrite is pinned here and applied below.
        source = inspect.getsource(BagelPipelineWithLogProb.load_weights)
        assert 'name.startswith("transformer.")' in source
        assert "f\"model.{name[len('transformer.') :]}\"" in source
        loaded = rollout.load_weights([(f"model.{name}", tensor) for name, tensor in named_tensors])
    finally:
        bagel_transformer.logger.warning_once = original
    return loaded, skipped


def test_weight_sync_reaches_every_trainable_parameter(monkeypatch):
    """Nothing the generation expert learns is dropped on the way to the rollout.

    ``load_weights`` skips a name it cannot place with a warning rather than
    an error, so a mismatch between the trainer's parameter names and the
    rollout's would cost nothing at sync time and silently train against a
    stale policy.  Assert the trainable set survives the trip.
    """
    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

    rollout = _build_rollout_language_model(monkeypatch)
    trainer = LanceForTraining(_tiny_config())
    LanceDiffusion.configure_trainable_params(trainer, model_config=None)
    trainable = {name for name, param in trainer.named_parameters() if param.requires_grad}

    with torch.no_grad():  # an unwritten parameter must not pass as correct
        for param in rollout.parameters():
            param.fill_(float("nan"))
    _, skipped = _sync_to_rollout(rollout, [(n, p.detach().clone()) for n, p in trainer.named_parameters()])

    assert not [name for name in skipped if "moe_gen" in name]
    assert not trainable & {name.removeprefix("model.") for name in skipped}
    # A name that lands nowhere is only warned about, and a renamed projection
    # can still match the text pattern by substring and leave the generation
    # half untouched, so check the destinations rather than the warnings.  The
    # LM head is the one parameter flow matching never reads.
    unwritten = {name for name, param in rollout.named_parameters() if torch.isnan(param).any()}
    assert unwritten == {"lm_head.weight"}
    # What the rollout does not receive is the diffusion adapter, which lives
    # outside the language model and which this recipe freezes.
    assert {name.removeprefix("model.").split(".")[0] for name in skipped} == {
        "time_embedder",
        "vae2llm",
        "llm2vae",
        "latent_pos_embed",
    }


def test_merged_lora_lands_in_the_rollout_packed_weights(monkeypatch):
    """A merged LoRA update arrives in the fused projections, in the right half.

    The trainer keeps q/k/v and gate/up apart; the rollout fuses them.  The
    remap is by substring, and ``_moe_gen`` names contain the text names, so a
    wrong ordering would still load and would only show up as a policy that
    does not improve.
    """
    peft = pytest.importorskip("peft")
    from verl.utils.fsdp_utils import _merge_or_unmerge_lora_, normalize_peft_param_name
    from verl.utils.model import convert_weight_keys

    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

    rollout = _build_rollout_language_model(monkeypatch)
    torch.manual_seed(0)
    trainer = LanceForTraining(_tiny_config())
    for param in trainer.parameters():
        with torch.no_grad():
            param.normal_(0.0, 0.5)
    LanceDiffusion.configure_trainable_params(trainer, model_config=None)
    base = {name: param.detach().clone().float() for name, param in trainer.named_parameters()}

    model = peft.get_peft_model(
        trainer,
        peft.LoraConfig(
            r=_LORA_RANK, lora_alpha=_LORA_ALPHA, target_modules=_LORA_TARGETS, lora_dropout=0.0, bias="none"
        ),
    )
    # PEFT zero-initialises ``lora_B``, so an export that dropped the adapter
    # would be indistinguishable from the base weights.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 0.1)

    # What ``_merged_lora_per_tensor_param`` streams.  ``merged_lora_context``
    # wraps the merge in an FSDP unshard, which has nothing to do without
    # sharding; the merge itself is ``_merge_or_unmerge_lora_``.
    _merge_or_unmerge_lora_(model, merge=True)
    exported = convert_weight_keys(normalize_peft_param_name(model.state_dict()), model)
    exported = {name: tensor.detach().clone() for name, tensor in exported.items()}
    _merge_or_unmerge_lora_(model, merge=False)

    assert not [name for name in exported if "lora_" in name or "base_layer" in name or "base_model" in name]

    with torch.no_grad():  # an unwritten parameter must not pass as correct
        for param in rollout.parameters():
            param.fill_(float("nan"))
    _sync_to_rollout(rollout, exported.items())
    synced = dict(rollout.named_parameters())

    state = dict(model.state_dict())

    def merged(name: str) -> torch.Tensor:
        prefix = "base_model.model." + name.removesuffix(".weight")
        lora_a = state[f"{prefix}.lora_A.default.weight"].float()
        lora_b = state[f"{prefix}.lora_B.default.weight"].float()
        return base[name] + (lora_b @ lora_a) * (_LORA_ALPHA / _LORA_RANK)

    assert (
        merged("layers.0.self_attn.q_proj_moe_gen.weight")
        .sub(base["layers.0.self_attn.q_proj_moe_gen.weight"])
        .abs()
        .max()
        > 1e-3
    ), "the adapter has to move the weights for this test to mean anything"

    for layer in range(_tiny_config().num_hidden_layers):
        attn = f"model.layers.{layer}.self_attn"
        torch.testing.assert_close(
            synced[f"{attn}.qkv_proj.gen_exp.weight"].float(),
            torch.cat([merged(f"layers.{layer}.self_attn.{p}_proj_moe_gen.weight") for p in "qkv"]),
        )
        torch.testing.assert_close(
            synced[f"{attn}.o_proj.gen_exp.weight"].float(),
            merged(f"layers.{layer}.self_attn.o_proj_moe_gen.weight"),
        )
        torch.testing.assert_close(
            synced[f"model.layers.{layer}.mlp_moe_gen.gate_up_proj.weight"].float(),
            torch.cat(
                [
                    merged(f"layers.{layer}.mlp_moe_gen.gate_proj.weight"),
                    merged(f"layers.{layer}.mlp_moe_gen.up_proj.weight"),
                ]
            ),
        )
        # The frozen text expert has to arrive untouched, in its own half.
        torch.testing.assert_close(
            synced[f"{attn}.qkv_proj.weight"].float(),
            torch.cat([base[f"layers.{layer}.self_attn.{p}_proj.weight"] for p in "qkv"]),
        )
        torch.testing.assert_close(
            synced[f"model.layers.{layer}.input_layernorm.gen_weight"].float(),
            base[f"layers.{layer}.input_layernorm_moe_gen.weight"],
        )


@pytest.mark.parametrize("missing", [None, 0])
def test_config_rejects_missing_vision_token_ids(tmp_path, monkeypatch, missing):
    from unittest.mock import Mock

    from transformers import AutoTokenizer

    config = dict(TINY_LLM_CONFIG)
    config.pop("vision_start_token_id")
    config.pop("vision_end_token_id")
    (tmp_path / "llm_config.json").write_text(json.dumps(config))
    tokenizer = Mock(unk_token_id=0)
    tokenizer.convert_tokens_to_ids.side_effect = [missing, 199]
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    with pytest.raises(ValueError, match="missing vision boundary tokens"):
        LanceTrainingConfig.from_model_path(str(tmp_path))


def test_config_propagates_tokenizer_load_failure(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from transformers import AutoTokenizer

    config = dict(TINY_LLM_CONFIG)
    config.pop("vision_start_token_id")
    config.pop("vision_end_token_id")
    (tmp_path / "llm_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(side_effect=OSError("broken tokenizer")))
    with pytest.raises(OSError, match="broken tokenizer"):
        LanceTrainingConfig.from_model_path(str(tmp_path))


def _synthetic_edit_condition(
    batch: int = 1, prefix: int = 3, ref: int = 9, tail: int = 4, latent: int = 4, hidden: int = 64
):
    """A condition bundle shaped like the rollout's image-edit export."""
    return {
        "condition_prefix_ids": torch.randint(1, 190, (batch, prefix)),
        "condition_prefix_positions": torch.arange(prefix).expand(batch, -1),
        "condition_ref_rows": torch.randn(batch, ref, hidden),
        "condition_ref_positions": torch.arange(prefix, prefix + ref).expand(3, -1).unsqueeze(0).expand(batch, -1, -1),
        "condition_ref_is_gen": torch.cat(
            [
                torch.zeros(batch, ref - latent - 1, dtype=torch.bool),
                torch.ones(batch, latent, dtype=torch.bool),
                torch.zeros(batch, 1, dtype=torch.bool),
            ],
            dim=1,
        ),
        "condition_tail_ids": torch.randint(1, 190, (batch, tail)),
        "condition_tail_positions": torch.arange(prefix + ref, prefix + ref + tail).expand(batch, -1),
        "condition_tail_mask": torch.ones(batch, tail, dtype=torch.bool),
        "condition_latent_positions": torch.zeros(batch, 3, latent + 2, dtype=torch.long),
        "condition_latent_grid": torch.arange(latent).expand(batch, -1),
    }


def test_lance_replays_an_edit_condition_from_the_rollout():
    """Exported reference rows, their routing and the latent basis all matter.

    ``_forward_image_edit`` prefills the reference through the ViT and the VAE,
    which this model cannot do; the rows and the positions they sat at travel
    with the trajectory, and the replay has to use them rather than rebuild a
    context of its own.
    """
    config = _tiny_config()
    config.max_latent_size = 8
    model = LanceForTraining(config).float()
    latent = torch.randn(1, 4, config.patch_latent_dim)
    timestep = torch.tensor([0.6])
    condition = _synthetic_edit_condition()

    (velocity,) = model(hidden_states=latent, timestep=timestep, condition=condition)
    assert velocity.shape == (1, 4, config.patch_latent_dim)
    assert torch.isfinite(velocity).all()

    # The reference rows are part of the context, not decoration.
    perturbed = {**condition, "condition_ref_rows": condition["condition_ref_rows"] + 1.0}
    (other,) = model(hidden_states=latent, timestep=timestep, condition=perturbed)
    assert not torch.allclose(velocity, other)

    # The latent block's rotary basis is the rollout's, used verbatim.
    shifted = {**condition, "condition_latent_positions": condition["condition_latent_positions"] + 5}
    (moved,) = model(hidden_states=latent, timestep=timestep, condition=shifted)
    assert not torch.allclose(velocity, moved)


def test_lance_edit_condition_ignores_padded_tail_slots():
    """Padding only keeps the batch rectangular, so it must not be attended to."""
    config = _tiny_config()
    config.max_latent_size = 8
    model = LanceForTraining(config).float()
    latent = torch.randn(1, 4, config.patch_latent_dim)
    timestep = torch.tensor([0.6])
    condition = _synthetic_edit_condition()
    condition["condition_tail_mask"] = torch.tensor([[True, True, False, False]])
    condition["condition_tail_positions"] = condition["condition_tail_positions"].clone()
    condition["condition_tail_positions"][0, 2:] = 0

    (velocity,) = model(hidden_states=latent, timestep=timestep, condition=condition)

    noisy = {**condition, "condition_tail_ids": condition["condition_tail_ids"].clone()}
    noisy["condition_tail_ids"][0, 2:] = 190  # outside the real instruction
    (other,) = model(hidden_states=latent, timestep=timestep, condition=noisy)

    assert torch.allclose(velocity, other), "masked tail slots changed the prediction"


def test_lance_positions_honor_the_rollouts_anchor():
    """The latent block sits where the rollout put it, not after the text.

    Lance's edit modes anchor the noise block at the reference VAE block's
    positions, which is what lets the model map noise tokens onto reference
    tokens; a replay that assumed the text length would rotate the whole block.
    """
    config = _tiny_config()
    config.max_latent_size = 64
    model = LanceForTraining(config)
    latent_pos_ids = get_flattened_position_ids(_IMAGE_HW, _IMAGE_HW, _LATENT_DOWNSAMPLE, 64)
    latent_pos_ids = latent_pos_ids.unsqueeze(0).expand(2, -1)
    num_text = 5
    anchor = torch.tensor([64, 96])

    default = model.build_position_ids(2, num_text, latent_pos_ids.shape[1], latent_pos_ids, torch.device("cpu"))
    shifted = model.build_position_ids(
        2,
        num_text,
        latent_pos_ids.shape[1],
        latent_pos_ids,
        torch.device("cpu"),
        text_attention_mask=torch.ones(2, num_text, dtype=torch.bool),
        position_anchor=anchor,
    )

    # The text rows are untouched; every block row moves by the anchor delta.
    assert torch.equal(default[:, :, :num_text], shifted[:, :, :num_text])
    delta = (anchor - num_text).reshape(2, 1, 1)
    assert torch.equal(shifted[:, :, num_text:], default[:, :, num_text:] + delta)


@pytest.mark.parametrize("is_video,reference", [(False, "image"), (True, "image"), (True, "video")])
def test_lance_condition_routing_keeps_reference(is_video, reference):
    from verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter import LancePipelineWithLogProb

    pipeline = object.__new__(LancePipelineWithLogProb)
    pipeline._is_video = is_video
    image = object()
    media = {reference: image}
    request = SimpleNamespace(
        prompts=[{"prompt": "edit this", "multi_modal_data": media}], sampling_params=SimpleNamespace(extra_args={})
    )
    pipeline._ensure_bagel_prompt_text(request)
    expected = "first_frame" if is_video and reference == "image" else reference
    assert request.prompts[0]["multi_modal_data"][expected] is image
    assert media == {reference: image}


def test_lance_rejects_ambiguous_video_reference():
    from verl_omni.pipelines.lance_flow_grpo.vllm_omni_rollout_adapter import LancePipelineWithLogProb

    pipeline = object.__new__(LancePipelineWithLogProb)
    pipeline._is_video = True
    request = SimpleNamespace(
        prompts=[{"prompt": "edit", "multi_modal_data": {"image": object(), "video": object()}}],
        sampling_params=SimpleNamespace(extra_args={}),
    )
    with pytest.raises(ValueError, match="one reference"):
        pipeline._ensure_bagel_prompt_text(request)


def test_i2v_replay_uses_zero_sigma_only_on_pinned_tokens():
    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

    conditional = {"hidden_states": torch.zeros(2, 6, 48), "timestep": torch.tensor([0.7, 0.3])}
    unconditional = {**conditional}
    original = conditional["timestep"].clone()
    LanceDiffusion._set_reference_timesteps(
        conditional, unconditional, {"frame_condition_token_indexes": torch.tensor([[0, 1], [0, 1]])}
    )
    for inputs in (conditional, unconditional):
        torch.testing.assert_close(inputs["timestep"][:, :2], torch.zeros(2, 2))
        torch.testing.assert_close(inputs["timestep"][:, 2:], original[:, None].expand(-1, 4))
    torch.testing.assert_close(original, torch.tensor([0.7, 0.3]))


def test_edit_attention_matches_ordered_prefill_visibility(monkeypatch):
    import torch.nn.functional as functional

    model = LanceForTraining(_tiny_config()).float()
    condition = _synthetic_edit_condition()
    calls = []
    sdpa = functional.scaled_dot_product_attention

    def record(query, key, value, **kwargs):
        # Compare visibility in fp32, independently of bf16 rounding at segment boundaries.
        result = sdpa(query.float(), key.float(), value.float(), **kwargs)
        calls.append((query.float(), key.float(), value.float(), result))
        return result.to(query.dtype)

    monkeypatch.setattr(functional, "scaled_dot_product_attention", record)
    model(hidden_states=torch.randn(1, 4, 48), timestep=torch.tensor([0.6]), condition=condition)
    # Prefix, ViT, VAE, instruction and generated latent are distinct prefills.
    mask = torch.zeros(22, 22, dtype=torch.bool)
    mask[:3, :3] = torch.ones(3, 3, dtype=torch.bool).tril()
    mask[3:6, :6] = True
    mask[6:12, :12] = True
    mask[12:16, :16] = torch.arange(16)[None] <= torch.arange(12, 16)[:, None]
    mask[16:, :] = True
    assert len(calls) == 5 * len(model.layers)
    for layer in range(len(model.layers)):
        blocks = calls[layer * 5 : (layer + 1) * 5]
        queries = torch.cat([block[0] for block in blocks], dim=2)
        expected = sdpa(queries, blocks[-1][1], blocks[-1][2], attn_mask=mask[None, None])
        actual = torch.cat([block[3] for block in blocks], dim=2)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_i2v_pinned_indexes_stay_sample_local():
    from verl_omni.pipelines.lance_flow_grpo.diffusers_training_adapter import LanceDiffusion

    inputs = {"hidden_states": torch.zeros(2, 6, 48), "timestep": torch.ones(2)}
    negative = {**inputs}
    LanceDiffusion._set_reference_timesteps(
        inputs, negative, {"frame_condition_token_indexes": torch.tensor([[0, 1], [2, 3]])}
    )
    expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
    torch.testing.assert_close(inputs["timestep"], expected)
    torch.testing.assert_close(negative["timestep"], expected)


@pytest.mark.parametrize("pinned", [None, torch.empty(0, dtype=torch.long), torch.empty(2, 0, dtype=torch.long)])
def test_lance_unpinned_scheduler_inputs_are_unchanged(pinned):
    tensors = tuple(torch.randn(2, 6, 4) for _ in range(3))
    metadata = {} if pinned is None else {"frame_condition_token_indexes": pinned}
    selected = LanceDiffusion._select_scheduler_tokens(*tensors, metadata)
    assert all(actual is original for actual, original in zip(selected, tensors, strict=True))


@pytest.mark.parametrize("pinned", [torch.tensor([0, 1]), torch.tensor([[1, 0], [4, 2]])])
@pytest.mark.parametrize("cfg_scale", [1.0, 4.0])
@pytest.mark.parametrize("sde_type", ["sde", "cps", "dance_sde"])
def test_i2v_replay_scores_only_active_tokens_after_full_cfg(pinned, cfg_scale, sde_type):
    scheduler = FlowMatchSDEDiscreteScheduler()
    setup_lance_sigmas(scheduler, 4, device="cpu")
    latents = torch.linspace(-1.0, 1.0, 2 * 2 * 6 * 4).reshape(2, 2, 6, 4)
    conditional = torch.linspace(0.2, 1.0, 2 * 6 * 4).reshape(2, 6, 4).requires_grad_()
    unconditional = torch.linspace(-0.4, 0.1, 2 * 6 * 4).reshape(2, 6, 4).requires_grad_()
    timestep = scheduler.timesteps[1].expand(2)
    calls = []

    def module(hidden_states, text_token_ids):
        assert hidden_states.shape == (2, 6, 4)
        torch.testing.assert_close(hidden_states, latents[:, 0])
        calls.append(text_token_ids)
        return (conditional if text_token_ids is not None else unconditional,)

    config = SimpleNamespace(
        pipeline=SimpleNamespace(cfg_text_scale=cfg_scale, cfg_img_scale=1.0, cfg_renorm_type="global"),
        algo=SimpleNamespace(noise_level=0.7, sde_type=sde_type),
    )
    inputs = {"hidden_states": latents[:, 0], "text_token_ids": torch.ones(2, 1, dtype=torch.long)}
    negative = {**inputs, "text_token_ids": None}
    actual = LanceDiffusion.forward_and_sample_previous_step(
        module,
        scheduler,
        config,
        inputs,
        negative,
        {
            "all_latents": latents,
            "all_timesteps": timestep[:, None],
            "frame_condition_token_indexes": pinned,
        },
        0,
    )
    assert len(calls) == (2 if cfg_scale > 1 else 1)
    velocity = conditional
    if cfg_scale > 1:
        guided = unconditional + cfg_scale * (conditional - unconditional)
        scale = (conditional.norm(dim=(1, 2), keepdim=True) / (guided.norm(dim=(1, 2), keepdim=True) + 1e-8)).clamp(
            min=LanceDiffusion._get_cfg_params(config)["cfg_renorm_min"], max=1.0
        )
        velocity = guided * scale
        assert calls[1] is None
    expected = []
    for row in range(2):
        fixed = pinned if pinned.ndim == 1 else pinned[row]
        active = [index for index in range(6) if index not in fixed.tolist()]
        expected.append(
            scheduler.sample_previous_step(
                sample=latents[row : row + 1, 0, active],
                model_output=velocity[row : row + 1, active],
                timestep=timestep[row : row + 1],
                prev_sample=latents[row : row + 1, 1, active],
                noise_level=0.7,
                sde_type=sde_type,
                return_sqrt_dt=True,
                include_logprob_normalizer=False,
            )
        )
    for index, result in enumerate(actual, start=1):
        torch.testing.assert_close(result, torch.cat([row[index] for row in expected]))
    assert actual[1].shape == (2, 4, 4)
    actual_grad = torch.autograd.grad(
        actual[0].sum(), (conditional, unconditional), allow_unused=True, retain_graph=True
    )
    expected_grad = torch.autograd.grad(
        sum(row[1].sum() for row in expected), (conditional, unconditional), allow_unused=True
    )
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        if expected_value is None:
            assert actual_value is None
        else:
            torch.testing.assert_close(actual_value, expected_value)
            assert actual_value.abs().sum() > 0
    if cfg_scale == 1:
        indexes = pinned.expand(2, -1) if pinned.ndim == 1 else pinned
        assert torch.count_nonzero(actual_grad[0].gather(1, indexes[..., None].expand(-1, -1, 4))) == 0


def test_i2v_replay_rejects_an_entirely_fixed_latent():
    tensors = tuple(torch.randn(2, 6, 4) for _ in range(3))
    with pytest.raises(ValueError, match="at least one unpinned"):
        LanceDiffusion._select_scheduler_tokens(*tensors, {"frame_condition_token_indexes": torch.arange(6)})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "video"])
async def test_lance_agent_preserves_raw_reference_for_pipeline(kind):
    from verl_omni.pipelines.lance_flow_grpo.agent_loop import LanceDiffusionSingleTurnAgentLoop

    agent = object.__new__(LanceDiffusionSingleTurnAgentLoop)
    reference = object() if kind == "image" else "/data/reference.mp4"
    messages = [{"role": "user", "content": [{"type": kind, kind: reference}, {"type": "text", "text": "edit"}]}]
    assert await agent.process_multi_modal_info(messages) == {f"{kind}s": [reference]}


@pytest.mark.asyncio
async def test_lance_agent_rejects_unsupported_reference_combinations():
    from verl_omni.pipelines.lance_flow_grpo.agent_loop import LanceDiffusionSingleTurnAgentLoop

    agent = object.__new__(LanceDiffusionSingleTurnAgentLoop)
    with pytest.raises(ValueError, match="one reference"):
        await agent.ct_build_initial_tokens([], images=[object()], videos=[object()])
    with pytest.raises(ValueError, match="without audio"):
        await agent.ct_build_initial_tokens([], audios=[object()])
