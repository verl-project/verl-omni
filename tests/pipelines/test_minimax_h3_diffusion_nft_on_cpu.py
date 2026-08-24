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
"""CPU tests for the MiniMax H3 DiffusionNFT adapter."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    AUDIO_TAG,
    MINIMAX_H3_TOKEN_ID_NATIVE_KEY,
    TEXT_TAG,
    VIDEO_ROW_WIDTH,
    VIDEO_TAG,
    MiniMaxH3RolloutWeightSyncMixin,
    build_layout_from_meta,
    h3_dit_timestep,
    h3_velocity_to_flow_match,
    pack_video_audio_rows,
    unpack_video_audio_rows,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig

_META = [4, 6, 1, 4, 4, 3]
_NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS = 4, 6
_BATCH = 2
_TEXT_LEN = 12
_TEXT_DIM = 64


def _rows(batch=_BATCH, num_video_rows=_NUM_VIDEO_ROWS, num_audio_rows=_NUM_AUDIO_ROWS):
    video_rows = torch.randn(batch, num_video_rows, VIDEO_ROW_WIDTH)
    audio_rows = torch.randn(batch, num_audio_rows, AUDIO_ROW_WIDTH)
    return video_rows, audio_rows


def _micro_batch(batch=_BATCH):
    return TensorDict({"latent_meta": torch.tensor([_META] * batch, dtype=torch.long)}, batch_size=batch)


def _module(side_effect):
    """Build a mock transformer."""
    module = MagicMock(side_effect=side_effect)
    module.config = None
    return module


def _identity(**kwargs):
    return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def _prepared_inputs(video_rows, audio_rows, timesteps, mask=None):
    packed = pack_video_audio_rows(video_rows, audio_rows)
    batch = video_rows.shape[0]
    if mask is None:
        mask = torch.ones(batch, _TEXT_LEN, dtype=torch.int32)
    return MiniMaxH3DiffusionNFT.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=packed,
        timesteps=timesteps,
        prompt_embeds=torch.randn(batch, _TEXT_LEN, _TEXT_DIM),
        prompt_embeds_mask=mask,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=_micro_batch(batch),
        step=0,
    )


class TestMiniMaxH3DiffusionNFTRegistry:
    def test_registered_for_minimax_h3_diffusion_nft(self):
        resolved = DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", "diffusion_nft")
        assert resolved is MiniMaxH3DiffusionNFT

    def test_importing_the_package_registers_the_algorithm(self):
        # Registration is an import side effect and ``verl_omni/pipelines/__init__.py`` is the only
        # importer, so a subpackage missing from there silently does not exist. Importing the module
        # directly (as the test above does) hides that, hence the package-level check.
        import verl_omni.pipelines as pipelines

        assert hasattr(pipelines, "minimax_h3_diffusion_nft")
        assert DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", "diffusion_nft") is not None


class TestMiniMaxH3PackUnpack:
    def test_batched_round_trip_inverts_exactly(self):
        video_rows, audio_rows = _rows()
        packed = pack_video_audio_rows(video_rows, audio_rows)
        expected_width = _NUM_VIDEO_ROWS * VIDEO_ROW_WIDTH + _NUM_AUDIO_ROWS * AUDIO_ROW_WIDTH
        assert packed.shape == (_BATCH, expected_width)

        out_video, out_audio = unpack_video_audio_rows(packed, _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        assert torch.equal(out_video, video_rows)
        assert torch.equal(out_audio, audio_rows)

    def test_unbatched_inputs_gain_leading_batch_dim(self):
        video_rows = torch.randn(_NUM_VIDEO_ROWS, VIDEO_ROW_WIDTH)
        audio_rows = torch.randn(_NUM_AUDIO_ROWS, AUDIO_ROW_WIDTH)
        packed = pack_video_audio_rows(video_rows, audio_rows)
        assert packed.shape == (1, _NUM_VIDEO_ROWS * VIDEO_ROW_WIDTH + _NUM_AUDIO_ROWS * AUDIO_ROW_WIDTH)

        out_video, out_audio = unpack_video_audio_rows(packed, _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        assert torch.equal(out_video, video_rows.unsqueeze(0))
        assert torch.equal(out_audio, audio_rows.unsqueeze(0))


class TestMiniMaxH3Conventions:
    """Both MiniMax H3 conventions, shared by every H3 algorithm (see ``common.py``)."""

    def test_dit_timestep_mirrors_the_sigma_schedule(self):
        # sigma 1.0 (pure noise) -> t 0.0, sigma 0.0 (clean data) -> t 1.0.
        torch.testing.assert_close(h3_dit_timestep(torch.tensor([1000.0, 750.0, 0.0])), torch.tensor([0.0, 0.25, 1.0]))

    def test_dit_timestep_is_its_own_inverse(self):
        sigmas = torch.tensor([1000.0, 600.0, 120.0, 0.0])
        torch.testing.assert_close(h3_dit_timestep(h3_dit_timestep(sigmas) * 1000.0), sigmas / 1000.0)

    def test_velocity_conversion_flips_the_sign(self):
        velocity = torch.randn(3, VIDEO_ROW_WIDTH)
        torch.testing.assert_close(h3_velocity_to_flow_match(velocity), -velocity)
        torch.testing.assert_close(h3_velocity_to_flow_match(h3_velocity_to_flow_match(velocity)), velocity)

    def test_converted_velocity_recovers_x0_under_the_flow_match_rule(self):
        # H3: x0 = xt + sigma * v. Flow-match consumers: x0 = xt - sigma * model_output.
        # The two agree only for model_output = -v, which is what the helper produces.
        sigma = 0.4
        x0 = torch.randn(2, VIDEO_ROW_WIDTH)
        noise = torch.randn(2, VIDEO_ROW_WIDTH)
        xt = (1.0 - sigma) * x0 + sigma * noise
        velocity = x0 - noise
        torch.testing.assert_close(xt - sigma * h3_velocity_to_flow_match(velocity), x0)


class TestMiniMaxH3BuildLayoutFromMeta:
    def test_row_counts_and_tags_match_meta(self):
        position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = (
            build_layout_from_meta(_META, _TEXT_LEN)
        )
        seq_len = _TEXT_LEN + _NUM_VIDEO_ROWS + _NUM_AUDIO_ROWS
        assert position_ids.shape == (seq_len, 3)
        assert video_indices.shape[0] == _NUM_VIDEO_ROWS
        assert audio_indices.shape[0] == _NUM_AUDIO_ROWS
        assert text_indices.shape[0] == _TEXT_LEN
        assert (num_cond_video, num_cond_audio) == (0, 0)
        # t2va tags: text rows 1, audio rows 2, video rows 0.
        assert torch.equal(token_tags[text_indices], torch.full((_TEXT_LEN,), TEXT_TAG))
        assert torch.equal(token_tags[audio_indices], torch.full((_NUM_AUDIO_ROWS,), AUDIO_TAG))
        assert torch.equal(token_tags[video_indices], torch.full((_NUM_VIDEO_ROWS,), VIDEO_TAG))

    def test_inconsistent_meta_is_rejected(self):
        # audio_t=4 does not divide Na=6 evenly -> derived rows disagree with meta.
        with pytest.raises(ValueError, match="audio rows"):
            build_layout_from_meta([4, 6, 1, 4, 4, 4], _TEXT_LEN)


class TestMiniMaxH3PrepareModelInputs:
    def test_unpacks_latents_into_video_and_audio_rows(self):
        video_rows, audio_rows = _rows()
        model_inputs, negative_model_inputs = _prepared_inputs(
            video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0])
        )

        assert negative_model_inputs is None
        assert torch.equal(model_inputs["video_rows"], video_rows)
        assert torch.equal(model_inputs["audio_rows"], audio_rows)
        assert model_inputs["latent_meta"] == _META

    def test_timestep_is_mirrored_into_the_dit_data_fraction(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))
        # sigma 0.5 -> t 0.5, sigma 0.25 -> t 0.75: the DiT reads t as a data fraction, not a noise level.
        torch.testing.assert_close(model_inputs["timestep"], torch.tensor([0.5, 0.75]))


class TestMiniMaxH3Forward:
    def test_identity_transformer_repacks_to_input_layout(self):
        video_rows, audio_rows = _rows()
        packed = pack_video_audio_rows(video_rows, audio_rows)
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))

        # Echoing the per-sample rows back and re-packing must reproduce the flat ``xt`` layout exactly,
        # negated: H3's velocity is ``x0 - noise`` and the loss expects the opposite convention.
        out = MiniMaxH3DiffusionNFT.forward(
            module=_module(_identity), model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        assert out.shape == packed.shape
        torch.testing.assert_close(out, -packed)

    def test_per_sample_loop_stacks_scaled_velocity_video_then_audio(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))

        module = _module(lambda **kw: (kw["hidden_states"] * 2.0, kw["audio_hidden_states"] * 3.0))
        out = MiniMaxH3DiffusionNFT.forward(
            module=module, model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        assert module.call_count == _BATCH
        torch.testing.assert_close(out, pack_video_audio_rows(video_rows * -2.0, audio_rows * -3.0))

    def test_forward_calls_module_with_real_packed_sequence_kwargs(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))

        module = _module(_identity)
        MiniMaxH3DiffusionNFT.forward(
            module=module, model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        kwargs = module.call_args_list[0].kwargs
        seq_len = _TEXT_LEN + _NUM_VIDEO_ROWS + _NUM_AUDIO_ROWS
        assert kwargs["hidden_states"].shape == (1, _NUM_VIDEO_ROWS, VIDEO_ROW_WIDTH)
        assert kwargs["audio_hidden_states"].shape == (1, _NUM_AUDIO_ROWS, AUDIO_ROW_WIDTH)
        assert kwargs["encoder_hidden_states"].shape == (1, _TEXT_LEN, _TEXT_DIM)
        assert kwargs["timestep"].shape == (1,)
        assert kwargs["token_tags"].shape == (seq_len,)
        assert kwargs["position_ids"].shape == (seq_len, 3)
        assert kwargs["return_dict"] is False
        # Option C: the engine noised the whole packed latent at one level, so every row shares it.
        assert kwargs["timestep_indices"].shape == (seq_len,)
        assert torch.equal(kwargs["timestep_indices"], torch.zeros(seq_len, dtype=torch.long))

    def test_forward_slices_encoder_to_true_text_length(self):
        video_rows, audio_rows = _rows()
        mask = torch.zeros(_BATCH, _TEXT_LEN, dtype=torch.int32)
        mask[0, :5] = 1  # first prompt is 5 tokens, second is full length
        mask[1, :] = 1
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]), mask=mask)

        module = _module(_identity)
        MiniMaxH3DiffusionNFT.forward(
            module=module, model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        assert module.call_args_list[0].kwargs["encoder_hidden_states"].shape == (1, 5, _TEXT_DIM)
        assert module.call_args_list[1].kwargs["encoder_hidden_states"].shape == (1, _TEXT_LEN, _TEXT_DIM)

    def test_non_tuple_output_raises(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))
        with pytest.raises(TypeError, match="Unexpected MiniMax H3 transformer output"):
            MiniMaxH3DiffusionNFT.forward(
                module=_module(lambda **kw: torch.randn(1, 8)),
                model_config=MagicMock(),
                model_inputs=model_inputs,
                negative_model_inputs=None,
            )


class TestMiniMaxH3ForwardAndSamplePreviousStep:
    def test_reverse_sampling_is_not_implemented(self):
        with pytest.raises(NotImplementedError, match="forward-process objective"):
            MiniMaxH3DiffusionNFT.forward_and_sample_previous_step(
                module=MagicMock(),
                scheduler=MagicMock(),
                model_config=MagicMock(),
                model_inputs={},
                negative_model_inputs=None,
                scheduler_inputs=None,
                step=0,
            )


class TestMiniMaxH3BuildScheduler:
    def test_build_scheduler_sets_video_timesteps(self):
        cfg = object.__new__(DiffusionModelConfig)
        object.__setattr__(cfg, "architecture", "MiniMaxH3Pipeline")
        object.__setattr__(cfg, "algorithm", "diffusion_nft")
        object.__setattr__(cfg, "pipeline", DiffusionPipelineConfig(num_inference_steps=4, video_flow_shift=12.0))

        scheduler = MiniMaxH3DiffusionNFT.build_scheduler(cfg)
        assert len(scheduler.timesteps) == 4


# Small stand-ins for the real fused DiT dims, so the translation is checked on tensors
# whose per-head layout can be read by eye: 2 heads * 4 dims, GEGLU half 6, 3 rope pairs.
_HEADS, _HEAD_DIM, _FF_HALF, _ROPE_LEN = 2, 4, 6, 3


class _RecordingLoader:
    """Stands in for the vllm pipeline's exact-name loader at the end of the MRO."""

    def load_weights(self, weights):
        self.received = list(weights)
        return {name for name, _ in self.received}


class _StubSyncPipeline(MiniMaxH3RolloutWeightSyncMixin, _RecordingLoader):
    def __init__(self):
        self.transformer = MagicMock()
        self.transformer.arch.num_attention_heads = _HEADS
        self.transformer.arch.attention_head_dim = _HEAD_DIM
        self.transformer.arch.ffn_hidden_size = _FF_HALF
        self.transformer.arch.rope_inv_freq_len = _ROPE_LEN


class TestMiniMaxH3RolloutWeightSync:
    def test_rope_inv_freq_is_synthesized(self):
        """diffusers never streams this buffer, and the vllm one is registered uninitialized."""
        pipeline = _StubSyncPipeline()
        pipeline.load_weights([])

        emitted = dict(pipeline.received)
        assert "transformer.rope.inv_freq" in emitted
        expected = 10000.0 ** (-(torch.arange(0, 2 * _ROPE_LEN, 2, dtype=torch.float32) / (2 * _ROPE_LEN)))
        torch.testing.assert_close(emitted["transformer.rope.inv_freq"], expected)

    def test_rope_inv_freq_is_loaded_once(self):
        pipeline = _StubSyncPipeline()
        pipeline.load_weights([])
        pipeline.load_weights([])

        assert pipeline.received == []

    def test_qkv_fuses_per_head_across_sync_buckets(self):
        """The base sync arrives in buckets, so one block's q/k/v may span several calls."""
        pipeline = _StubSyncPipeline()
        width = _HEADS * _HEAD_DIM
        parts = {c: torch.randn(width, width) for c in ("q", "k", "v")}

        pipeline.load_weights([(f"transformer.transformer_blocks.0.attn.to_{c}.weight", parts[c]) for c in ("q", "k")])
        assert not any("qkv_proj" in name for name, _ in pipeline.received)
        pipeline.load_weights([("transformer.transformer_blocks.0.attn.to_v.weight", parts["v"])])

        fused = dict(pipeline.received)["transformer.blocks.0.attn.qkv_proj.weight"]
        assert fused.shape == (3 * width, width)
        for head in range(_HEADS):
            for offset, comp in enumerate(("q", "k", "v")):
                start = (head * 3 + offset) * _HEAD_DIM
                expected = parts[comp][head * _HEAD_DIM : (head + 1) * _HEAD_DIM]
                torch.testing.assert_close(fused[start : start + _HEAD_DIM], expected)

    def test_geglu_halves_are_swapped(self):
        pipeline = _StubSyncPipeline()
        proj = torch.randn(2 * _FF_HALF, 4)

        pipeline.load_weights([("transformer.transformer_blocks.0.ff.net.0.proj.weight", proj)])

        swapped = dict(pipeline.received)["transformer.blocks.0.mlp.fc1.weight"]
        torch.testing.assert_close(swapped, torch.cat([proj[_FF_HALF:], proj[:_FF_HALF]]))

    @pytest.mark.parametrize(
        ("diffusers_name", "vllm_name"),
        [
            ("audio_proj_in.weight", "audio_patch_proj.weight"),  # must win over the proj_in rename
            ("proj_in.weight", "video_patch_proj.weight"),
            ("norm_out.linear.weight", "final_layer.adaln_proj.linear.weight"),
            ("transformer_blocks.0.attn.norm_q.weight", "blocks.0.attn.q_norm.weight"),
            ("token_refiner.refiner_blocks.0.ff.net.2.weight", "token_refiner.blocks.0.mlp.fc2.weight"),
        ],
    )
    def test_names_are_renamed(self, diffusers_name, vllm_name):
        pipeline = _StubSyncPipeline()
        pipeline.load_weights([(f"transformer.{diffusers_name}", torch.zeros(1))])

        assert f"transformer.{vllm_name}" in dict(pipeline.received)

    def test_lora_deltas_are_dropped(self):
        """They reach the engine through ``add_lora``, not the base weight stream."""
        pipeline = _StubSyncPipeline()
        pipeline.load_weights([("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", torch.zeros(1))])

        assert [name for name, _ in pipeline.received] == ["transformer.rope.inv_freq"]


class TestMiniMaxH3TokenIdNativePrompt:
    @staticmethod
    def _request(prompt: dict):
        sampling_params = SimpleNamespace(extra_args={MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True})
        return MagicMock(prompts=[prompt], sampling_params=sampling_params)

    def test_request_ids_are_retained_without_tokenizer_decode(self):
        pipeline = _StubSyncPipeline()
        pipeline.tokenizer = MagicMock()
        request = self._request({"prompt_token_ids": [1, 2, 3]})

        pipeline._ensure_prompt_text(request)

        assert request.prompts[0]["prompt"] == "[pretokenized]"
        assert torch.equal(pipeline._h3_prompt_ids, torch.tensor([1, 2, 3]))
        pipeline.tokenizer.decode.assert_not_called()

    def test_empty_prompt_ids_are_rejected(self):
        pipeline = _StubSyncPipeline()

        with pytest.raises(ValueError, match="non-empty prompt_token_ids"):
            pipeline._ensure_prompt_text(self._request({"prompt_token_ids": []}))

    def test_generic_agent_loop_ids_are_rejected(self):
        pipeline = _StubSyncPipeline()
        request = MagicMock(
            prompts=[{"prompt_token_ids": [1, 2, 3]}],
            sampling_params=SimpleNamespace(extra_args={}),
        )

        with pytest.raises(ValueError, match="default_agent_loop=minimax_h3_diffusion_single_turn_agent"):
            pipeline._ensure_prompt_text(request)

    def test_missing_prompt_ids_clear_stale_request_state(self):
        pipeline = _StubSyncPipeline()
        pipeline._h3_prompt_ids = torch.tensor([7])

        pipeline._ensure_prompt_text(MagicMock(prompts=[{"prompt": "plain text"}]))

        assert pipeline._h3_prompt_ids is None

    def test_t2va_encoder_consumes_exact_request_ids(self, monkeypatch):
        module_name = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"
        pipeline_module = ModuleType(module_name)
        pipeline_module._dit_rank_world = lambda: (None, 0, 1)
        pipeline_module._broadcast_tensor = lambda value, **kwargs: value
        monkeypatch.setitem(sys.modules, module_name, pipeline_module)

        pipeline = _StubSyncPipeline()
        pipeline._h3_prompt_ids = torch.tensor([101, 17, 202])
        pipeline.text_encoder_tp_size = 1
        pipeline.device = torch.device("cpu")
        pipeline._distribute_encode_inputs = lambda ids, vision_kwargs: ids
        pipeline._encode_text_hidden = lambda ids, vision_kwargs: ids[:, None].float()

        hidden, tags = pipeline.encode_prompt(
            task="t2va",
            prompt="[pretokenized]",
            image=None,
            prepared_videos=None,
        )

        assert hidden[:, 0].tolist() == [101.0, 17.0, 202.0]
        assert tags.tolist() == [1, 1, 1]
