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
"""CPU tests for the MiniMax H3 image DiffusionNFT training adapter: PDD schedule, heads and layout."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from safetensors.torch import save_file
from tensordict import TensorDict
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MINIMAX_H3_IMGVID_COND_TIMESTEP, MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import (
    _image_packed_seq_len,
    minimax_h3_image_pdd_grid,
    minimax_h3_image_run_sigmas,
)

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import build_row_timesteps, serialize_ref_blocks
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import (
    IMAGE_CONDITION_TIMESTEP,
    MiniMaxH3PDDHeadLinear,
    build_image_layout,
    ensure_pdd_heads,
    image_latent_meta,
    install_pdd_heads,
    parse_image_latent_meta,
    pdd_dit_timestep,
    pdd_step_indices,
    pdd_train_timesteps,
)
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.diffusers_training_adapter import MiniMaxH3ImageDiffusionNFT
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.rollout_kernels import (
    _straight_through,
    rollout_kernels_enabled,
)
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionNFTLoss

_RELEASE_PDD = {
    "num_inference_steps": 6,
    "sigmas": [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0],
}
_HIDDEN, _TEXT_DIM, _TEXT_LEN = 32, 24, 5
_LATENT_H, _LATENT_W = 4, 6
_NUM_ROWS = (_LATENT_H // 2) * (_LATENT_W // 2)


def _run_sigmas() -> torch.Tensor:
    sigmas = minimax_h3_image_run_sigmas(minimax_h3_image_pdd_grid(_RELEASE_PDD))
    return torch.tensor(sigmas[:-1], dtype=torch.float32)


def _tiny_actor() -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=_HIDDEN,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=48,
        text_dim=_TEXT_DIM,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    ).eval()


def _heads(steps: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(steps, 96, _HIDDEN, generator=generator), torch.randn(steps, 96, generator=generator)


def _model_config(tmp_path) -> SimpleNamespace:
    weight, bias = _heads()
    save_file({"video_out.weight": weight, "video_out.bias": bias}, str(tmp_path / "pdd_fused_heads.safetensors"))
    return SimpleNamespace(config_path=str(tmp_path), local_path=None, path=None)


def _micro_batch(batch: int, **extra) -> TensorDict:
    sigmas = _run_sigmas()
    data = {
        "latents_clean": torch.randn(batch, _NUM_ROWS * 96),
        "train_timesteps": pdd_train_timesteps(sigmas).long().unsqueeze(0).expand(batch, -1).clone(),
        "pdd_sigmas": sigmas.unsqueeze(0).expand(batch, -1).clone(),
        "latent_meta": image_latent_meta(_NUM_ROWS, _LATENT_H, _LATENT_W).expand(batch, -1).clone(),
    }
    data.update(extra)
    return TensorDict(data, batch_size=batch)


_SCHEDULER = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))


class TestRegistry:
    def test_training_adapter_is_registered_through_the_pipelines_package(self):
        import verl_omni.pipelines as pipelines

        assert hasattr(pipelines, "minimax_h3_image_diffusion_nft")
        resolved = DiffusionModelBase.get_class_by_name("MiniMaxH3ImagePipeline", "diffusion_nft")
        assert resolved is MiniMaxH3ImageDiffusionNFT

    def test_actor_sequence_parallelism_is_rejected(self):
        MiniMaxH3ImageDiffusionNFT.validate_engine_config(SimpleNamespace(ulysses_sequence_parallel_size=1))
        with pytest.raises(NotImplementedError, match="Ulysses"):
            MiniMaxH3ImageDiffusionNFT.validate_engine_config(SimpleNamespace(ulysses_sequence_parallel_size=2))

    @pytest.mark.parametrize(
        ("targets", "accepted"),
        [(["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"], True), ("all-linear", False)],
        ids=["block-targets", "all-linear"],
    )
    def test_lora_targets_share_the_h3_whitelist(self, targets, accepted):
        config = SimpleNamespace(lora_rank=64, target_modules=targets)
        if accepted:
            MiniMaxH3ImageDiffusionNFT.validate_lora_config(config)
        else:
            with pytest.raises(ValueError):
                MiniMaxH3ImageDiffusionNFT.validate_lora_config(config)


class TestPDDSchedule:
    def test_truncated_and_exact_timesteps_resolve_to_step_order(self):
        sigmas = _run_sigmas()
        timesteps = pdd_train_timesteps(sigmas)
        assert timesteps.long().tolist() == [1000, 967, 887, 750, 532, 232]
        expected = torch.arange(6)
        assert torch.equal(pdd_step_indices(timesteps, sigmas), expected)
        assert torch.equal(pdd_step_indices(timesteps.long(), sigmas), expected)
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        batched = sigmas.unsqueeze(0).expand(6, -1)
        assert torch.equal(pdd_step_indices(timesteps.long()[permutation], batched), permutation)

    @pytest.mark.parametrize(
        ("timestep", "sigmas", "message"),
        [
            (500, None, "only on the rollout's PDD sigmas"),
            (965, None, "only on the rollout's PDD sigmas"),
            (750, [1.0, 0.751, 0.75, 0.2], "decrease by more than"),
        ],
        ids=["off-grid", "two-units-off", "dense-grid"],
    )
    def test_lookup_fails_closed(self, timestep, sigmas, message):
        grid = _run_sigmas() if sigmas is None else torch.tensor(sigmas)
        with pytest.raises(ValueError, match=message):
            pdd_step_indices(torch.tensor([timestep]), grid)

    def test_model_time_reproduces_the_rollout_loop_rounding(self):
        run_sigmas = minimax_h3_image_run_sigmas(minimax_h3_image_pdd_grid(_RELEASE_PDD))[:-1]
        rollout = torch.empty(len(run_sigmas), dtype=torch.float32)
        for step, sigma in enumerate(run_sigmas):
            rollout[step : step + 1].fill_(float(1.0 - sigma))
        assert torch.equal(pdd_dit_timestep(torch.tensor(run_sigmas, dtype=torch.float32)), rollout)

    def test_training_inputs_noise_at_the_exact_rollout_sigma(self):
        batch = 3
        noise = torch.randn(batch, _NUM_ROWS * 96)
        micro_batch = _micro_batch(batch, forward_noise=noise)
        sigmas = _run_sigmas()
        for step in range(6):
            xt, timesteps, context = MiniMaxH3ImageDiffusionNFT.prepare_training_inputs(micro_batch, step, _SCHEDULER)
            sigma = sigmas[step]
            assert torch.equal(timesteps, (sigma * 1000.0).expand(batch))
            assert torch.equal(xt, (1.0 - sigma) * micro_batch["latents_clean"] + sigma * noise)
            assert torch.equal(context["t_expanded"], sigma.expand(batch, 1))
        xt, _, _ = MiniMaxH3ImageDiffusionNFT.prepare_training_inputs(micro_batch, 0, _SCHEDULER)
        assert torch.equal(xt, noise)

    def test_training_inputs_require_the_rollout_grid(self):
        micro_batch = _micro_batch(1)
        del micro_batch["pdd_sigmas"]
        with pytest.raises(ValueError, match="pdd_sigmas"):
            MiniMaxH3ImageDiffusionNFT.prepare_training_inputs(micro_batch, 0, _SCHEDULER)

    def test_nft_timestep_selection_keeps_every_step_resolvable(self):
        sigmas = _run_sigmas()
        train_timesteps = pdd_train_timesteps(sigmas).unsqueeze(0).expand(4, -1)
        selected = DiffusionNFTLoss._select_train_timesteps(train_timesteps, timestep_fraction=0.5, seed=3)
        assert selected.dtype == torch.long and selected.shape == (4, 3)
        for column in range(selected.shape[1]):
            steps = pdd_step_indices(selected[:, column], sigmas.unsqueeze(0).expand(4, -1))
            assert torch.equal(pdd_train_timesteps(sigmas)[steps].long(), selected[:, column])


class TestPDDHeads:
    def test_head_selection_matches_the_fused_head_of_that_step(self):
        actor = _tiny_actor()
        weight, bias = _heads()
        state_keys = set(actor.state_dict())
        head = install_pdd_heads(actor, weight, bias)
        assert isinstance(head, MiniMaxH3PDDHeadLinear) and install_pdd_heads(actor, weight, bias) is head
        assert set(actor.state_dict()) == state_keys
        hidden = torch.randn(2, 7, _HIDDEN)
        for step in range(6):
            head.pdd_step = step
            assert torch.equal(actor.proj_out(hidden), torch.nn.functional.linear(hidden, weight[step], bias[step]))

    @pytest.mark.parametrize("step", [None, -1, 6], ids=["unset", "negative", "past-last"])
    def test_head_requires_a_valid_step(self, step):
        actor = _tiny_actor()
        head = install_pdd_heads(actor, *_heads())
        head.pdd_step = step
        with pytest.raises(ValueError, match="fused heads"):
            actor.proj_out(torch.randn(1, 2, _HIDDEN))

    def test_heads_must_match_the_output_projection(self):
        actor = _tiny_actor()
        with pytest.raises(ValueError, match="do not match proj_out"):
            install_pdd_heads(actor, torch.randn(6, 96, _HIDDEN + 1), torch.randn(6, 96))

    def test_heads_load_from_the_transformer_directory(self, tmp_path):
        actor = _tiny_actor()
        head = ensure_pdd_heads(actor, _model_config(tmp_path))
        assert torch.equal(head.pdd_weight, _heads()[0])
        with pytest.raises(FileNotFoundError):
            ensure_pdd_heads(
                _tiny_actor(), SimpleNamespace(config_path=str(tmp_path / "missing" / "transformer"), local_path=None)
            )


class TestLayoutMatchesRollout:
    """The Actor layout must be the rollout denoiser's packed layout without its padding document."""

    @pytest.mark.parametrize("grids", [[], [(4, 4), (2, 6)]], ids=["t2i", "edit-two-references"])
    def test_rows_positions_tags_and_timesteps(self, grids):
        blocks = [{"kind": "image", "latent_h": h, "latent_w": w} for h, w in grids]
        text_tags = torch.tensor([1, 0, 0, 1, 1])
        used = _TEXT_LEN + _NUM_ROWS + sum(h * w // 4 for h, w in grids)
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=_TEXT_LEN,
            latent_t=1,
            latent_h=_LATENT_H,
            latent_w=_LATENT_W,
            audio_t=0,
            ref_blocks=blocks,
            seq_len=_image_packed_seq_len(used),
        )
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"].view(-1)] = text_tags
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(_TEXT_LEN, _TEXT_DIM),
            token_tags=tags,
            device=torch.device("cpu"),
        )
        sigma = _run_sigmas()[3]
        model_time = float(pdd_dit_timestep(sigma))
        rollout_timesteps = torch.empty(branch.seq_len, dtype=torch.float32)
        branch.fill_timesteps(
            rollout_timesteps,
            t_video=model_time,
            t_audio=model_time,
            imgvid_cond_timestep=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            audio_ref_cond_timestep=1.0,
        )

        meta, count = serialize_ref_blocks(blocks) if blocks else (None, 0)
        position_ids, token_tags, image_indices, text_indices, num_condition = build_image_layout(
            _LATENT_H, _LATENT_W, _TEXT_LEN, ref_block_meta=meta, ref_block_count=count, text_token_tags=text_tags
        )
        unique, indices = build_row_timesteps(
            image_indices,
            torch.empty(0, dtype=torch.long),
            num_condition,
            0,
            _TEXT_LEN,
            video_timestep=model_time,
            audio_timestep=model_time,
            condition_video_timestep=IMAGE_CONDITION_TIMESTEP,
            condition_audio_timestep=1.0,
        )

        assert IMAGE_CONDITION_TIMESTEP == MINIMAX_H3_IMGVID_COND_TIMESTEP
        assert position_ids.shape == (used, 3)
        assert torch.equal(position_ids, packed["img_position_ids"].reshape(-1, 3)[:used])
        assert torch.equal(token_tags, tags[:used])
        assert torch.equal(image_indices, branch.img_pos) and torch.equal(text_indices, packed["text_pos"])
        assert num_condition == int((~branch.update_mask).sum()) == sum(h * w // 4 for h, w in grids)
        assert torch.equal(unique[indices], rollout_timesteps[:used])
        assert packed["cu_seqlens"].tolist() == [0, used, _image_packed_seq_len(used)]

    def test_latent_meta_must_describe_one_audio_free_canvas(self):
        assert parse_image_latent_meta(image_latent_meta(_NUM_ROWS, _LATENT_H, _LATENT_W)) == (6, 4, 6)
        with pytest.raises(ValueError, match="share one generation canvas"):
            parse_image_latent_meta(torch.tensor([[6, 0, 1, 4, 6, 0], [4, 0, 1, 4, 4, 0]]))
        with pytest.raises(ValueError, match="audio-free"):
            parse_image_latent_meta(torch.tensor([[6, 8, 1, 4, 6, 2]]))


class TestForward:
    def _inputs(self, micro_batch, step, actor, model_config, text_len=_TEXT_LEN, tags=None):
        batch = micro_batch.batch_size[0]
        xt, timesteps, _ = MiniMaxH3ImageDiffusionNFT.prepare_training_inputs(micro_batch, step, _SCHEDULER)
        if tags is not None:
            micro_batch["prompt_token_tags"] = tags
        torch.manual_seed(7)
        inputs, negative = MiniMaxH3ImageDiffusionNFT.prepare_model_inputs(
            module=actor,
            model_config=model_config,
            latents=xt,
            timesteps=timesteps,
            prompt_embeds=torch.randn(batch, text_len, _TEXT_DIM),
            prompt_embeds_mask=torch.ones(batch, text_len, dtype=torch.long),
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch=micro_batch,
            step=step,
        )
        assert negative is None
        return inputs

    def test_each_sample_uses_its_own_step_head_and_exact_time(self, tmp_path, monkeypatch):
        monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "use_rollout_kernels", False)
        model_config = _model_config(tmp_path)
        seen = []

        class _Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_out = torch.nn.Linear(_HIDDEN, 96)

            def forward(self, **kwargs):
                seen.append({**kwargs, "pdd_step": self.proj_out.pdd_step})
                return kwargs["hidden_states"] * 2.0, kwargs["audio_hidden_states"]

        module = _Recorder()
        micro_batch = _micro_batch(2)
        micro_batch["train_timesteps"][:, 0] = torch.tensor([532, 967])
        inputs = self._inputs(micro_batch, 0, module, model_config)
        prediction = MiniMaxH3ImageDiffusionNFT.forward(module, model_config, inputs)

        sigmas = _run_sigmas()
        assert [call["pdd_step"] for call in seen] == [4, 1]
        assert module.proj_out.pdd_step is None
        for call, step in zip(seen, (4, 1), strict=True):
            assert torch.equal(call["timestep"], pdd_dit_timestep(sigmas[step : step + 1]))
            assert call["audio_hidden_states"].shape == (1, 0, 32) and call["audio_indices"].numel() == 0
            assert call["hidden_states"].shape == (1, _NUM_ROWS, 96)
            assert call["timestep_indices"].unique().tolist() == [0]
        assert torch.equal(prediction, -2.0 * inputs["image_rows"].reshape(2, -1))

    def test_edit_prepends_reference_rows_at_the_condition_time(self, tmp_path, monkeypatch):
        monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "use_rollout_kernels", False)
        model_config = _model_config(tmp_path)
        grids = [(4, 4)]
        meta, count = serialize_ref_blocks([{"kind": "image", "latent_h": 4, "latent_w": 4}])
        condition = torch.randn(1, 6, 96)
        micro_batch = _micro_batch(
            1,
            condition_video_rows=condition,
            condition_video_row_count=torch.tensor([[4]]),
            ref_block_meta=meta.unsqueeze(0),
            ref_block_count=torch.tensor([[count]]),
        )
        seen = {}

        class _Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_out = torch.nn.Linear(_HIDDEN, 96)

            def forward(self, **kwargs):
                seen.update(kwargs)
                return kwargs["hidden_states"], kwargs["audio_hidden_states"]

        module = _Recorder()
        inputs = self._inputs(micro_batch, 2, module, model_config)
        prediction = MiniMaxH3ImageDiffusionNFT.forward(module, model_config, inputs)

        assert torch.equal(seen["hidden_states"][0, :4], condition[0, :4])
        assert seen["hidden_states"].shape == (1, 4 + _NUM_ROWS, 96)
        step_time = float(pdd_dit_timestep(_run_sigmas()[2]))
        assert seen["timestep"].tolist() == pytest.approx(sorted([step_time, 0.999]))
        reference_rows = seen["video_indices"][: sum(h * w // 4 for h, w in grids)]
        assert seen["timestep"][seen["timestep_indices"][reference_rows]].tolist() == pytest.approx([0.999] * 4)
        assert prediction.shape == (1, _NUM_ROWS * 96)

        micro_batch["condition_video_row_count"] = torch.tensor([[3]])
        inputs = self._inputs(micro_batch, 2, module, model_config)
        with pytest.raises(ValueError, match="do not match layout rows"):
            MiniMaxH3ImageDiffusionNFT.forward(module, model_config, inputs)

    def test_audio_free_diffusers_forward_runs_and_depends_on_the_step_head(self, tmp_path):
        model_config = _model_config(tmp_path)
        actor = _tiny_actor()
        micro_batch = _micro_batch(2, forward_noise=torch.randn(2, _NUM_ROWS * 96))
        outputs = []
        for step in (0, 5):
            inputs = self._inputs(micro_batch, step, actor, model_config)
            with torch.no_grad():
                outputs.append(MiniMaxH3ImageDiffusionNFT.forward(actor, model_config, inputs))
        assert outputs[0].shape == (2, _NUM_ROWS * 96) and all(torch.isfinite(value).all() for value in outputs)

        inputs = self._inputs(micro_batch, 5, actor, model_config)
        with torch.no_grad():
            right = MiniMaxH3ImageDiffusionNFT.forward(actor, model_config, inputs)
            wrong = MiniMaxH3ImageDiffusionNFT.forward(
                actor, model_config, {**inputs, "pdd_step": torch.tensor([4, 4])}
            )
        assert torch.equal(right, outputs[1]) and not torch.allclose(right, wrong)

        actor.audio_proj_in.weight.data.fill_(float("nan"))
        actor.audio_proj_out.weight.data.zero_()
        actor.train()
        prediction = MiniMaxH3ImageDiffusionNFT.forward(actor, model_config, inputs)
        assert torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        assert actor.transformer_blocks[0].attn.to_q.weight.grad.abs().sum() > 0

    def test_rollout_kernel_install_keeps_the_cpu_forward_and_gradients(self, tmp_path, monkeypatch):
        from peft import LoraConfig, inject_adapter_in_model

        model_config = _model_config(tmp_path)
        actor = _tiny_actor()
        inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "ff.net.2"]), actor)
        for name, parameter in actor.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.05)
        micro_batch = _micro_batch(2, forward_noise=torch.randn(2, _NUM_ROWS * 96))
        inputs = self._inputs(micro_batch, 3, actor, model_config)

        def run():
            actor.zero_grad()
            prediction = MiniMaxH3ImageDiffusionNFT.forward(actor, model_config, inputs)
            prediction.square().mean().backward()
            return prediction.detach(), actor.transformer_blocks[0].attn.to_q.lora_B["default"].weight.grad.clone()

        monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "use_rollout_kernels", False)
        stock = run()
        assert not rollout_kernels_enabled(actor)
        monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "use_rollout_kernels", True)
        installed = run()
        assert rollout_kernels_enabled(actor)
        assert torch.equal(installed[0], stock[0]) and torch.equal(installed[1], stock[1])
        assert stock[1].abs().sum() > 0

    def _lora_actor(self):
        from peft import LoraConfig, inject_adapter_in_model

        actor = _tiny_actor()
        inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "ff.net.2"]), actor)
        for name, parameter in actor.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.05)
        return actor

    def test_packed_micro_batch_matches_the_per_sample_forward(self, tmp_path, monkeypatch):
        model_config = _model_config(tmp_path)
        actor = self._lora_actor()
        micro_batch = _micro_batch(3, forward_noise=torch.randn(3, _NUM_ROWS * 96))
        micro_batch["train_timesteps"][:, 0] = torch.tensor([532, 967, 750])
        inputs = self._inputs(micro_batch, 0, actor, model_config)
        inputs["encoder_mask"][1, 3:] = 0
        inputs["encoder_mask"][2, 4:] = 0

        def run(packed):
            monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "pack_micro_batch", packed)
            actor.zero_grad()
            prediction = MiniMaxH3ImageDiffusionNFT.forward(actor, model_config, inputs)
            prediction.square().mean().backward()
            grads = {name: parameter.grad.clone() for name, parameter in actor.named_parameters() if "lora_" in name}
            return prediction.detach(), grads

        per_sample, per_sample_grads = run(False)
        packed, packed_grads = run(True)
        assert torch.allclose(packed, per_sample, atol=1e-5, rtol=1e-4)
        for name, grad in per_sample_grads.items():
            assert grad.abs().sum() > 0, name
            assert torch.allclose(packed_grads[name], grad, atol=1e-5, rtol=1e-4), name

        monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "pack_micro_batch", True)
        other_text = inputs["encoder_hidden_states"].clone()
        other_text[2] += 1.0
        with torch.no_grad():
            changed = MiniMaxH3ImageDiffusionNFT.forward(
                actor, model_config, {**inputs, "encoder_hidden_states": other_text}
            )
        assert torch.allclose(changed[:2], packed[:2], atol=1e-6) and not torch.allclose(changed[2], packed[2])

    def test_packed_micro_batch_rejects_gradient_checkpointing(self, tmp_path, monkeypatch):
        model_config = _model_config(tmp_path)
        actor = self._lora_actor()
        actor.enable_gradient_checkpointing()
        actor.train()
        monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "pack_micro_batch", True)
        inputs = self._inputs(_micro_batch(2, forward_noise=torch.randn(2, _NUM_ROWS * 96)), 2, actor, model_config)
        with pytest.raises(NotImplementedError, match="gradient checkpointing off"):
            MiniMaxH3ImageDiffusionNFT.forward(actor, model_config, inputs)
        with torch.no_grad():
            assert MiniMaxH3ImageDiffusionNFT.forward(actor, model_config, inputs).shape == (2, _NUM_ROWS * 96)

    def test_straight_through_returns_the_kernel_value_with_the_reference_gradient(self):
        source = torch.tensor([1.0, 2.0], requires_grad=True)
        value = torch.tensor([10.0, 20.0])
        result = _straight_through(value, lambda: source * 3)
        assert torch.equal(result, value)
        result.sum().backward()
        assert torch.equal(source.grad, torch.tensor([3.0, 3.0]))
        with torch.no_grad():
            assert _straight_through(value, lambda: source * 3) is value

    def test_reverse_sampling_is_rejected(self):
        with pytest.raises(NotImplementedError, match="forward-process"):
            MiniMaxH3ImageDiffusionNFT.forward_and_sample_previous_step(None, None, None, {}, None, None, 0)
