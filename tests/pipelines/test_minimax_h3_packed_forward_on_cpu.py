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
"""Numerical contracts for a real tiny H3 transformer, without checkpoints or GPUs."""

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig, get_peft_model
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import pack_video_audio_rows, serialize_ref_blocks
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import (
    MiniMaxH3DiffusionNFT,
    PackedSequenceLayout,
    enable_packed_forward,
    pack_model_inputs,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.common import (
    configure_flow_scheduler,
    flatten_joint_latents,
    h3_sigma_schedules,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionNFTLoss, FlowGRPOLoss
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig

_MODEL_KWARGS = dict(
    hidden_size=32,
    num_attention_heads=2,
    attention_head_dim=16,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=64,
    text_dim=16,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=2,
)


def _models(lora=False, checkpointing=False, install_packed=True):
    torch.manual_seed(13)
    serial = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    packed = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    serial.set_attention_backend("native")
    packed.load_state_dict(serial.state_dict(), strict=True)
    if lora:
        config = LoraConfig(
            r=2, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
        )
        serial, packed = get_peft_model(serial, config), get_peft_model(packed, config)
        serial.add_adapter("old", config)
        packed.add_adapter("old", config)
        with torch.no_grad():
            for name, parameter in serial.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.02)
        packed.load_state_dict(serial.state_dict(), strict=True)
    if checkpointing:
        serial.enable_gradient_checkpointing()
        packed.enable_gradient_checkpointing()
    if install_packed:
        enable_packed_forward(packed)
    return serial, packed


def _inputs(model, lengths=(3, 5, 4), task="t2va"):
    torch.manual_seed(19)
    batch = len(lengths)
    meta = torch.tensor([4, 6, 1, 4, 4, 3]).repeat(batch, 1)
    latents = pack_video_audio_rows(torch.randn(batch, 4, 96), torch.randn(batch, 6, 32))
    micro_batch = TensorDict({"latent_meta": meta}, batch_size=[batch])
    if task == "fl2va":
        micro_batch["condition_video_rows"] = torch.randn(batch, 4, 96)
        micro_batch["condition_video_row_count"] = torch.full((batch, 1), 4)
        micro_batch["keyframe_frame_indices"] = torch.zeros(batch, 1, dtype=torch.long)
    elif task == "ref2va":
        blocks = [
            [{"kind": "image", "latent_h": 4, "latent_w": 4}] * (index + 1)
            + ([{"kind": "audio", "ref_audio_t": 2}] if index == 1 else [])
            for index in range(batch)
        ]
        metadata = [serialize_ref_blocks(refs) for refs in blocks]
        micro_batch["ref_block_meta"] = torch.stack([meta for meta, _ in metadata])
        micro_batch["ref_block_count"] = torch.tensor([[count] for _, count in metadata])
        micro_batch["condition_video_rows"] = torch.randn(batch, batch * 4, 96)
        micro_batch["condition_video_row_count"] = torch.arange(1, batch + 1)[:, None] * 4
        micro_batch["condition_audio_rows"] = torch.randn(batch, 4, 32)
        micro_batch["condition_audio_row_count"] = torch.tensor([[4 if i == 1 else 0] for i in range(batch)])
    inputs, _ = MiniMaxH3DiffusionNFT.prepare_model_inputs(
        model,
        SimpleNamespace(),
        latents,
        torch.tensor([200.0, 700.0, 500.0, 300.0])[:batch],
        torch.randn(batch, max(lengths), 16),
        torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None],
        None,
        None,
        micro_batch,
        0,
    )
    return inputs


def _forward(model, inputs, packed):
    if packed:
        return MiniMaxH3DiffusionNFT.forward(model, SimpleNamespace(), inputs)
    # The former per-sample implementation is retained only as a numerical baseline.
    outputs = []
    for sample, video_count, audio_count in MiniMaxH3DiffusionNFT._iter_sample_inputs(model, inputs):
        video, audio = model(**sample)
        outputs.append(pack_video_audio_rows(-video[:, video_count:], -audio[:, audio_count:]))
    return torch.cat(outputs)


def _loss(prediction, old_prediction, ref_prediction):
    torch.manual_seed(23)
    return DiffusionNFTLoss.compute_loss(
        forward_prediction=prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_prediction,
        x0=torch.randn_like(prediction),
        xt=torch.randn_like(prediction),
        t_expanded=torch.tensor([0.2, 0.7, 0.5], device=prediction.device)[: len(prediction)].reshape(
            -1, *([1] * (prediction.ndim - 1))
        ),
        reward_prob=torch.tensor([0.1, 0.9, 0.6])[: len(prediction)],
        config=SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="diffusion_nft", ref_kl_coef=0.1)),
    )[0]


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_packed_nft_matches_serial_outputs_loss_and_lora_gradients(task, checkpointing):
    serial, packed = _models(lora=True, checkpointing=checkpointing)
    inputs = _inputs(serial, task=task)
    old, ref = [], []
    for model, enabled in ((serial, False), (packed, True)):
        with torch.no_grad():
            model.set_adapter("old")
            old.append(_forward(model, inputs, enabled))
            with model.disable_adapter():
                ref.append(_forward(model, inputs, enabled))
        model.set_adapter("default")
    predictions = [_forward(serial, inputs, False), _forward(packed, inputs, True)]
    for pair in (old, ref, predictions):
        torch.testing.assert_close(pair[0], pair[1], atol=2e-6, rtol=2e-5)
    losses = [
        _loss(prediction, previous, reference)
        for prediction, previous, reference in zip(predictions, old, ref, strict=True)
    ]
    torch.testing.assert_close(losses[0], losses[1], atol=2e-6, rtol=2e-5)
    for loss in losses:
        loss.backward()
    grads = [{name: p.grad for name, p in model.named_parameters() if p.requires_grad} for model in (serial, packed)]
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert grads[0][name] is not None, name
        torch.testing.assert_close(grads[0][name], grads[1][name], atol=3e-6, rtol=3e-4, msg=name)


def test_one_forward_per_micro_batch_and_no_cross_sample_attention():
    serial, packed = _models()
    inputs = _inputs(serial)
    calls = []
    hook = packed.register_forward_pre_hook(lambda *_: calls.append(1))
    expected = _forward(packed, inputs, True)
    assert len(calls) == 1
    changed = copy.deepcopy(inputs)
    for key in ("video_rows", "audio_rows", "encoder_hidden_states"):
        changed[key][1] += 9
    changed["timestep"][1] = 0.95
    actual = _forward(packed, changed, True)
    torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]])
    assert not torch.allclose(actual[1], expected[1])
    hook.remove()


def test_fa3_receives_separate_dit_and_text_boundaries(monkeypatch):
    from verl_omni.pipelines.minimax_h3_diffusion_nft import diffusers_training_adapter as packed_forward

    calls = []

    def fa3(query, key, value, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
        assert cu_seqlens_q is cu_seqlens_k
        assert max_seqlen_q == max_seqlen_k and not causal
        boundaries = cu_seqlens_q.tolist()
        calls.append(boundaries)
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)]
        assert max_seqlen_q == max(lengths)
        layout = PackedSequenceLayout.from_lengths(lengths, query.device)
        return layout.attention(query[None], key[None], value[None], "native").squeeze(0)

    monkeypatch.setattr(packed_forward, "_get_fa3_varlen", lambda: fa3)
    serial, packed = _models(checkpointing=True)
    packed.set_attention_backend("_flash_3_varlen_hub")
    inputs = _inputs(serial)
    actual = _forward(packed, inputs, True)
    expected = _forward(serial, inputs, False)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert set(map(tuple, calls)) == {(0, 3, 8, 12), (0, 13, 28, 42)}
    for (name, parameter), (other_name, reference) in zip(
        packed.named_parameters(), serial.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_unavailable_fa3_fails_instead_of_falling_back(monkeypatch):
    from verl_omni.pipelines.minimax_h3_diffusion_nft import diffusers_training_adapter as packed_forward

    def unavailable():
        raise RuntimeError("FA3 kernel unavailable")

    monkeypatch.setattr(packed_forward, "_get_fa3_varlen", unavailable)
    _, packed = _models()
    with pytest.raises(RuntimeError, match="FA3 kernel unavailable"):
        packed.set_attention_backend("_flash_3_varlen_hub")
    assert packed.transformer_blocks[0].attn.processor._attention_backend == "native"


@pytest.mark.parametrize("fail", [False, True])
def test_packed_numerical_matmul_settings_are_scoped(monkeypatch, fail):
    from tests.special_e2e.minimax_h3_lora_sync_tp2 import _packed_matmul_context

    backend_calls = []

    def preferred_backend(backend=None):
        if backend is not None:
            backend_calls.append(backend)
        return "original"

    matmul = SimpleNamespace(
        allow_bf16_reduced_precision_reduction=True,
        allow_bf16_reduced_precision_reduction_split_k=True,
    )
    monkeypatch.setattr(
        torch.backends, "cuda", SimpleNamespace(preferred_blas_library=preferred_backend, matmul=matmul)
    )
    with pytest.raises(RuntimeError, match="test failure") if fail else nullcontext():
        with _packed_matmul_context():
            assert matmul.allow_bf16_reduced_precision_reduction == (False, False)
            assert backend_calls == ["cublaslt"]
            if fail:
                raise RuntimeError("test failure")
    assert matmul.allow_bf16_reduced_precision_reduction == (True, True)
    assert backend_calls == ["cublaslt", "original"]


def test_varlen_layout_does_not_materialize_native_padding():
    layout = PackedSequenceLayout.from_lengths([2, 4], torch.device("cpu"))
    assert layout.total_tokens == 6
    assert "valid_mask" not in vars(layout) and "padded_indices" not in vars(layout)
    query = torch.randn(1, 6, 2, 16)
    assert layout.attention(query, query, query, "native").shape == query.shape
    assert "valid_mask" in vars(layout) and "padded_indices" in vars(layout)


def test_packing_preserves_row_timesteps_and_resets_positions():
    serial, _ = _models()
    inputs = _inputs(serial, task="fl2va")
    samples = [sample for sample, _, _ in MiniMaxH3DiffusionNFT._iter_sample_inputs(serial, inputs)]
    packed = pack_model_inputs(samples)
    expected = torch.cat([sample["timestep"][sample["timestep_indices"]] for sample in samples])
    torch.testing.assert_close(packed["timestep"][packed["timestep_indices"]], expected)
    torch.testing.assert_close(packed["position_ids"], torch.cat([sample["position_ids"] for sample in samples]))
    assert packed["sequence_layout"].cu_seqlens.tolist() == [0, 17, 36, 54]
    assert packed["text_sequence_layout"].cu_seqlens.tolist() == [0, 3, 8, 12]


def test_checkpoint_recompute_keeps_each_forward_boundaries():
    serial, packed = _models(checkpointing=True)
    batches = [_inputs(serial, lengths=(3, 5)), _inputs(serial, lengths=(4, 2, 3))]
    # Two forwards remain live before backward; no mutable processor-side layout is allowed.
    serial_loss = sum(_forward(serial, inputs, False).square().mean() for inputs in batches)
    packed_loss = sum(_forward(packed, inputs, True).square().mean() for inputs in batches)
    serial_loss.backward()
    packed_loss.backward()
    for (name, p), (other_name, q) in zip(serial.named_parameters(), packed.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_checkpoint_roundtrip_preserves_class_config_and_weight_names(tmp_path):
    serial, packed = _models()
    serial.save_pretrained(tmp_path)
    loaded = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path)
    enable_packed_forward(loaded)
    assert loaded.config.hidden_size == 32
    assert loaded.state_dict().keys() == serial.state_dict().keys()
    inputs = _inputs(serial)
    torch.testing.assert_close(_forward(loaded, inputs, True), _forward(packed, inputs, True))
    loaded.save_pretrained(tmp_path / "packed")
    restored = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path / "packed")
    restored.set_attention_backend("native")
    assert restored.state_dict().keys() == serial.state_dict().keys()
    torch.testing.assert_close(_forward(restored, inputs, False), _forward(serial, inputs, False))


def test_fsdp_loader_keeps_attention_checkpointing_and_fp32_islands(tmp_path, monkeypatch):
    from verl_omni.workers.config import DiffusionModelConfig
    from verl_omni.workers.engine.fsdp import diffusers_impl

    serial, _ = _models()
    serial.save_pretrained(tmp_path)
    config = DiffusionModelConfig(
        path=str(tmp_path),
        load_tokenizer=False,
        architecture="MiniMaxH3Pipeline",
        algorithm="diffusion_nft",
        external_lib=None,
        config_path=str(tmp_path),
        local_path=str(tmp_path),
        trust_remote_code=False,
        attn_backend="native",
        enable_gradient_checkpointing=True,
    )
    engine = SimpleNamespace(
        model_config=config,
        engine_config=SimpleNamespace(model_dtype="bf16"),
        device_mesh=None,
        _build_module_from_registry=lambda _: None,
    )
    monkeypatch.setattr(diffusers_impl, "get_init_weight_context_manager", lambda **_: nullcontext)
    model = diffusers_impl.DiffusersFSDPEngine._build_module(engine)
    assert type(model) is MiniMaxH3Transformer3DModel
    assert not getattr(model, "supports_packed_batch", False)
    enable_packed_forward(model)
    assert model.gradient_checkpointing and model.token_refiner.gradient_checkpointing
    assert model.proj_in.weight.dtype == torch.float32
    assert model.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    assert model.transformer_blocks[0].attn.processor._attention_backend == "native"


def test_packed_model_preserves_recipe_fsdp_wrap_targets():
    from verl.utils.fsdp_utils import _select_fsdp2_wrap_targets

    for model in _models():
        targets = _select_fsdp2_wrap_targets(model, ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"])
        assert set(targets) == set([*model.transformer_blocks, *model.token_refiner.refiner_blocks])


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_h3_packing_is_default_with_the_generic_model_config(tmp_path, algorithm):
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config import DiffusionModelConfig

    (tmp_path / "model_index.json").write_text('{"_class_name": "MiniMaxH3Pipeline"}')
    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config/diffusion/model"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="diffusion_model",
            overrides=[
                f"path={tmp_path}",
                "+load_tokenizer=false",
                f"algorithm={algorithm}",
                "attn_backend=native",
            ],
        )
    model_config = omega_conf_to_dataclass(config)
    assert type(model_config) is DiffusionModelConfig
    adapter = DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", algorithm)
    model = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    enable_packed_forward(model)
    assert getattr(model, "supports_packed_batch", False)
    assert adapter is not None
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

    rollout_model_config = DiffusionStrategy(None).init_model_config(config)
    assert type(rollout_model_config) is DiffusionModelConfig
    assert "use_packed_batch" not in config


def test_generic_diffusion_config_does_not_expose_packing():
    from dataclasses import fields

    from hydra import compose, initialize_config_dir

    from verl_omni.workers.config import DiffusionModelConfig

    assert "use_packed_batch" not in {field.name for field in fields(DiffusionModelConfig)}
    with pytest.raises(TypeError, match="use_packed_batch"):
        DiffusionModelConfig(use_packed_batch=True)
    with pytest.raises(ValueError, match="Invalid attn_backend"):
        DiffusionModelConfig(attn_backend="torch_varlen")
    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="diffusion_trainer")
    assert "use_packed_batch" not in config.actor_rollout_ref.model


def test_packed_boundaries_must_match_rows():
    serial, packed = _models()
    samples = [sample for sample, _, _ in MiniMaxH3DiffusionNFT._iter_sample_inputs(serial, _inputs(serial))]
    inputs = pack_model_inputs(samples)
    inputs["text_sequence_layout"] = PackedSequenceLayout.from_lengths([1], torch.device("cpu"))
    with pytest.raises(ValueError, match="text attention boundaries"):
        packed(**inputs)
    inputs = pack_model_inputs(samples)
    inputs["sequence_layout"] = PackedSequenceLayout.from_lengths([1], torch.device("cpu"))
    with pytest.raises(ValueError, match="attention boundaries"):
        packed(**inputs)


def test_packed_input_preparation_rejects_mixed_target_geometry():
    serial, _ = _models()
    batch = TensorDict({"latent_meta": torch.tensor([[4, 6, 1, 4, 4, 3], [4, 6, 1, 2, 8, 3]])}, batch_size=[2])
    with pytest.raises(ValueError, match="shared target latent layout"):
        MiniMaxH3DiffusionNFT.prepare_model_inputs(
            serial,
            SimpleNamespace(),
            None,
            None,
            None,
            None,
            None,
            None,
            batch,
            0,
        )


def test_automodel_forward_override_and_unsupported_backend_fail_closed():
    serial, packed = _models()
    assert type(packed) is MiniMaxH3Transformer3DModel
    assert packed is enable_packed_forward(packed)
    with pytest.raises(ValueError, match="attn_backend"):
        packed.set_attention_backend("unsupported_backend")

    sequence_parallel = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    sequence_parallel.transformer_blocks[0].attn.processor._parallel_config = object()
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        enable_packed_forward(sequence_parallel)
    with pytest.raises(TypeError, match="MiniMaxH3Transformer3DModel"):
        enable_packed_forward(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("lengths", [[], [0], [3, -1]])
def test_empty_or_invalid_sequences_are_rejected(lengths):
    with pytest.raises(ValueError, match="positive lengths"):
        PackedSequenceLayout.from_lengths(lengths, torch.device("cpu"))


# FlowGRPO packed replay contracts.


def _config(sde_type="sde"):
    return SimpleNamespace(
        algo=SimpleNamespace(noise_level=0.6, sde_type=sde_type),
        pipeline=SimpleNamespace(av_logprob_video_weight=0.25, av_logprob_audio_weight=0.75),
    )


def _schedulers(device="cpu"):
    schedulers = (FlowMatchSDEDiscreteScheduler(), FlowMatchSDEDiscreteScheduler())
    for scheduler, sigmas in zip(schedulers, h3_sigma_schedules(4), strict=True):
        configure_flow_scheduler(scheduler, sigmas, device)
    return schedulers


def _flow_batch(model, task="t2va", lengths=(3, 5, 4), shared_steps=False):
    nft = _inputs(model, task=task, lengths=lengths)
    samples = list(MiniMaxH3DiffusionNFT._iter_sample_inputs(model, nft))
    batch = len(samples)
    steps = torch.zeros(batch, dtype=torch.long) if shared_steps else torch.arange(batch)
    video_sigmas, audio_sigmas = (torch.tensor(sigmas) for sigmas in h3_sigma_schedules(4))
    currents = []
    for sample, cv, ca in samples:
        video, audio = sample["hidden_states"], sample["audio_hidden_states"]
        if task == "ref2va":
            video, audio = video[:, cv:], audio[:, ca:]
        currents.append(flatten_joint_latents(video, audio))
    current = torch.cat(currents)
    result = {
        "all_latents": current.unsqueeze(1),
        "all_next_latents": (current + torch.randn_like(current) * 0.05).unsqueeze(1),
        "all_timesteps": video_sigmas[steps, None],
        "h3_audio_timesteps": audio_sigmas[steps, None],
        "h3_step_indices": steps[:, None],
        "prompt_embeds": nft["encoder_hidden_states"],
        "prompt_embeds_mask": nft["encoder_mask"],
    }
    if task == "ref2va":
        for key in (
            "ref_block_meta",
            "ref_block_count",
            "condition_video_rows",
            "condition_audio_rows",
            "condition_video_row_count",
            "condition_audio_row_count",
        ):
            result[key] = nft[key]
        result["latent_meta"] = torch.tensor(nft["latent_meta"]).repeat(batch, 1)
        result["prompt_token_tags"] = pad_sequence(
            [sample["token_tags"][sample["text_indices"]] for sample, _, _ in samples], batch_first=True
        )
    else:
        for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
            result[f"h3_{name}"] = pad_sequence([sample[name] for sample, _, _ in samples], batch_first=True)
        result["h3_video_rows"] = torch.tensor([sample["hidden_states"].shape[1] for sample, _, _ in samples])
        result["h3_audio_rows"] = torch.tensor([sample["audio_hidden_states"].shape[1] for sample, _, _ in samples])
        result["h3_seq_len"] = torch.tensor([sample["position_ids"].shape[0] for sample, _, _ in samples])
        result["h3_video_update_mask"] = torch.stack(
            [torch.arange(sample["hidden_states"].shape[1]) >= cv for sample, cv, _ in samples]
        )
    return TensorDict(result, batch_size=[batch])


def _prepare(model, data, packed, step=0):
    if not packed:
        return MiniMaxH3FlowGRPO._prepare_batch_inputs(
            data["all_latents"], data["all_timesteps"], data["prompt_embeds"], data["prompt_embeds_mask"], data, step
        )[0]
    return MiniMaxH3FlowGRPO.prepare_model_inputs(
        model,
        _config(),
        data["all_latents"],
        data["all_timesteps"],
        data["prompt_embeds"],
        data["prompt_embeds_mask"],
        None,
        None,
        data,
        step,
    )[0]


def _run(model, data, packed, schedulers, sde_type="sde", step=0):
    if not packed:
        inputs = _prepare(model, data, False, step)
        device = data["all_latents"].device
        kwargs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
            if not key.startswith("_h3_")
        }
        video, audio = model(**kwargs)
        return MiniMaxH3FlowGRPO._sample_previous_step(schedulers, _config(sde_type), inputs, data, video, audio, step)
    return MiniMaxH3FlowGRPO.forward_and_sample_previous_step(
        model,
        schedulers,
        _config(sde_type),
        _prepare(model, data, packed, step),
        None,
        data,
        step,
    )


def _serial(model, data, schedulers, sde_type="sde"):
    outputs = [_run(model, data[i : i + 1], False, schedulers, sde_type) for i in range(data.shape[0])]
    return tuple(torch.cat(parts) for parts in zip(*outputs, strict=True))


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("sde_type", ["sde", "cps"])
def test_variable_batches_match_serial_transitions_loss_and_lora_gradients(task, sde_type):
    serial, packed = _models(lora=True, checkpointing=True)
    data = _flow_batch(serial, task)
    schedulers = _schedulers()
    serial.set_adapter("old")
    with torch.no_grad():
        old_log_probs = _serial(serial, data, schedulers, sde_type)[0]
    serial.set_adapter("default")
    packed.set_adapter("default")
    expected = _serial(serial, data, schedulers, sde_type)
    calls = []
    handle = packed.register_forward_pre_hook(lambda *_: calls.append(1))
    actual = _run(packed, data, True, schedulers, sde_type)
    handle.remove()
    assert len(calls) == 1
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)
    losses = [
        FlowGRPOLoss.compute_loss(
            old_log_prob=old_log_probs,
            log_prob=output[0],
            advantages=torch.tensor([0.6, -0.8, 0.3]),
            config=SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.2)),
        )[0]
        for output in (expected, actual)
    ]
    torch.testing.assert_close(*losses, rtol=3e-4, atol=3e-5)
    for loss in losses:
        loss.backward()
    gradients = dict(serial.named_parameters())
    norm = 0.0
    for name, parameter in packed.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and gradients[name].grad is not None
            torch.testing.assert_close(parameter.grad, gradients[name].grad, rtol=1e-3, atol=3e-5)
            norm += parameter.grad.norm().item()
    assert norm > 0


@pytest.mark.parametrize("steps", [[1, 0, 1], [1, 1, 0]])
def test_reference_layouts_batch_scheduler_replay_by_step_and_restore_order(monkeypatch, steps):
    from unittest.mock import Mock

    serial, packed = _models()
    data = _flow_batch(serial, "ref2va")
    video_sigmas, audio_sigmas = (torch.tensor(sigmas) for sigmas in h3_sigma_schedules(4))
    data["h3_step_indices"] = torch.tensor(steps)[:, None]
    data["all_timesteps"] = video_sigmas[steps, None]
    data["h3_audio_timesteps"] = audio_sigmas[steps, None]
    expected = _serial(serial, data, _schedulers())
    replay = Mock(wraps=MiniMaxH3FlowGRPO._sample_previous_step)
    monkeypatch.setattr(MiniMaxH3FlowGRPO, "_sample_previous_step", replay)
    actual = _run(packed, data, True, _schedulers())
    assert replay.call_count == 2
    assert [call.args[2]["hidden_states"].shape[0] for call in replay.call_args_list] == [2, 1]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)


def test_packed_matches_existing_dense_batch_when_layouts_are_shared():
    dense, packed = _models()
    data = _flow_batch(dense, lengths=(4, 4, 4), shared_steps=True)
    expected = _run(dense, data, False, _schedulers())
    actual = _run(packed, data, True, _schedulers())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)


def test_packed_ref2va_accepts_variable_reference_layouts_rejected_by_dense_batch():
    dense, packed = _models()
    data = _flow_batch(dense, task="ref2va", lengths=(4, 4, 4, 4), shared_steps=True)
    grouped_fields = (
        "prompt_embeds",
        "prompt_embeds_mask",
        "ref_block_meta",
        "ref_block_count",
        "condition_video_rows",
        "condition_audio_rows",
        "condition_video_row_count",
        "condition_audio_row_count",
        "prompt_token_tags",
    )
    for key in grouped_fields:
        first_prompt, second_prompt = data[key][0].clone(), data[key][1].clone()
        data[key][0:2] = first_prompt
        data[key][2:4] = second_prompt

    # Model two rollout groups with n=2: layouts match within each prompt and differ across prompts.
    assert data["condition_video_row_count"].flatten().tolist() == [4, 4, 8, 8]
    assert data["condition_audio_row_count"].flatten().tolist() == [0, 0, 4, 4]
    torch.testing.assert_close(data["prompt_embeds"][0], data["prompt_embeds"][1])
    torch.testing.assert_close(data["prompt_embeds"][2], data["prompt_embeds"][3])

    with pytest.raises(ValueError, match="shared condition video row count"):
        _prepare(dense, data, False)

    prepared = _prepare(packed, data, True)
    sequence_lengths = [sample["position_ids"].shape[0] for sample in prepared["_h3_samples"]]
    assert sequence_lengths[0] == sequence_lengths[1]
    assert sequence_lengths[2] == sequence_lengths[3]
    assert sequence_lengths[0] != sequence_lengths[2]

    expected = _serial(dense, data, _schedulers())
    calls = []
    handle = packed.register_forward_pre_hook(lambda *_: calls.append(1))
    actual = _run(packed, data, True, _schedulers())
    handle.remove()
    assert len(calls) == 1
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)


def test_different_text_lengths_still_fail_closed_in_dense_mode():
    dense, _ = _models()
    with pytest.raises(ValueError, match="shared text length"):
        _prepare(dense, _flow_batch(dense), False)


def test_flowgrpo_installs_packed_forward_on_automodel():
    dense = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    data = _flow_batch(dense)
    assert not getattr(dense, "supports_packed_batch", False)
    _run(dense, data, True, _schedulers())
    assert getattr(dense, "supports_packed_batch", False)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_engine_prepares_packed_replay_without_slicing_unrelated_nested_fields(task):
    from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine

    _, packed = _models()
    data = _flow_batch(packed, task)
    raw = data.clone()
    lengths = data["prompt_embeds_mask"].sum(dim=1).tolist()
    for key in ("prompt_embeds", "prompt_embeds_mask"):
        raw[key] = torch.nested.nested_tensor(
            [value[:length] for value, length in zip(data[key], lengths, strict=True)], layout=torch.jagged
        )
    raw["input_ids"] = torch.nested.nested_tensor(
        [torch.ones(n, dtype=torch.long) for n in lengths], layout=torch.jagged
    )
    for key in ("condition_video_rows", "condition_audio_rows"):
        if key not in raw:
            continue
        counts = data[key.replace("_rows", "_row_count")].flatten().tolist()
        raw[key] = torch.nested.nested_tensor(
            [value[:count] for value, count in zip(data[key], counts, strict=True)], layout=torch.jagged
        )
        raw[f"{key}_mask"] = torch.nested.nested_tensor(
            [torch.ones(count, dtype=torch.bool) for count in counts], layout=torch.jagged
        )
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = packed
    engine.model_config = _config()
    engine.model_config.architecture = "MiniMaxH3Pipeline"
    engine.model_config.algorithm = "flow_grpo"
    engine.model_config.external_lib = None
    engine.use_ulysses_sp = False
    inputs, _ = engine.prepare_model_inputs(raw, step=0)
    expected = _prepare(packed, data, True)
    for a, b in zip(inputs["_h3_samples"], expected["_h3_samples"], strict=True):
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key])
            else:
                assert a[key] == b[key]
    assert raw["input_ids"].is_nested
    assert raw["prompt_embeds"].is_nested


def test_packed_replay_selects_the_requested_trajectory_column():
    serial, packed = _models()
    data = _flow_batch(serial)
    for key in ("all_latents", "all_next_latents", "all_timesteps", "h3_audio_timesteps", "h3_step_indices"):
        data[key] = torch.cat([data[key], data[key]], dim=1)
    data["all_next_latents"][:, 1] += 0.5
    actual = _run(packed, data, True, _schedulers(), step=1)
    expected = [_run(serial, data[i : i + 1], False, _schedulers(), step=1) for i in range(3)]
    for a, parts in zip(actual, zip(*expected, strict=True), strict=True):
        torch.testing.assert_close(a, torch.cat(parts), rtol=3e-4, atol=3e-5)
    assert not torch.allclose(actual[0], _run(packed, data, True, _schedulers(), step=0)[0])


def test_packed_rejects_different_target_row_counts():
    _, packed = _models()
    data = _flow_batch(packed)
    data["h3_video_update_mask"][1, 0] = False
    with pytest.raises(ValueError, match="shared target video/audio row counts"):
        _prepare(packed, data, True)


def test_packed_rejects_truncated_reference_rows():
    _, packed = _models()
    data = _flow_batch(packed, "ref2va")
    data["condition_video_rows"] = data["condition_video_rows"][:, :2]
    with pytest.raises(ValueError, match="counts exceed the supplied condition tensors"):
        _prepare(packed, data, True)


def test_packed_samples_cannot_attend_to_each_other():
    _, packed = _models()
    data = _flow_batch(packed)
    expected = _run(packed, data, True, _schedulers())
    modified = data.clone()
    modified["prompt_embeds"][1] += 10
    modified["all_latents"][1] += 3
    actual = _run(packed, modified, True, _schedulers())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a[[0, 2]], b[[0, 2]])
    assert not torch.allclose(actual[1][1], expected[1][1])
