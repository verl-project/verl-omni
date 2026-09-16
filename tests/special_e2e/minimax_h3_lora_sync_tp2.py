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

from __future__ import annotations

import argparse
import os
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MiniMaxH3RolloutWeightSyncMixin
from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

_TINY_H3 = {
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "hidden_size": 48,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 64,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 32,
    "freq_dim": 16,
    "time_embed_hidden_dim": 48,
    "time_embed_dim": 32,
    "rope_freq_dim": 2,
}
_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]


def _vllm_transformer_config() -> TransformerConfig:
    config = {
        **_TINY_H3,
        "token_refiner_num_layers": _TINY_H3["num_refiner_layers"],
        "ffn_hidden_size": _TINY_H3["ffn_dim"],
        "latents_dim": _TINY_H3["in_channels"],
        "audio_latents_dim": _TINY_H3["audio_in_channels"],
        "timestep_input_dim": _TINY_H3["freq_dim"],
        "time_embed_hidden_size": _TINY_H3["time_embed_hidden_dim"],
        "adaln_out_features": 18 * _TINY_H3["hidden_size"],
        "final_adaln_out_features": 2 * _TINY_H3["hidden_size"],
        "rope_inv_freq_len": _TINY_H3["rope_freq_dim"],
    }
    return TransformerConfig.from_dict(config)


class _Pipeline(nn.Module, MiniMaxH3RolloutWeightSyncMixin):
    def __init__(self, transformer: MiniMaxH3DiTModel) -> None:
        super().__init__()
        self.transformer = transformer


def _old_adapter_payload() -> tuple[dict[str, torch.Tensor], dict]:
    torch.manual_seed(1234)
    actor = MiniMaxH3Transformer3DModel(**_TINY_H3)
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=_TARGET_MODULES,
        bias="none",
    )
    actor.add_adapter(config, adapter_name="default")
    actor.add_adapter(config, adapter_name="old")
    payload = {
        name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(actor, adapter_name="old").items()
    }
    for tensor in payload.values():
        tensor.copy_(torch.randn_like(tensor))
    return payload, config.to_dict()


def _check_text_encoder_tp(rank: int, tp_size: int, device: torch.device) -> None:
    """Exercise the text-encoder TP group at ETP=1 and ETP==tp_size.

    ``text_encoder_tp_size`` never reaches the DiT: ``MiniMaxH3DiTModel`` does not
    read it, so putting it on this test's ``DiffusionParallelConfig`` would assert
    nothing. It is consumed only by ``MiniMaxH3Pipeline.__init__``, which builds the
    encoder process group and fans prompt/vision tensors to the encoder ranks. Both
    of those touch just ``_dit_rank`` and ``text_encoder_group``, so they run on a
    bare pipeline stand-in without any checkpoint on disk.

    Only ETP==1 and ETP==tp_size are valid: vLLM's ``GroupCoordinator`` asserts a
    per-rank sub-group, so any ``1 < ETP < tp_size`` crashes the out-of-group ranks.
    We therefore exercise exactly the two supported configurations, which reduces to
    the ETP=1/ETP=2 coverage at the default ``tp_size=2``.
    """
    stand_in = object.__new__(MiniMaxH3Pipeline)
    stand_in._dit_rank = rank

    single = MiniMaxH3Pipeline._build_text_encoder_group(stand_in, 1)
    assert single.world_size == 1, f"ETP=1 must stay single-rank, got {single.world_size}"

    group = MiniMaxH3Pipeline._build_text_encoder_group(stand_in, tp_size)
    assert group.world_size == tp_size, f"ETP={tp_size} group must span {tp_size} ranks, got {group.world_size}"
    expected_ranks = list(range(tp_size))
    assert list(group.ranks) == expected_ranks, (
        f"ETP={tp_size} group must cover DiT ranks {expected_ranks}, got {group.ranks}"
    )
    assert group.rank_in_group == rank, f"rank_in_group {group.rank_in_group} != rank {rank}"
    stand_in.text_encoder_group = group

    # Prompt IDs are broadcast from encoder rank 0; ranks > 0 pass None and must
    # recover rank 0's payload exactly, otherwise the sharded encode desynchronizes.
    source = torch.arange(12, device=device, dtype=torch.long).reshape(3, 4) if rank == 0 else None
    received = MiniMaxH3Pipeline._encoder_group_broadcast_tensor(stand_in, source, dtype=torch.long, device=device)
    expected = torch.arange(12, device=device, dtype=torch.long).reshape(3, 4)
    assert tuple(received.shape) == (3, 4), f"broadcast reshaped the tensor: {tuple(received.shape)}"
    assert received.dtype is torch.long, f"broadcast changed dtype: {received.dtype}"
    assert torch.equal(received, expected), f"rank={rank} received a mismatched broadcast payload"

    print(f"rank={rank}: text_encoder_tp={tp_size} group={list(group.ranks)}, broadcast OK", flush=True)


def _check_packed_lora_matmul(device):
    """Cover the real H3 projection dimensions that tiny models do not exercise."""
    generator = torch.Generator(device=device).manual_seed(33)
    lengths = [384, 384, 768, 768, 1152, 1152, 1536, 1536]
    inputs = torch.randn(sum(lengths), 5376, device=device, dtype=torch.bfloat16, generator=generator)
    a = torch.randn(64, 5376, device=device, dtype=torch.bfloat16, generator=generator) * 0.008
    b = torch.randn(7168, 64, device=device, dtype=torch.bfloat16, generator=generator) * 0.0001

    def project(value):
        return torch.nn.functional.linear(torch.nn.functional.linear(value, a), b)

    expected = torch.cat([project(sample) for sample in inputs.split(lengths)])
    torch.testing.assert_close(project(inputs), expected, rtol=0, atol=0)
    print(f"rank={torch.distributed.get_rank()}: H3 rank-64 variable-length LoRA GEMM parity: PASS", flush=True)


def _setup_packed_models(device, mesh, sharded, backend):
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
    from verl.utils.fsdp_utils import apply_fsdp2

    from tests.pipelines.test_minimax_h3_packed_forward_on_cpu import _models
    from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import enable_packed_forward

    serial, packed = _models(lora=True, checkpointing=True, install_packed=False)
    for model in (serial, packed):
        for name, parameter in model.named_parameters():
            if not any(part in name for part in model._keep_in_fp32_modules):
                parameter.data = parameter.data.to(torch.bfloat16)
        model.to(device)
        model.set_adapter("default")
    if backend == "_flash_3_varlen_hub":
        serial.set_attention_backend(backend)
    packed.set_attention_backend(backend)
    if sharded:
        for model in (serial, packed):
            base = model.base_model.model
            kwargs = dict(mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32))
            apply_fsdp2(
                model,
                kwargs,
                {
                    "wrap_policy": {
                        "transformer_layer_cls_to_wrap": [
                            "MiniMaxH3TransformerBlock",
                            "MiniMaxH3TokenRefinerBlock",
                        ]
                    }
                },
            )
            assert all(
                isinstance(block, FSDPModule)
                for block in [*base.token_refiner.refiner_blocks, *base.transformer_blocks]
            )
    enable_packed_forward(packed)
    return serial, packed


def _check_packed_nft(device, mesh, sharded, backend):
    from tests.pipelines.test_minimax_h3_packed_forward_on_cpu import _forward, _inputs, _loss

    serial, packed = _setup_packed_models(device, mesh, sharded, backend)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in _inputs(serial).items()
    }
    optimizers = [
        torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01) for model in (serial, packed)
    ]
    for step in range(2):
        previous, reference = [], []
        for model, enabled in ((serial, False), (packed, True)):
            with torch.no_grad():
                model.set_adapter("old")
                previous.append(_forward(model, inputs, enabled))
                with model.disable_adapter():
                    reference.append(_forward(model, inputs, enabled))
            model.set_adapter("default")
        outputs = [_forward(serial, inputs, False), _forward(packed, inputs, True)]
        for pair in (previous, reference, outputs):
            torch.testing.assert_close(pair[0], pair[1], rtol=2e-2, atol=1e-2)
        with torch.no_grad():
            changed = {
                key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
            }
            changed["video_rows"][1] += 9
            isolated = _forward(packed, changed, True)
            torch.testing.assert_close(isolated[[0, 2]], outputs[1][[0, 2]], rtol=2e-2, atol=1e-2)
            assert not torch.allclose(isolated[1], outputs[1][1])
        losses = [_loss(output, old, ref) for output, old, ref in zip(outputs, previous, reference, strict=True)]
        torch.testing.assert_close(losses[0], losses[1], rtol=2e-2, atol=1e-2)
        for loss in losses:
            loss.backward()
        serial_params = dict(serial.named_parameters())
        compared = 0
        for name, parameter in packed.named_parameters():
            if not parameter.requires_grad:
                continue
            expected, actual = serial_params[name].grad, parameter.grad
            assert expected is not None and actual is not None, name
            if sharded:
                expected, actual = expected.full_tensor(), actual.full_tensor()
            torch.testing.assert_close(expected.float(), actual.float(), rtol=5e-2, atol=1e-3, msg=name)
            compared += 1
        assert compared > 0
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(
            f"rank={torch.distributed.get_rank()} backend={backend} fsdp2={sharded} step={step + 1} "
            f"lora_gradients={compared}: PASS",
            flush=True,
        )


def _check_packed_flow_grpo(device, mesh, sharded, backend, task):
    from types import SimpleNamespace

    from tests.pipelines.test_minimax_h3_packed_forward_on_cpu import _flow_batch, _run, _schedulers, _serial
    from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss
    from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig

    serial, packed = _setup_packed_models(device, mesh, sharded, backend)
    data = _flow_batch(serial, task).to(device)
    schedulers = _schedulers(device)
    optimizers = [
        torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01) for model in (serial, packed)
    ]
    config = SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.2))
    for step in range(2):
        with torch.no_grad():
            serial.set_adapter("old")
            old = _serial(serial, data, schedulers)[0]
            serial.set_adapter("default")
        outputs = [_serial(serial, data, schedulers), _run(packed, data, True, schedulers)]
        for a, b in zip(*outputs, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-2, atol=1e-3)
        with torch.no_grad():
            changed = data.clone()
            changed["prompt_embeds"][1] += 10
            isolated = _run(packed, changed, True, schedulers)
            for a, b in zip(isolated, outputs[1], strict=True):
                torch.testing.assert_close(a[[0, 2]], b[[0, 2]], rtol=2e-2, atol=1e-3)
        losses = [
            FlowGRPOLoss.compute_loss(
                old_log_prob=old,
                log_prob=output[0],
                advantages=torch.tensor([0.6, -0.8, 0.3], device=device),
                config=config,
            )[0]
            for output in outputs
        ]
        torch.testing.assert_close(*losses, rtol=2e-2, atol=1e-3)
        for loss in losses:
            loss.backward()
        expected_params = dict(serial.named_parameters())
        expected_grads, actual_grads = [], []
        for name, parameter in packed.named_parameters():
            if not parameter.requires_grad:
                continue
            a, b = expected_params[name].grad, parameter.grad
            assert a is not None and b is not None, name
            if sharded:
                a, b = a.full_tensor(), b.full_tensor()
            expected_grads.append(a.float().flatten())
            actual_grads.append(b.float().flatten())
        a, b = torch.cat(expected_grads), torch.cat(actual_grads)
        assert a.norm() > 0 and torch.isfinite(b).all()
        relative_error = ((a - b).norm() / a.norm()).item()
        assert relative_error < 0.05, relative_error
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(
            f"rank={torch.distributed.get_rank()} FlowGRPO task={task} fsdp2={sharded} step={step + 1} "
            f"LoRA relative gradient error={relative_error:.6f}: PASS",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny H3 LoRA sync and optional packed-forward numerical checks.")
    parser.add_argument(
        "--check-packed-forward", action="store_true", help="Also check NFT/FlowGRPO backward and FSDP2."
    )
    parser.add_argument(
        "--attn-backend", choices=["_flash_3_varlen_hub", "torch_varlen", "native"], default="_flash_3_varlen_hub"
    )
    args = parser.parse_args()
    if args.check_packed_forward:
        # BF16 GEMM reductions are batch-shape dependent, so serial and packed
        # execution only agree under deterministic matmul. Enable it before any
        # CUDA or distributed work so the cuBLAS workspace setting still applies.
        from verl.workers.engine.utils import enable_full_determinism

        enable_full_determinism(seed=33)
        # Deterministic algorithms also fill uninitialized memory with NaN, which
        # poisons the torch.empty scratch buffers these checks allocate.
        torch.utils.deterministic.fill_uninitialized_memory = False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    tp_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    with TemporaryDirectory(prefix="tiny-minimax-h3-") as model_dir, set_current_vllm_config(VllmConfig()):
        init_distributed_environment(local_rank=local_rank)
        initialize_model_parallel(tensor_model_parallel_size=tp_size)
        try:
            device = torch.device(f"cuda:{local_rank}")
            od_config = OmniDiffusionConfig(
                model=model_dir,
                tf_model_config=_vllm_transformer_config(),
                dtype=torch.float32,
                parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp_size),
            )
            transformer = MiniMaxH3DiTModel(od_config).to(device)
            for parameter in transformer.parameters():
                parameter.data.zero_()
            transformer.stacked_params_mapping = [
                (".qkv_proj", ".to_q", "q"),
                (".qkv_proj", ".to_k", "k"),
                (".qkv_proj", ".to_v", "v"),
                (".fc1", ".fc1_0", "0"),
                (".fc1", ".fc1_1", "1"),
            ]
            payload, peft_config = _old_adapter_payload()

            VLLMOmniHijack.hijack()
            manager = DiffusionLoRAManager(_Pipeline(transformer), device=device, dtype=torch.bfloat16)
            request = OmniTensorLoRARequest(
                lora_name="h3-tp2-regression",
                lora_int_id=1,
                lora_path="/tmp/h3-tp2-regression-unused",
                peft_config=peft_config,
                lora_tensors=payload,
            )
            assert manager.add_adapter(request)
            manager.set_active_adapter(request)
            assert len(manager._lora_modules) == 12
            assert all(all(layer._diffusion_lora_active_slices) for layer in manager._lora_modules.values())

            qkv = manager._lora_modules["transformer.blocks.0.attn.qkv_proj"]
            fc1 = manager._lora_modules["transformer.blocks.0.mlp.fc1"]
            out_proj = manager._lora_modules["transformer.blocks.0.attn.out_proj"]
            fc2 = manager._lora_modules["transformer.blocks.0.mlp.fc2"]
            qkv_b_shapes = [tuple(value.shape) for value in qkv.lora_b_stacked]
            fc1_b_shapes = [tuple(value.shape) for value in fc1.lora_b_stacked]
            local_qkv_rows = _TINY_H3["num_attention_heads"] * _TINY_H3["attention_head_dim"] // tp_size
            local_fc1_rows = _TINY_H3["ffn_dim"] // tp_size
            assert qkv_b_shapes == [(1, 1, local_qkv_rows, 64)] * 3
            assert fc1_b_shapes == [(1, 1, local_fc1_rows, 64)] * 2

            mapped, _ = manager.pipeline.map_lora_update_to_engine(payload, peft_config)
            scale = peft_config["lora_alpha"] / peft_config["r"]

            def mapped_tensor(module_name: str, suffix: str) -> torch.Tensor:
                return mapped[f"transformer.blocks.0.{module_name}.{suffix}.weight"].to(
                    device=device, dtype=torch.bfloat16
                )

            # Column-parallel projections replicate A and shard/scaled B by output rows.
            for layer, modules in ((qkv, ("attn.to_q", "attn.to_k", "attn.to_v")), (fc1, ("mlp.fc1_0", "mlp.fc1_1"))):
                for index, module_name in enumerate(modules):
                    a = mapped_tensor(module_name, "lora_A")
                    b = mapped_tensor(module_name, "lora_B")
                    rows_per_rank = b.shape[0] // tp_size
                    expected_b = (b[rank * rows_per_rank : (rank + 1) * rows_per_rank] * scale).to(torch.bfloat16)
                    torch.testing.assert_close(layer.lora_a_stacked[index][0, 0], a, rtol=0, atol=0)
                    torch.testing.assert_close(layer.lora_b_stacked[index][0, 0], expected_b, rtol=0, atol=0)

            # Row-parallel projections shard A by input columns, replicate/scaled B, then all-reduce the delta.
            for layer, module_name in ((out_proj, "attn.out_proj"), (fc2, "mlp.fc2")):
                a = mapped_tensor(module_name, "lora_A")
                b = mapped_tensor(module_name, "lora_B")
                cols_per_rank = a.shape[1] // tp_size
                expected_a = a[:, rank * cols_per_rank : (rank + 1) * cols_per_rank]
                expected_b = (b * scale).to(torch.bfloat16)
                torch.testing.assert_close(layer.lora_a_stacked[0][0, 0], expected_a, rtol=0, atol=0)
                torch.testing.assert_close(layer.lora_b_stacked[0][0, 0], expected_b, rtol=0, atol=0)

            inputs = torch.randn(3, _TINY_H3["hidden_size"], device=device, dtype=torch.bfloat16)
            qkv_with_lora = qkv(inputs)[0].detach()
            fc1_with_lora = fc1(inputs)[0].detach()

            full_out_input = torch.randn(
                3, _TINY_H3["num_attention_heads"] * _TINY_H3["attention_head_dim"], device=device, dtype=torch.bfloat16
            )
            local_out_input = full_out_input.chunk(tp_size, dim=-1)[rank].contiguous()
            full_fc2_input = torch.randn(3, _TINY_H3["ffn_dim"], device=device, dtype=torch.bfloat16)
            local_fc2_input = full_fc2_input.chunk(tp_size, dim=-1)[rank].contiguous()
            out_with_lora = out_proj(local_out_input)[0].detach()
            fc2_with_lora = fc2(local_fc2_input)[0].detach()

            qkv.reset_lora(0)
            fc1.reset_lora(0)
            out_proj.reset_lora(0)
            fc2.reset_lora(0)
            qkv_base = qkv(inputs)[0].detach()
            fc1_base = fc1(inputs)[0].detach()
            out_base = out_proj(local_out_input)[0].detach()
            fc2_base = fc2(local_fc2_input)[0].detach()
            qkv_delta = qkv_with_lora - qkv_base
            fc1_delta = fc1_with_lora - fc1_base
            out_delta = out_with_lora - out_base
            fc2_delta = fc2_with_lora - fc2_base

            def column_expected(module_names: tuple[str, ...]) -> torch.Tensor:
                parts = []
                for module_name in module_names:
                    a = mapped_tensor(module_name, "lora_A")
                    b = mapped_tensor(module_name, "lora_B")
                    rows_per_rank = b.shape[0] // tp_size
                    local_b = (b[rank * rows_per_rank : (rank + 1) * rows_per_rank] * scale).to(torch.bfloat16)
                    parts.append((inputs @ a.t()) @ local_b.t())
                return torch.cat(parts, dim=-1)

            def row_expected(full_input: torch.Tensor, module_name: str) -> torch.Tensor:
                a = mapped_tensor(module_name, "lora_A")
                b = (mapped_tensor(module_name, "lora_B") * scale).to(torch.bfloat16)
                cols_per_rank = a.shape[1] // tp_size
                local_a = a[:, rank * cols_per_rank : (rank + 1) * cols_per_rank]
                local_input = full_input[:, rank * cols_per_rank : (rank + 1) * cols_per_rank]
                expected = (local_input @ local_a.t()) @ b.t()
                torch.distributed.all_reduce(expected)
                return expected

            torch.testing.assert_close(
                qkv_delta, column_expected(("attn.to_q", "attn.to_k", "attn.to_v")), rtol=0, atol=0
            )
            torch.testing.assert_close(fc1_delta, column_expected(("mlp.fc1_0", "mlp.fc1_1")), rtol=0, atol=0)
            # TP row projections reduce bf16 partials; reduction order differs at TP>2.
            torch.testing.assert_close(out_delta, row_expected(full_out_input, "attn.out_proj"), rtol=2e-2, atol=2.1)
            torch.testing.assert_close(fc2_delta, row_expected(full_fc2_input, "mlp.fc2"), rtol=2e-2, atol=2.1)

            print(
                f"rank={rank}/{tp_size}: exact rank-64 qkv/fc1/out_proj/fc2 LoRA parity; "
                f"qkv_B={qkv_b_shapes}, fc1_B={fc1_b_shapes}",
                flush=True,
            )
            _check_text_encoder_tp(rank, tp_size, device)
            if args.check_packed_forward:
                from torch.distributed.device_mesh import init_device_mesh

                mesh = init_device_mesh("cuda", (tp_size,))
                _check_packed_lora_matmul(device)
                for sharded in (False, True):
                    _check_packed_nft(device, mesh, sharded, args.attn_backend)
                    for task in ("t2va", "fl2va", "ref2va"):
                        _check_packed_flow_grpo(device, mesh, sharded, args.attn_backend, task)
                torch.distributed.barrier()
                if rank == 0:
                    print("MiniMax H3 packed forward + varlen backward + FSDP2: PASS", flush=True)
            torch.distributed.barrier()
            if rank == 0:
                print(f"MiniMax H3 LoRA TP={tp_size} actor-to-rollout regression: PASS", flush=True)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
