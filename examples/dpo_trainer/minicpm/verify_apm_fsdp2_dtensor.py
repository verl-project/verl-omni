#!/usr/bin/env python
# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Minimal FSDP2 repro for MiniCPM APM ``inputs_embeds + embed_pos``.

C–H passing on 2 GPUs means leftover ``apm`` on the MiniCPMO root is not
enough to reproduce the training crash. Remaining deltas vs training:
decoder-layer wrap of *Whisper encoder layers* (``min_num_params``), and
PEFT wrapping MiniCPMO before FSDP.

    torchrun --nproc_per_node=2 examples/dpo_trainer/minicpm/verify_apm_fsdp2_dtensor.py
    MINICPM_PATH=/path/to/MiniCPM-o-4_5 torchrun --nproc_per_node=2 \\
        examples/dpo_trainer/minicpm/verify_apm_fsdp2_dtensor.py
"""

from __future__ import annotations

import os
import traceback
import types

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate


class TinyDecoderLayer(nn.Module):
    """Stand-in for Qwen3DecoderLayer / MiniCPMWhisperEncoderLayer."""

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, hidden):
        return self.lin(hidden)


class TinyWhisperEncoder(nn.Module):
    """Same add as MiniCPM-o: ``hidden_states = inputs_embeds + embed_pos``."""

    def __init__(self, n_mels: int = 80, d_model: int = 32, pos_len: int = 64, n_layers: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(pos_len, d_model)
        self.layers = nn.ModuleList([TinyDecoderLayer(d_model) for _ in range(n_layers)])

    def forward(self, input_features, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        hidden = F.gelu(self.conv1(input_features))
        hidden = F.gelu(self.conv2(hidden)).permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[: hidden.shape[1], :]
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"    add operands: {type(hidden).__name__} + {type(embed_pos).__name__} "
                f"(conv1.weight={type(self.conv1.weight).__name__})"
            )
        hidden = hidden + embed_pos
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyMiniCPMO(nn.Module):
    """Parent calls ``self.apm`` the way remote MiniCPMO.get_audio_embedding does."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.apm = TinyWhisperEncoder()
        self.layers = nn.ModuleList([TinyDecoderLayer() for _ in range(n_layers)])
        self.llm = nn.Linear(32, 32)

    def forward(self, data, **kwargs):
        del kwargs
        wavforms = data.get("audio_features", [])
        if len(wavforms) > 0:
            audio = self.apm(wavforms)
        else:
            dummy = torch.zeros(
                (1, 80, 100),
                device=self.apm.embed_positions.weight.device,
                dtype=self.apm.embed_positions.weight.dtype,
            )
            audio = self.apm(dummy)
        hidden = audio.mean(dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.llm(hidden)


def _init_dist():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if "RANK" not in os.environ:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29631")
        dist.init_process_group(backend, rank=0, world_size=1)
        return
    dist.init_process_group(backend)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
    return torch.device("cpu")


def _mesh():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(device_type, (dist.get_world_size(),))


def _mp_policy():
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )


def _shard(module: nn.Module) -> nn.Module:
    fully_shard(module, mesh=_mesh(), mp_policy=_mp_policy())
    return module


def _shard_like_training(
    module: TinyMiniCPMO,
    *,
    shard_apm: bool,
    shard_apm_layers: bool = False,
) -> TinyMiniCPMO:
    """Match verl FSDP2: wrap decoder layers first, then the MiniCPMO root.

    ``apm`` is leftover on the root unless ``shard_apm``.
    ``min_num_params`` may also wrap ``MiniCPMWhisperEncoderLayer`` while
    leaving ``conv1`` / ``embed_positions`` on the unwrapped encoder.
    """
    mesh = _mesh()
    mp_policy = _mp_policy()
    if shard_apm:
        fully_shard(module.apm, mesh=mesh, mp_policy=mp_policy)
    elif shard_apm_layers:
        for encoder_layer in module.apm.layers:
            fully_shard(encoder_layer, mesh=mesh, mp_policy=mp_policy)
    for layer in module.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(module, mesh=mesh, mp_policy=mp_policy)
    return module


def _mels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(1, 80, 40, device=device, dtype=dtype)


class TinyPeftRoot(nn.Module):
    """Stand-in for PEFT wrapping MiniCPMO before FSDP."""

    def __init__(self, base: TinyMiniCPMO):
        super().__init__()
        self.base_model = base

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


def _wrap_parent_like_adapter(module: TinyMiniCPMO) -> TinyMiniCPMO:
    original_forward = module.__class__.forward

    def _forward(self, data=None, **kwargs):
        payload = kwargs if data is None else {"data": data, **kwargs}
        return original_forward(self, payload.get("data", payload), **{k: v for k, v in payload.items() if k != "data"})

    module.forward = types.MethodType(_forward, module)
    return module


def _run(name: str, fn) -> None:
    rank0 = dist.get_rank() == 0
    if rank0:
        print(f"\n== {name} ==")
    try:
        fn()
        dist.barrier()
        if rank0:
            print("   PASS")
    except Exception as exc:
        dist.barrier()
        if rank0:
            print(f"   FAIL: {type(exc).__name__}: {exc}")
            traceback.print_exc()


def main() -> None:
    _init_dist()
    device = _device()
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dist.get_rank() == 0:
        print(f"world_size={dist.get_world_size()} device={device} dtype={dtype}")

    def case_encoder_plain():
        model = _shard(TinyWhisperEncoder().to(device=device, dtype=dtype))
        model(_mels(device, dtype))

    def case_encoder_from_local():
        model = _shard(TinyWhisperEncoder().to(device=device, dtype=dtype))
        weight = model.conv1.weight
        mels = DTensor.from_local(
            _mels(device, dtype).to(device=weight.device, dtype=weight.dtype),
            weight.device_mesh,
            tuple(Replicate() for _ in range(weight.device_mesh.ndim)),
            run_check=False,
        )
        model(mels)

    def case_parent_only_real_audio():
        model = _shard(TinyMiniCPMO().to(device=device, dtype=dtype))
        model({"audio_features": _mels(device, dtype)})

    def case_parent_only_dummy_audio():
        model = _shard(TinyMiniCPMO().to(device=device, dtype=dtype))
        model({"audio_features": []})

    def case_adapter_wrap_then_shard_parent():
        model = _wrap_parent_like_adapter(TinyMiniCPMO().to(device=device, dtype=dtype))
        model = _shard(model)
        model({"audio_features": []})

    def case_shard_apm_then_parent_dummy():
        model = TinyMiniCPMO().to(device=device, dtype=dtype)
        fully_shard(model.apm, mesh=_mesh(), mp_policy=_mp_policy())
        fully_shard(model, mesh=_mesh(), mp_policy=_mp_policy())
        model({"audio_features": []})

    def case_training_wrap_apm_leftover():
        model = _wrap_parent_like_adapter(TinyMiniCPMO().to(device=device, dtype=dtype))
        _shard_like_training(model, shard_apm=False)
        if dist.get_rank() == 0:
            print(f"    leftover embed_positions.weight: {type(model.apm.embed_positions.weight).__name__}")
        model({"audio_features": []})

    def case_training_wrap_apm_own_unit():
        model = _wrap_parent_like_adapter(TinyMiniCPMO().to(device=device, dtype=dtype))
        _shard_like_training(model, shard_apm=True)
        model({"audio_features": []})

    def case_shard_whisper_layers_conv_leftover():
        model = _wrap_parent_like_adapter(TinyMiniCPMO().to(device=device, dtype=dtype))
        _shard_like_training(model, shard_apm=False, shard_apm_layers=True)
        model({"audio_features": []})

    def case_peft_root_then_training_wrap():
        base = _wrap_parent_like_adapter(TinyMiniCPMO().to(device=device, dtype=dtype))
        root = TinyPeftRoot(base).to(device=device, dtype=dtype)
        mesh = _mesh()
        mp_policy = _mp_policy()
        for encoder_layer in base.apm.layers:
            fully_shard(encoder_layer, mesh=mesh, mp_policy=mp_policy)
        for layer in base.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
        fully_shard(root, mesh=mesh, mp_policy=mp_policy)
        root({"audio_features": []})

    def case_real_minicpmo_dummy():
        path = os.environ.get("MINICPM_PATH")
        if not path:
            if dist.get_rank() == 0:
                print("    skip (export MINICPM_PATH to the MiniCPM-o checkpoint)")
            return
        from types import SimpleNamespace

        from transformers import AutoModel

        from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter

        raw = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype)
        raw = raw.to(device=device)
        module = MiniCPMThinkerAdapter.configure_model(
            raw,
            SimpleNamespace(local_path=path, hf_config=raw.config, trust_remote_code=True, override_config={}),
        )
        mesh = _mesh()
        mp_policy = _mp_policy()
        if hasattr(module, "apm"):
            fully_shard(module.apm, mesh=mesh, mp_policy=mp_policy)
        llm = getattr(module, "llm", None)
        decoder = None
        if llm is not None:
            decoder = getattr(getattr(llm, "model", llm), "layers", None)
        if decoder is not None:
            for layer in decoder:
                fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
        fully_shard(module, mesh=mesh, mp_policy=mp_policy)
        dummy = torch.zeros((1, 80, 100), device=device, dtype=next(module.apm.parameters()).dtype)
        module.apm(dummy, output_hidden_states=True)

    _run("A: fully_shard(encoder) + plain Tensor (known PASS)", case_encoder_plain)
    _run("B: fully_shard(encoder) + from_local Replicate (known FAIL)", case_encoder_from_local)
    _run("C: fully_shard(parent) only + real audio via self.apm()", case_parent_only_real_audio)
    _run("D: fully_shard(parent) only + dummy zeros via self.apm()", case_parent_only_dummy_audio)
    _run("E: adapter MethodType on parent THEN fully_shard(parent) + dummy", case_adapter_wrap_then_shard_parent)
    _run("F: fully_shard(apm) then fully_shard(parent) + dummy", case_shard_apm_then_parent_dummy)
    _run("G: training wrap (shard layers then root, apm leftover) + dummy", case_training_wrap_apm_leftover)
    _run("H: training wrap + fully_shard(apm) as its own unit + dummy", case_training_wrap_apm_own_unit)
    _run("I: shard MiniCPMWhisperEncoderLayer, leave conv/embed leftover", case_shard_whisper_layers_conv_leftover)
    _run("J: PEFT-like root + shard whisper layers + dummy", case_peft_root_then_training_wrap)
    _run("K: real MiniCPM-o apm dummy (MINICPM_PATH)", case_real_minicpmo_dummy)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
