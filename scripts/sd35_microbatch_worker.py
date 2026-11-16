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
"""Bounded real SD3.5 Medium actor update worker for tune_microbatch.py."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

SEED = 1234
REQUEST = {"trial_id", "microbatch", "accumulation", "steps", "effective_batch", "workload", "workload_digest"}
RESULT_ECHO = ("trial_id", "microbatch", "accumulation", "steps", "workload", "workload_digest")
BATCH = {"all_latents", "all_timesteps", "prompt_embeds", "pooled_prompt_embeds", "old_log_probs", "advantages"}
SOURCE_FILES = (
    "verl_omni/workers/engine/fsdp/diffusers_impl.py",
    "verl_omni/workers/utils/losses.py",
    "verl_omni/pipelines/sd3_flow_grpo/diffusers_training_adapter.py",
)
MODEL_FILES = (
    "model_index.json",
    "transformer/config.json",
    "scheduler/scheduler_config.json",
    "transformer/diffusion_pytorch_model.safetensors",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_hash(files: dict[str, Path]) -> str:
    return hashlib.sha256(canonical({name: file_hash(path) for name, path in sorted(files.items())})).hexdigest()


def read_json(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_batch(payload: Any, torch: Any) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != BATCH:
        raise ValueError("actor batch requires exactly six tensor keys")
    shapes = {
        "all_latents": (2, 3, 16, 32, 32),
        "all_timesteps": (2, 2),
        "pooled_prompt_embeds": (2, 2048),
        "old_log_probs": (2, 2),
        "advantages": (2, 2),
    }
    for name, tensor in payload.items():
        if type(tensor) is not torch.Tensor or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be a plain CPU fp32 tensor")
        shape = tuple(tensor.shape)
        if name == "prompt_embeds":
            if len(shape) != 3 or shape[0] != 2 or shape[1] < 1 or shape[2] != 4096:
                raise ValueError("prompt_embeds shape mismatch")
        elif shape != shapes[name]:
            raise ValueError(f"{name} shape mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains nonfinite values")
    if not bool(torch.count_nonzero(payload["advantages"])):
        raise ValueError("all-zero advantages")
    return payload


def configure(microbatch: int, model_path: Path):
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig

    from verl_omni.workers.config import DiffusionModelConfig, FSDPDiffusionActorConfig
    from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig
    from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig, DiffusionRolloutAlgoConfig

    model = DiffusionModelConfig(
        path=str(model_path),
        algorithm="flow_grpo",
        load_tokenizer=False,
        attn_backend="native",
        enable_gradient_checkpointing=True,
        lora_rank=2,
        lora_alpha=2,
        lora_dtype="fp32",
        target_modules=["to_q"],
        pipeline=DiffusionPipelineConfig(height=256, width=256, num_inference_steps=4, guidance_scale=0.0),
        algo=DiffusionRolloutAlgoConfig(noise_level=0.7, sde_type="sde"),
    )
    engine = FSDPEngineConfig(
        strategy="fsdp2",
        model_dtype="fp32",
        dtype="fp32",
        fsdp_size=1,
        ulysses_sequence_parallel_size=1,
        forward_only=False,
        param_offload=False,
        optimizer_offload=False,
        use_torch_compile=False,
        mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
        seed=SEED,
        full_determinism=True,
    )
    optim = FSDPOptimizerConfig(
        optimizer="AdamW",
        optimizer_impl="torch.optim",
        lr=1e-5,
        weight_decay=0.0,
        clip_grad=1.0,
        total_training_steps=2,
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
    )
    actor = FSDPDiffusionActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=microbatch,
        fsdp_config=engine,
        optim=optim,
        checkpoint=CheckpointConfig(),
        model_config=model,
        diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.2, adv_clip_max=5.0),
    )
    return model, actor


def imports():
    from verl.workers.engine.base import BaseEngine

    from verl_omni.pipelines.model_base import DiffusionModelBase
    from verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter import FlowMatchSDEDiscreteScheduler
    from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
    from verl_omni.workers.utils.losses import diffusion_loss

    return BaseEngine, DiffusionModelBase, PPODiffusersFSDPEngine, diffusion_loss, FlowMatchSDEDiscreteScheduler


def identity(model_path: Path, batch_path: Path, base_cls: Any) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    source = {name: root / name for name in SOURCE_FILES}
    source["scripts/sd35_microbatch_worker.py"] = Path(__file__).resolve()
    source["upstream/verl.workers.engine.base.py"] = Path(inspect.getfile(base_cls)).resolve()
    from verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter import StableDiffusion3FlowGRPO
    from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
    from verl_omni.workers.utils.losses import diffusion_loss

    for name, symbol in zip(
        SOURCE_FILES, (PPODiffusersFSDPEngine, diffusion_loss, StableDiffusion3FlowGRPO), strict=True
    ):
        if Path(inspect.getfile(symbol)).resolve() != source[name].resolve():
            raise RuntimeError(f"import origin differs from hashed source: {name}")
    # The supported cache has a single transformer weight; no split or alternate layout is inferred.
    transformer = model_path / "transformer"
    weights = sorted(
        p.relative_to(model_path).as_posix()
        for p in transformer.rglob("*")
        if p.is_file() and p.suffix in (".safetensors", ".bin")
    )
    if weights != [MODEL_FILES[-1]]:
        raise ValueError("expected a single-file SD3.5 Medium transformer cache")
    return {
        "data_sha256": file_hash(batch_path),
        "model_sha256": group_hash({n: model_path / n for n in MODEL_FILES}),
        "source_sha256": group_hash(source),
        "precision": "fp32",
    }


def validate_request(request: Any, workload: dict[str, str]) -> dict[str, Any]:
    if type(request) is not dict or set(request) != REQUEST:
        raise ValueError("wrong request keys")
    m = request["microbatch"]
    if (
        type(request["trial_id"]) is not int
        or request["trial_id"] < 0
        or type(m) is not int
        or m not in (1, 2)
        or type(request["effective_batch"]) is not int
        or request["effective_batch"] != 2
        or type(request["accumulation"]) is not int
        or request["accumulation"] != 2 // m
        or type(request["steps"]) is not int
        or request["steps"] != 2
    ):
        raise ValueError("unsupported fixed search request")
    if (
        type(request["workload"]) is not dict
        or set(request["workload"]) != set(workload)
        or any(type(v) is not str for v in request["workload"].values())
        or type(request["workload_digest"]) is not str
        or request["workload"] != workload
        or request["workload_digest"] != hashlib.sha256(canonical(workload)).hexdigest()
    ):
        raise ValueError("workload identity mismatch")
    return request


def admit(model_path: Path, payload: dict[str, Any], microbatch: int, classes: tuple):
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    base, registry, engine_cls, loss_fn, scheduler_cls = classes
    model, actor = configure(microbatch, model_path)
    adapter = registry.get_class(model)
    if adapter.__name__ != "StableDiffusion3FlowGRPO" or not issubclass(engine_cls, base):
        raise RuntimeError("SD3.5 engine or adapter unavailable")
    scheduler = scheduler_cls.from_pretrained(str(model_path), subfolder="scheduler")
    adapter.set_timesteps(scheduler, model, "cpu")
    if tuple(scheduler.timesteps[:2].shape) != (2,):
        raise ValueError("scheduler has fewer than two timesteps")
    expected_timesteps = scheduler.timesteps[:2].detach().cpu()
    if expected_timesteps.dtype != payload["all_timesteps"].dtype or not torch.equal(
        payload["all_timesteps"], expected_timesteps.expand(2, -1)
    ):
        raise ValueError("actor batch all_timesteps differs from configured scheduler first two timesteps")
    data = TensorDict(payload, batch_size=2)
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=microbatch, height=256, width=256, vae_scale_factor=8)
    if tu.get_non_tensor_data(data, "micro_batch_size_per_gpu", default=None) != microbatch:
        raise RuntimeError("TensorDict metadata mismatch")
    inspect.signature(base.train_batch).bind(None, data=data, loss_function=None)
    if not callable(loss_fn) or not callable(engine_cls.forward_backward_batch):
        raise RuntimeError("actor update path unavailable")
    config = dataclasses.asdict(actor)
    config.pop("ppo_micro_batch_size_per_gpu")
    return model, actor, data, hashlib.sha256(canonical(config)).hexdigest()


def plain(tensor, torch):
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu().contiguous().clone()


def vector(params, torch):
    return [float(v) for name in sorted(params) for v in plain(params[name], torch).reshape(-1).tolist()]


def initial_digest(params, provenance, torch):
    h = hashlib.sha256(canonical(provenance))
    for name in sorted(params):
        tensor = plain(params[name], torch)
        h.update(canonical([name, list(tensor.shape), str(tensor.dtype)]))
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng_snapshot(torch, np):
    return (
        torch.get_rng_state().clone(),
        [x.clone() for x in torch.cuda.get_rng_state_all()],
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
    )


def restore(engine, params, initial, optim_state, scheduler_state, rng, torch, np):
    with torch.no_grad():
        for name in params:
            params[name].copy_(initial[name])
    engine.optimizer.load_state_dict(copy.deepcopy(optim_state))
    engine.lr_scheduler.load_state_dict(copy.deepcopy(scheduler_state))
    engine.optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state_all(rng[1])
    random.setstate(rng[2])
    np.random.set_state(rng[3])


def witness(out, params, grad_refs, previous, accumulation, torch):
    losses = [float(v) for row in out["loss"] for v in row]
    if len(losses) != accumulation * 2:
        raise RuntimeError("unexpected loss count")
    if any(grad_refs[name] is None for name in params):
        raise RuntimeError("missing LoRA gradient")
    grads = [float(v) for name in sorted(params) for v in plain(grad_refs[name], torch).reshape(-1).tolist()]
    updated = vector(params, torch)
    result = {"loss": [sum(losses)], "grads": grads, "params": updated}
    if any(not math.isfinite(v) for values in result.values() for v in values):
        raise RuntimeError("nonfinite update witness")
    if not any(v != 0 for v in grads) or not any(a != b for a, b in zip(previous, updated, strict=True)):
        raise RuntimeError("no gradient or parameter movement")
    return result


def run(admitted, request, workload, classes):
    import numpy as np
    import torch
    import torch.distributed as dist

    if (
        os.environ.get("WORLD_SIZE", "1") != "1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or dist.is_initialized()
    ):
        raise RuntimeError("run requires exactly one visible CUDA device, world size one and no process group")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model, actor, data, config_hash = admitted
    base, _, engine_cls, loss_fn, _ = classes
    with tempfile.TemporaryDirectory(prefix="sd35-tuner-") as directory:
        dist.init_process_group("nccl", init_method=f"file://{directory}/init", rank=0, world_size=1)
        try:
            engine = engine_cls(model, actor.engine, actor.optim, actor.checkpoint)
            engine.initialize()
            params = {name: p for name, p in engine.module.named_parameters() if p.requires_grad}
            if not params or not all("lora_" in name for name in params):
                raise RuntimeError("only LoRA tensors may be trainable")
            initial = {name: p.detach().clone() for name, p in params.items()}
            optim_state = copy.deepcopy(engine.optimizer.state_dict())
            scheduler_state = copy.deepcopy(engine.lr_scheduler.state_dict())
            if optim_state["state"]:
                raise RuntimeError("expected empty initial optimizer state")
            data = data.to("cuda:0")
            provenance = {"workload": workload, "effective_config_sha256_without_microbatch": config_hash}
            digest = initial_digest(params, provenance, torch)
            rng = rng_snapshot(torch, np)
            update = partial(loss_fn, config=actor)
            with engine.train_mode():
                base.train_batch(engine, data, update)
            engine.lr_scheduler_step()
            torch.cuda.synchronize()
            restore(engine, params, initial, optim_state, scheduler_state, rng, torch, np)
            if (
                initial_digest(params, provenance, torch) != digest
                or engine.optimizer.state_dict() != optim_state
                or engine.lr_scheduler.state_dict() != scheduler_state
            ):
                raise RuntimeError("warmup state restore failed")
            restored_rng = rng_snapshot(torch, np)
            if (
                not torch.equal(restored_rng[0], rng[0])
                or len(restored_rng[1]) != len(rng[1])
                or any(not torch.equal(a, b) for a, b in zip(restored_rng[1], rng[1], strict=False))
                or restored_rng[2] != rng[2]
                or restored_rng[3][0] != rng[3][0]
                or not np.array_equal(restored_rng[3][1], rng[3][1])
                or restored_rng[3][2:] != rng[3][2:]
            ):
                raise RuntimeError("warmup RNG restore failed")
            seconds, witnesses = [], []
            previous = vector(initial, torch)
            for index in range(2):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with engine.train_mode():
                    out = base.train_batch(engine, data, update)
                    grads = {name: p.grad for name, p in params.items()}
                engine.lr_scheduler_step()
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
                item = witness(out, params, grads, previous, request["accumulation"], torch)
                witnesses.append(item)
                previous = item["params"]
                steps = sorted({int(plain(state["step"], torch).item()) for state in engine.optimizer.state.values()})
                if steps != [index + 1]:
                    raise RuntimeError(f"optimizer step mismatch: {steps}")
            return (
                digest,
                seconds,
                witnesses,
                {"timing_boundary": "train_mode + train_batch + lr_scheduler_step", "config_sha256": config_hash},
            )
        finally:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("identity", "admit", "run"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run" and args.request is None:
        parser.error("run requires --request")
    request = read_json(args.request) if args.request else None
    import torch

    classes = imports()
    workload = identity(args.model.resolve(strict=True), args.batch.resolve(strict=True), classes[0])
    if request is not None:
        validate_request(request, workload)
    if args.mode == "identity":
        write_new(args.result, workload)
        return
    payload = validate_batch(torch.load(args.batch, map_location="cpu", weights_only=True), torch)
    admitted = admit(args.model, payload, request["microbatch"] if request else 1, classes)
    if args.mode == "admit":
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU admission initialized CUDA")
        write_new(args.result, {"status": "admitted", "workload": workload, "config_sha256": admitted[3]})
        return
    try:
        digest, seconds, witnesses, diagnostics = run(admitted, request, workload, classes)
        write_new(args.result.with_suffix(args.result.suffix + ".diagnostics.json"), diagnostics)
        write_new(
            args.result,
            {
                **{key: request[key] for key in RESULT_ECHO},
                "status": "ok",
                "reason": "",
                "initial_state_digest": digest,
                "update_seconds": seconds,
                "witnesses": witnesses,
            },
        )
    except Exception as error:
        if not args.result.exists():
            status = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failure"
            write_new(
                args.result,
                {
                    **{key: request[key] for key in RESULT_ECHO},
                    "status": status,
                    "reason": f"{type(error).__name__}: {error}",
                    "initial_state_digest": None,
                    "update_seconds": [],
                    "witnesses": [],
                },
            )
        raise


if __name__ == "__main__":
    main()
