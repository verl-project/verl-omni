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

"""Fixed public SD3.5 rollout -> PPO actor consistency capture.

Run each role in a fresh process from the pinned source checkout. This is a
functional screening experiment, not a trainer or transport identity test.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import socket
import sys
import tempfile
from pathlib import Path

import capture_scope
import model_contract as probe

HERE = Path(__file__).resolve().parent
OWNERS = (
    "verl_omni/workers/engine/fsdp/diffusers_impl.py",
    "verl_omni/workers/utils/losses.py",
    "verl_omni/pipelines/sd3_flow_grpo/diffusers_training_adapter.py",
    "verl_omni/pipelines/sd3_flow_grpo/vllm_omni_rollout_adapter.py",
    "verl_omni/pipelines/diffusion_rollout_output.py",
)
UIDS = ("sd35-prompt-0", "sd35-prompt-1")
PROMPTS = ("A red cube on a white table.", "A blue sphere on a white table.")
SEEDS = (1234, 1235)
SHAPES = {
    "all_latents": (2, 3, 16, 32, 32),
    "all_timesteps": (2, 2),
    "prompt_embeds": (2, None, 4096),
    "pooled_prompt_embeds": (2, 2048),
    "old_log_probs": (2, 2),
    "advantages": (2, 2),
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_root(output_root: Path, run_id: str) -> Path:
    if type(run_id) is not str or run_id in (".", "..") or RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run-id must be one safe path component")
    base = output_root.resolve()
    run = (base / run_id).resolve()
    if run == base or not run.is_relative_to(base):
        raise ValueError("resolved run root escapes output root")
    return run


def validate_rollout_location(rollout_root: Path, actor_root: Path, rollout_manifest: dict) -> None:
    expected = actor_root.parent / "rollout"
    if (
        rollout_root.resolve(strict=True) != expected
        or rollout_manifest.get("shared_run_root") != str(actor_root.parent.resolve())
        or rollout_manifest.get("rollout_root") != str(expected.resolve(strict=True))
    ):
        raise ValueError("rollout artifact moved from its shared run root")


def validate_source_manifest(source: Path, manifest_path: Path, expected_sha: str, expected_head: str) -> dict:
    """Check the transfer capsule before importing any deployed package."""
    if not HEX64.fullmatch(expected_sha) or sha(manifest_path) != expected_sha:
        raise ValueError("source manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not HEX40.fullmatch(expected_head)
        or type(manifest) is not dict
        or set(manifest) != {"head", "files"}
        or manifest.get("head") != expected_head
    ):
        raise ValueError("source manifest head differs from pinned source")
    rows = manifest.get("files")
    if type(rows) is not list or not rows:
        raise ValueError("source manifest needs files")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256", "size"}:
            raise ValueError("invalid source manifest file row")
        name, digest, size = row["path"], row["sha256"], row["size"]
        if (
            type(name) is not str
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or ".git" in Path(name).parts
        ):
            raise ValueError("invalid source manifest path")
        if name in seen or type(digest) is not str or not HEX64.fullmatch(digest) or type(size) is not int or size < 0:
            raise ValueError("invalid source manifest entry")
        seen.add(name)
        candidate = source / name
        file = candidate.resolve(strict=True)
        if (
            file != candidate
            or not file.is_relative_to(source)
            or not file.is_file()
            or file.stat().st_size != size
            or sha(file) != digest
        ):
            raise ValueError(f"deployed source byte mismatch: {name}")
    if not set(OWNERS).issubset(seen):
        raise ValueError("source manifest omits required owner")
    return manifest


def sampling_kwargs(seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError("seed outside fixed fixture")
    return {
        "seed": seed,
        "height": 256,
        "width": 256,
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_sequence_length": 16,
        "output_type": "latent",
        "num_outputs_per_prompt": 1,
        "extra_args": {
            "noise_level": 0.7,
            "sde_type": "sde",
            "logprobs": True,
            "sde_window_size": 2,
            "sde_window_range": [0, 2],
        },
    }


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def check_batch(batch, torch) -> None:
    if type(batch) is not dict or set(batch) != set(SHAPES):
        raise ValueError("actor batch must have exactly six tensor keys")
    for name, expected in SHAPES.items():
        value = batch[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError(f"{name} must be CPU float32 tensor")
        observed = tuple(value.shape)
        shape_ok = len(observed) == len(expected) and all(
            want is None and got >= 1 or want == got for got, want in zip(observed, expected, strict=True)
        )
        if not shape_ok or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} shape or finiteness mismatch")
    want = torch.tensor([[1.0, 1.0], [-0.5, -0.5]], dtype=torch.float32)
    if not torch.equal(batch["advantages"], want):
        raise ValueError("advantages differ from declared functional fixture")


def extract_output(output, torch) -> dict:
    if getattr(output, "error", None) is not None or getattr(output, "aborted", False):
        raise RuntimeError(f"native rollout error: {getattr(output, 'error', None)}")
    envelope = getattr(output, "output", None)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("metadata"), dict):
        raise ValueError("native rollout payload envelope absent")
    embeds = envelope["metadata"].get("prompt_embeddings")
    if not isinstance(embeds, dict):
        raise ValueError("native prompt embeddings absent")
    fields = {
        "all_latents": getattr(output, "trajectory_latents", None),
        "all_timesteps": getattr(output, "trajectory_timesteps", None),
        "old_log_probs": getattr(output, "trajectory_log_probs", None),
        "prompt_embeds": embeds.get("prompt_embeds"),
        "pooled_prompt_embeds": embeds.get("pooled_prompt_embeds"),
    }
    result = {}
    for name, value in fields.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
            raise ValueError(f"native {name} absent or not B=1")
        result[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    return result


def capture_context(args, weight_sha: str, uid: str, kwargs: dict) -> dict:
    timestep = kwargs.get("timestep")
    if timestep is None or timestep.numel() != 1:
        raise ValueError("first forward needs one actual raw timestep")
    return {
        "run_id": args.run_id,
        "policy_id": weight_sha,
        "global_step": 1,
        "sample_id": uid,
        "trajectory_index": 0,
        "branch": "cond",
        "timestep_unit": "raw_scheduler",
        "timestep": float(timestep.item()),
        "noise_level": 0.7,
        "sde_type": "sde",
    }


def _capture_input_snapshot(inputs):
    """Materialize observer-only storage; the real forward keeps original inputs."""
    import torch

    return {
        name: inputs[name].detach().clone(memory_format=torch.contiguous_format)
        for name in ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep")
    }


def capture_first(module, args, role: str, weight_sha: str, uid: str, output_root: Path):
    blocks = getattr(module, "transformer_blocks", None)
    if blocks is None or len(blocks) < 1:
        raise RuntimeError("actual SD3 transformer_blocks[0] missing")
    original = module.forward
    count = 0

    def wrapped(*pos, **kwargs):
        nonlocal count
        if count:
            return original(*pos, **kwargs)
        if pos:
            raise RuntimeError("actual transformer forward uses unexpected positional inputs")
        count += 1
        return capture_scope.capture_forward(
            enabled=True,
            role=role,
            module=blocks[0],
            forward=lambda: original(*pos, **kwargs),
            inputs=_capture_input_snapshot(kwargs),
            context=capture_context(args, weight_sha, uid, kwargs),
            output_root=output_root,
        )

    module.forward = wrapped
    return original, lambda: count


def one_record(root: Path, role: str) -> dict:
    return json.loads(one_record_path(root, role).read_text(encoding="utf-8"))


def one_record_path(root: Path, role: str) -> Path:
    paths = list((root / role).glob("*/record.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one valid {role} capture, got {len(paths)}")
    return paths[0]


def run_record(record: dict, shared_root: Path) -> dict:
    """Express a same-root dump relative to its shared run, preserving the helper record."""
    path = Path(record["dump_path"]).resolve(strict=True)
    if not path.is_relative_to(shared_root):
        raise ValueError("capture dump is outside run artifact")
    inputs = record.get("inputs")
    names = ("hidden_states", "prompt_embeds", "pooled_projections")
    if type(inputs) is not dict or set(inputs) != set(names):
        raise ValueError("capture input metadata must contain exactly the three pairing fields")
    for name in names:
        if name in record and record[name] != inputs[name]:
            raise ValueError(f"conflicting capture input metadata: {name}")
    return {**record, **inputs, "dump_path": str(path.relative_to(shared_root))}


def provenance(args, _probe, manifest: dict, weight_sha: str) -> dict:
    _, actor_config = probe.configure(args.model)
    config_bytes = json.dumps(dataclasses.asdict(actor_config), sort_keys=True, default=str).encode()
    return {
        "source_head": manifest["head"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "runner_sha256": sha(Path(__file__)),
        "capture_helper_sha256": sha(HERE / "capture_scope.py"),
        "probe_sha256": sha(HERE / "model_contract.py"),
        "transformer_checkpoint_source_revision": probe.MODEL_REVISION,
        "transformer_weight_sha256": weight_sha,
        "transformer_weight_hash_verified": args.role != "admit",
        "cached_model_metadata": probe.model_identity(args.model),
        "effective_actor_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "settings": {
            "dtype": "fp32",
            "actor_attention": "native",
            "rollout_attention": "TORCH_SDPA",
            "rollout_offload": args.rollout_offload,
            "cfg": False,
            "height": 256,
            "width": 256,
            "steps": 4,
            "noise_level": 0.7,
            "sde_type": "sde",
            "sde_window_size": 2,
            "sde_window_range": [0, 2],
            "max_sequence_length": 16,
            "actor_microbatch": 1,
            "policy_version": 0,
            "trainer_diagnostic_global_step": 1,
            "msprobe_local_step": 0,
        },
    }


def _enrich_rollout_config(config, model_dir):
    """Run the public stage metadata step before native worker construction."""
    config.enrich_config()
    expected = json.loads((model_dir / "transformer/config.json").read_text())
    names = (
        "num_layers",
        "sample_size",
        "in_channels",
        "out_channels",
        "num_attention_heads",
        "attention_head_dim",
        "caption_projection_dim",
        "pooled_projection_dim",
        "joint_attention_dim",
        "patch_size",
        "pos_embed_max_size",
    )
    observed = {}
    for name in names:
        actual = getattr(config.tf_model_config, name, None)
        if name not in expected or actual != expected[name]:
            raise ValueError(f"rollout transformer config missing or inconsistent: {name}")
        observed[name] = actual
    return observed


def _validate_debugger_api(debugger_type) -> None:
    """Bind the public classmethod calls without creating a debugger."""
    inspect.signature(debugger_type).bind(config_path="/nonexistent/admission-only.json")
    inspect.signature(debugger_type.start).bind(model=None)
    inspect.signature(debugger_type.stop).bind()
    inspect.signature(debugger_type.step).bind()


def rollout_offload_kwargs(mode: str) -> dict:
    if mode == "none":
        return {}
    if mode == "dit-layerwise":
        return {"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}}
    raise ValueError(f"unsupported rollout offload mode: {mode}")


def cuda_memory_metrics(torch) -> dict:
    free, _ = torch.cuda.mem_get_info()
    return {
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_free": free,
    }


def admit(args, probe, identity: dict, manifest: dict, root: Path):
    import torch
    from msprobe.pytorch import PrecisionDebugger
    from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb
    from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine

    _validate_debugger_api(PrecisionDebugger)
    actor_model, actor = probe.configure(args.model)
    actor_engine = actor.engine
    od = OmniDiffusionConfig(
        model=str(args.model),
        model_class_name="StableDiffusion3Pipeline",
        dtype=torch.float32,
        num_gpus=1,
        enforce_eager=True,
        master_port=29500,
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        **rollout_offload_kwargs(args.rollout_offload),
    )
    derived_config = _enrich_rollout_config(od, args.model)
    inspect.signature(DiffusionWorker).bind(0, 0, od, skip_load_model=True)
    inspect.signature(DiffusionWorker.load_model).bind(
        None, load_format="custom_pipeline", custom_pipeline_name=StableDiffusion3PipelineWithLogProb
    )
    inspect.signature(DiffusionWorker.execute_model).bind(None, None, od)
    inspect.signature(OmniDiffusionSamplingParams).bind(**sampling_kwargs(SEEDS[0]))
    inspect.signature(OmniDiffusionRequest).bind(
        prompt={"prompt": PROMPTS[0]}, sampling_params=None, request_id="probe-admit"
    )
    inspect.signature(PPODiffusersFSDPEngine).bind(actor_model, actor_engine, actor.optim, actor.checkpoint)
    inspect.signature(PPODiffusersFSDPEngine.forward_backward_batch).bind(
        None, None, loss_function=None, forward_only=True
    )
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
        str(args.model), subfolder="scheduler", local_files_only=True
    )
    from verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter import StableDiffusion3FlowGRPO

    StableDiffusion3FlowGRPO.set_timesteps(scheduler, actor_model, "cpu")
    if len(scheduler.timesteps) != 4 or torch.cuda.is_initialized():
        raise RuntimeError("CPU admission scheduler/CUDA contract failed")
    root.mkdir(parents=True, exist_ok=False)
    (root / "admission.json").write_text(
        json.dumps(
            {
                "role": "admit",
                "identity": identity,
                "provenance": provenance(args, probe, manifest, probe.WEIGHT_SHA256),
                "rollout_transformer_config": derived_config,
                "scheduler_timesteps": scheduler.timesteps.tolist(),
                "cuda_initialized": False,
            },
            indent=2,
        )
        + "\n"
    )


def validate_admission(args, identity: dict, manifest: dict, shared_root: Path) -> str:
    path = shared_root / "admit/admission.json"
    raw = path.read_bytes()
    admission = json.loads(raw)
    expected = provenance(args, probe, manifest, probe.WEIGHT_SHA256)
    expected["transformer_weight_hash_verified"] = False
    if (
        type(admission) is not dict
        or admission.get("role") != "admit"
        or admission.get("identity") != identity
        or admission.get("provenance") != expected
        or admission.get("cuda_initialized") is not False
        or type(admission.get("scheduler_timesteps")) is not list
        or len(admission["scheduler_timesteps"]) != 4
        or type(admission.get("rollout_transformer_config")) is not dict
    ):
        raise ValueError("CPU admission token does not match this run/source/model/config")
    return hashlib.sha256(raw).hexdigest()


class NumericalMismatch(RuntimeError):
    """Complete producer bytes exist, but log probabilities differ."""


def rollout(args, probe, identity: dict, manifest: dict, root: Path, weight_sha: str):
    import torch
    from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("rollout needs exactly one visible GPU")
    root.mkdir(parents=True, exist_ok=False)
    captures = root / "captures"
    captures.mkdir()
    port = free_local_port()
    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    od = OmniDiffusionConfig(
        model=str(args.model),
        model_class_name="StableDiffusion3Pipeline",
        dtype=torch.float32,
        num_gpus=1,
        enforce_eager=True,
        master_port=port,
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        **rollout_offload_kwargs(args.rollout_offload),
    )
    _enrich_rollout_config(od, args.model)
    worker = None
    try:
        worker = DiffusionWorker(0, 0, od, skip_load_model=True)
        worker.load_model(load_format="custom_pipeline", custom_pipeline_name=StableDiffusion3PipelineWithLogProb)
        if args.rollout_offload == "dit-layerwise":
            backend = worker.model_runner.offload_backend
            if backend is None or not backend.enabled:
                raise RuntimeError("DiT layerwise offload was not activated")
            print(json.dumps({"milestone": "rollout-loaded", **cuda_memory_metrics(torch)}), flush=True)
        pipeline = worker.model_runner.pipeline
        if not isinstance(pipeline, StableDiffusion3PipelineWithLogProb):
            raise RuntimeError("custom real pipeline was not loaded")
        transformer = pipeline.transformer
        original, count = capture_first(transformer, args, "rollout", weight_sha, UIDS[0], captures)
        outputs = []
        mapping = []
        try:
            for uid, prompt, seed in zip(UIDS, PROMPTS, SEEDS, strict=True):
                reqid = f"{args.run_id}:{uid}"
                req = OmniDiffusionRequest(
                    prompt={"prompt": prompt},
                    sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs(seed)),
                    request_id=reqid,
                )
                out = worker.execute_model(req, od)
                outputs.append(extract_output(out, torch))
                if args.rollout_offload == "dit-layerwise":
                    print(
                        json.dumps(
                            {"milestone": "rollout-request-completed", "sample": uid, **cuda_memory_metrics(torch)}
                        ),
                        flush=True,
                    )
                mapping.append(
                    {
                        "request_id": req.request_id,
                        "sample_id": uid,
                        "prompt": prompt,
                        "seed": seed,
                        "api_output_index": len(outputs) - 1,
                    }
                )
        finally:
            transformer.forward = original
        if count() != 1:
            raise RuntimeError("rollout first-forward capture cardinality mismatch")
        batch = {name: torch.cat([part[name] for part in outputs], dim=0) for name in outputs[0]}
        batch["advantages"] = torch.tensor([[1.0, 1.0], [-0.5, -0.5]], dtype=torch.float32)
        check_batch(batch, torch)
        torch.save(batch, root / "actor_batch.pt")
        record = run_record(one_record(captures, "rollout"), root.parent)
        (root / "rollout_record.json").write_text(json.dumps(record, indent=2) + "\n")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "role": "rollout",
                    "real_prompt": True,
                    "shared_run_root": str(root.parent.resolve()),
                    "rollout_root": str(root.resolve()),
                    "identity": identity,
                    "provenance": provenance(args, probe, manifest, weight_sha),
                    "request_map": mapping,
                    "actor_batch_sha256": sha(root / "actor_batch.pt"),
                    "capture_record": record,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        primary = sys.exc_info()[1]
        if worker is not None:
            try:
                worker.shutdown()
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"DiffusionWorker.shutdown also failed: {cleanup!r}")


def actor(args, probe, identity: dict, manifest: dict, root: Path, weight_sha: str):
    import torch
    import torch.distributed as dist
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    rollout_root = args.rollout_root.resolve(strict=True)
    rollout_manifest = json.loads((rollout_root / "manifest.json").read_text())
    validate_rollout_location(rollout_root, root, rollout_manifest)
    if (
        rollout_manifest.get("role") != "rollout"
        or rollout_manifest.get("real_prompt") is not True
        or rollout_manifest.get("rollout_root") != str(rollout_root)
    ):
        raise ValueError("rollout identity or real-prompt declaration mismatch")
    if rollout_manifest.get("identity") != identity or rollout_manifest.get("provenance") != provenance(
        args, probe, manifest, weight_sha
    ):
        raise ValueError("rollout complete provenance differs from actor")
    request_map = rollout_manifest.get("request_map")
    if (
        type(request_map) is not list
        or len(request_map) != 2
        or any(
            type(row) is not dict
            or row.get("request_id") != f"{args.run_id}:{uid}"
            or row.get("sample_id") != uid
            or row.get("prompt") != prompt
            or row.get("seed") != seed
            or row.get("api_output_index") != index
            for index, (row, uid, prompt, seed) in enumerate(zip(request_map, UIDS, PROMPTS, SEEDS, strict=True))
        )
    ):
        raise ValueError("rollout request identity map mismatch")
    batch_path = rollout_root / "actor_batch.pt"
    if sha(batch_path) != rollout_manifest.get("actor_batch_sha256"):
        raise ValueError("actor batch SHA256 mismatch")
    batch = torch.load(batch_path, map_location="cpu", weights_only=True)
    check_batch(batch, torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or dist.is_initialized():
        raise RuntimeError("actor needs fresh process with exactly one visible GPU")
    root.mkdir(parents=True, exist_ok=False)
    captures = root / "captures"
    captures.mkdir()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, actor_config = probe.configure(args.model)
    engine_config = actor_config.engine
    with tempfile.TemporaryDirectory(prefix="paired-actor-rdzv-") as rendezvous:
        dist.init_process_group("nccl", init_method=f"file://{rendezvous}/init", rank=0, world_size=1)
        try:
            from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine

            engine = PPODiffusersFSDPEngine(model, engine_config, actor_config.optim, actor_config.checkpoint)
            engine.initialize()
            original, count = capture_first(engine.module, args, "actor", weight_sha, UIDS[0], captures)
            data = TensorDict(batch, batch_size=2).to("cuda:0")
            tu.assign_non_tensor(data, micro_batch_size_per_gpu=1, height=256, width=256, vae_scale_factor=8)
            try:
                with engine.eval_mode():
                    result = engine.forward_backward_batch(data, loss_function=None, forward_only=True)
            finally:
                engine.module.forward = original
            if count() != 1:
                raise RuntimeError("actor first-forward capture cardinality mismatch")
            log_probs = result["model_output"]["log_probs"]
            actual = probe.plain(log_probs).float()
            if actual.shape != (2, 2) or not bool(torch.isfinite(actual).all()):
                raise ValueError("real actor log probabilities invalid")
            old = batch["old_log_probs"]
            maxdiff = float((actual - old).abs().max())
            close = bool(torch.allclose(actual, old, atol=1e-5, rtol=1e-3))
            torch.save({"actor_log_probs": actual, "rollout_log_probs": old}, root / "comparison.pt")
            actor_record = run_record(one_record(captures, "actor"), root.parent)
            comparison = {
                "role": "actor",
                "identity": identity,
                "provenance": provenance(args, probe, manifest, weight_sha),
                "shared_run_root": str(root.parent.resolve()),
                "rollout_root": str(rollout_root),
                "rollout_batch_sha256": sha(batch_path),
                "comparison_sha256": sha(root / "comparison.pt"),
                "max_abs_logprob_difference": maxdiff,
                "allclose_atol": 1e-5,
                "allclose_rtol": 1e-3,
                "allclose": close,
                "pair_status": "pending_public_replay",
                "capture_record": actor_record,
            }
            # The public CPU replay reader owns pairing and byte validation.
            (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
            if not close:
                raise NumericalMismatch("actor/rollout numerical mismatch; see comparison.json")
        finally:
            primary = sys.exc_info()[1]
            try:
                dist.destroy_process_group()
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"process-group destroy also failed: {cleanup!r}")


def source_identity(source: Path, manifest: dict) -> dict:
    import diffusers
    import msprobe
    import torch
    import verl
    import vllm
    import vllm_omni

    import verl_omni
    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb
    from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine

    for owner in (verl_omni,):
        if not Path(owner.__file__).resolve().is_relative_to(source):
            raise RuntimeError("package imported from outside declared source")
    for symbol in (PPODiffusersFSDPEngine, StableDiffusion3PipelineWithLogProb):
        if not Path(inspect.getfile(symbol)).resolve().is_relative_to(source):
            raise RuntimeError("owner imported from outside declared source")
    from verl.workers.engine.base import BaseEngine

    upstream_file = Path(inspect.getfile(BaseEngine)).resolve(strict=True)
    return {
        "owners": {row["path"]: row["sha256"] for row in manifest["files"]},
        "upstream_engine_base": {"path": str(upstream_file), "sha256": sha(upstream_file)},
        "versions": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "verl": getattr(verl, "__version__", "unknown"),
            "vllm_module": getattr(vllm, "__version__", "unknown"),
            "vllm_distribution": importlib.metadata.version("vllm"),
            "vllm_omni_module": getattr(vllm_omni, "__version__", "unknown"),
            "msprobe_module": getattr(msprobe, "__version__", "unknown"),
            "vllm_omni_distribution": importlib.metadata.version("vllm-omni"),
            "mindstudio_probe_distribution": importlib.metadata.version("mindstudio-probe"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("admit", "rollout", "actor"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--rollout-offload", choices=("none", "dit-layerwise"), default="none")
    args = parser.parse_args()
    shared_root = run_root(args.output_root, args.run_id)
    if args.role == "admit" and shared_root.exists():
        raise FileExistsError("capture run root must be fresh")
    source = args.source.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    if Path.cwd().resolve() != source:
        raise RuntimeError("run from deployed pinned source directory")
    manifest = validate_source_manifest(
        source, args.source_manifest, args.source_manifest_sha256, args.expected_source_head
    )
    if source != Path(os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]).resolve():
        raise RuntimeError("deployed source must be first PYTHONPATH entry")
    if (args.role == "actor") != (args.rollout_root is not None):
        parser.error("--rollout-root is required exactly for actor")
    identity = source_identity(source, manifest)
    root = shared_root / args.role
    if args.role != "admit":
        validate_admission(args, identity, manifest, shared_root)
    if args.role == "admit":
        admit(args, probe, identity, manifest, root)
    else:
        weight_sha = probe.verify_weight(args.model)
        if args.role == "rollout":
            rollout(args, probe, identity, manifest, root, weight_sha)
        else:
            try:
                actor(args, probe, identity, manifest, root, weight_sha)
            except NumericalMismatch as exc:
                print(str(exc), file=sys.stderr)
                raise SystemExit(10) from exc


if __name__ == "__main__":
    main()
