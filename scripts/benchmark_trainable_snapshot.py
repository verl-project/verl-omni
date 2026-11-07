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
"""Measure pretrained Qwen-Image snapshot bookkeeping, not training throughput."""

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import verl
from omegaconf import OmegaConf
from snapshot_benchmark_metrics import arm_order, summarize_pairs
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig

from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.detach_actor_worker import DiffusionDetachActorWorker, _TrainableSnapshot
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]
_MODEL_METADATA = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
)
_PREFLIGHT_TIMEOUT_SECONDS = 900
_ATTENTION_TARGETS = frozenset(TARGETS[:8])


def _git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def _regular_path(root, relative):
    path = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are forbidden in benchmark provenance: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing regular provenance file: {relative}")
    return path


def _tracked_file(root, revision, relative):
    path = _regular_path(root, relative)
    working = path.read_bytes()
    pinned = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root)
    if working != pinned:
        raise ValueError(f"Tracked metadata differs from pinned Git bytes: {relative}")
    return hashlib.sha256(working).hexdigest()


def _symbol_source(symbol, expected_root, expected_relative):
    actual = Path(inspect.getsourcefile(symbol) or "").resolve()
    expected = _regular_path(expected_root, expected_relative).resolve()
    if actual != expected:
        raise ValueError(f"Imported {symbol.__qualname__} from {actual}, expected {expected}")
    return {"path": str(actual), "sha256": hashlib.sha256(actual.read_bytes()).hexdigest()}


def _provenance(args):
    root = Path(__file__).resolve().parents[1]
    if _git(root, "rev-parse", "HEAD") != args.source_revision:
        raise ValueError("Actual source Git revision differs from the declared revision")
    tracked_source = [
        "verl_omni/workers/detach_actor_worker.py",
        "verl_omni/workers/engine/fsdp/diffusers_impl.py",
        "verl_omni/workers/engine/lora_adapter_mixin.py",
        "verl_omni/workers/config/diffusion/model.py",
        ".github/verl_pin.txt",
    ]
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *tracked_source],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    paths = ["scripts/benchmark_trainable_snapshot.py", "scripts/snapshot_benchmark_metrics.py", *tracked_source]
    digests = {path: hashlib.sha256(_regular_path(root, path).read_bytes()).hexdigest() for path in paths}
    imported = {
        "worker": _symbol_source(DiffusionDetachActorWorker, root, "verl_omni/workers/detach_actor_worker.py"),
        "snapshot_helper": _symbol_source(_TrainableSnapshot, root, "verl_omni/workers/detach_actor_worker.py"),
        "engine": _symbol_source(PPODiffusersFSDPEngine, root, "verl_omni/workers/engine/fsdp/diffusers_impl.py"),
        "lora_adapter_mixin": _symbol_source(LoRAAdapterMixin, root, "verl_omni/workers/engine/lora_adapter_mixin.py"),
        "model_config": _symbol_source(DiffusionModelConfig, root, "verl_omni/workers/config/diffusion/model.py"),
        "metrics_helper": _symbol_source(summarize_pairs, root, "scripts/snapshot_benchmark_metrics.py"),
    }
    verl_package = Path(verl.__file__).resolve().parent
    verl_root = Path(_git(verl_package, "rev-parse", "--show-toplevel")).resolve()
    verl_pin = (root / ".github/verl_pin.txt").read_text().strip()
    if _git(verl_root, "rev-parse", "HEAD") != verl_pin:
        raise ValueError("Imported verl checkout HEAD differs from .github/verl_pin.txt")
    handler_worker = object.__new__(DiffusionDetachActorWorker)
    handler_worker.config = OmegaConf.create({"actor": {"strategy": "fsdp2"}})
    handler_worker._strategy_handlers = None
    save_handler, restore_handler = handler_worker._get_strategy_handlers()
    handler_resolver_path = "verl/experimental/separation/engine_workers.py"
    handler_helper_path = "verl/utils/fsdp_utils.py"
    verl_symbols = (CheckpointConfig, FSDPEngineConfig, FSDPOptimizerConfig)
    verl_sources = sorted(
        {str(Path(inspect.getsourcefile(symbol) or "").resolve().relative_to(verl_root)) for symbol in verl_symbols}
        | {handler_resolver_path, handler_helper_path}
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *verl_sources],
        cwd=verl_root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    imported["verl"] = {
        "root": str(verl_root),
        "revision": verl_pin,
        "sources": {
            path: hashlib.sha256(_regular_path(verl_root, path).read_bytes()).hexdigest() for path in verl_sources
        },
        "strategy_handlers": {
            "resolver": _symbol_source(handler_worker._get_strategy_handlers, verl_root, handler_resolver_path),
            "save": _symbol_source(save_handler, verl_root, handler_helper_path),
            "restore": _symbol_source(restore_handler, verl_root, handler_helper_path),
        },
    }
    return {
        "source_files": digests,
        "imported_sources": imported,
        "source_bundle_sha256": hashlib.sha256(
            json.dumps({"files": digests, "imports": imported}, sort_keys=True).encode()
        ).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "peft")},
    }


def _verify_weights(args):
    # Run on rank zero before CUDA/NCCL; model data is never accepted from its path/size alone.
    if args.model.is_symlink() or _git(args.model, "rev-parse", "HEAD") != args.checkpoint_revision:
        raise ValueError("Model root must be a non-symlink Git checkout at checkpoint revision")
    metadata = {path: _tracked_file(args.model, args.checkpoint_revision, path) for path in _MODEL_METADATA}
    if args.checkpoint_manifest.is_symlink() or not args.checkpoint_manifest.is_file():
        raise ValueError("Checkpoint manifest must be a regular non-symlink file")
    manifest = {}
    sizes = {}
    for row in args.checkpoint_manifest.read_text().splitlines():
        name, size, digest = row.split("|")
        if not re.fullmatch(r"transformer/diffusion_pytorch_model-\d{5}-of-00009.safetensors", name):
            raise ValueError("Unexpected checkpoint manifest path")
        if name in manifest or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Duplicate checkpoint object or invalid digest")
        pointer = subprocess.check_output(
            ["git", "show", f"{args.checkpoint_revision}:{name}"], cwd=args.model, text=True
        )
        if f"oid sha256:{digest}\n" not in pointer or f"size {int(size)}\n" not in pointer:
            raise ValueError("Manifest differs from the pinned Git LFS pointer")
        path = args.model / name
        if path.is_symlink() or path.stat().st_size != int(size):
            raise ValueError("Missing or wrong-sized checkpoint shard")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
        manifest[name] = digest
        sizes[name] = int(size)
    index_path = _regular_path(args.model, "transformer/diffusion_pytorch_model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError("Malformed transformer safetensors index")
    indexed_names = list(index["weight_map"].values())
    if any(
        not isinstance(name, str) or not re.fullmatch(r"diffusion_pytorch_model-\d{5}-of-00009.safetensors", name)
        for name in indexed_names
    ):
        raise ValueError("Transformer index contains an invalid shard path")
    indexed = {f"transformer/{name}" for name in indexed_names}
    if len(manifest) != 9 or set(manifest) != indexed:
        raise ValueError("Checkpoint manifest must cover every indexed transformer shard")
    config = json.loads(_regular_path(args.model, "transformer/config.json").read_text())
    if (
        config.get("_class_name"),
        config.get("num_layers"),
        config.get("num_attention_heads"),
        config.get("attention_head_dim"),
    ) != ("QwenImageTransformer2DModel", 60, 24, 128):
        raise ValueError("Expected original-width/depth Qwen-Image checkpoint")
    return {
        "checkpoint_manifest_sha256": hashlib.sha256(args.checkpoint_manifest.read_bytes()).hexdigest(),
        "metadata_sha256": metadata,
        "shards_sha256": manifest,
        "index_tensor_bytes": index.get("metadata", {}).get("total_size"),
        "shard_file_bytes": sum(sizes.values()),
        "transformer_config": config,
    }


def _write(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _manifest_payload(args, provenance, checkpoint):
    return {
        "schema": 1,
        "source_revision": args.source_revision,
        "checkpoint_revision": args.checkpoint_revision,
        "source": provenance,
        "checkpoint": checkpoint,
        "configuration": {
            "run_id": args.run_id,
            "model": str(args.model.resolve()),
            "layers": args.layers,
            "offload": args.offload,
            "lora_rank": args.lora_rank,
            "lora_alpha": 2 * args.lora_rank,
            "target_modules": TARGETS,
            "base_dtype": "bfloat16",
            "lora_dtype": "float32",
            "sync_steps": args.sync_steps,
            "warmup_pairs": args.warmup,
            "timed_pairs": args.repeats,
            "seed": args.seed,
            "world_size": 2,
        },
        "runtime": {"python": platform.python_version(), **provenance["versions"]},
    }


def _manifest_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _preflight(args, rank):
    """Verify provenance before CUDA/NCCL; rank zero owns output and checkpoint hashing."""
    provenance = _provenance(args)  # Every rank validates its actual imports independently.
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("Frozen benchmark matrix requires exactly two GPU ranks")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        try:
            checkpoint = _verify_weights(args)
            payload = _manifest_payload(args, provenance, checkpoint)
            digest = _manifest_digest(payload)
            _write(args.output / "run-manifest.json", {"manifest_sha256": digest, "manifest": payload})
            _write(
                args.output / "preflight.json",
                {"status": "passed", "run_id": args.run_id, "manifest_sha256": digest},
            )
            return digest, provenance, checkpoint
        except BaseException as error:
            _write(
                args.output / "preflight.json",
                {"status": "failed", "run_id": args.run_id, "error_type": type(error).__name__, "error": str(error)},
            )
            raise

    deadline = time.monotonic() + _PREFLIGHT_TIMEOUT_SECONDS
    preflight_path = args.output / "preflight.json"
    while time.monotonic() < deadline:
        try:
            preflight = json.loads(preflight_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        if preflight.get("run_id") != args.run_id:
            raise RuntimeError("Output directory belongs to a different benchmark run_id")
        if preflight.get("status") == "failed":
            raise RuntimeError(f"Rank-zero preflight failed: {preflight.get('error_type')}: {preflight.get('error')}")
        if preflight.get("status") != "passed":
            raise RuntimeError("Invalid preflight status artifact")
        envelope = json.loads((args.output / "run-manifest.json").read_text())
        digest = preflight.get("manifest_sha256")
        if envelope.get("manifest_sha256") != digest or _manifest_digest(envelope.get("manifest")) != digest:
            raise RuntimeError("Run manifest digest mismatch")
        if envelope["manifest"]["source"]["source_bundle_sha256"] != provenance["source_bundle_sha256"]:
            raise RuntimeError("Rank-local imported source identity differs from run manifest")
        return digest, provenance, envelope["manifest"]["checkpoint"]
    raise TimeoutError(f"Timed out after {_PREFLIGHT_TIMEOUT_SECONDS}s waiting for rank-zero preflight")


def _local(param):
    return param.to_local() if isinstance(param, DTensor) else param


def _payload(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_payload(item) for item in value.values())
    if isinstance(value, tuple | list):
        return sum(_payload(item) for item in value)
    return 0


def _fingerprints(module):
    # Stream one local tensor at a time; never retain full-model baseline copies.
    digests = {False: hashlib.sha256(), True: hashlib.sha256()}
    for name, param in module.named_parameters():
        tensor = _local(param).detach().cpu().contiguous()
        digest = digests[param.requires_grad]
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return {str(key): value.hexdigest() for key, value in digests.items()}


class _FullSnapshotWorker(DiffusionDetachActorWorker):
    def _supports_trainable_snapshot(self, module):
        return False


def _worker(engine, arm):
    cls = DiffusionDetachActorWorker if arm == "trainable" else _FullSnapshotWorker
    worker = object.__new__(cls)
    worker.actor = SimpleNamespace(engine=engine)
    worker.config = OmegaConf.create({"actor": {"strategy": "fsdp2"}})
    worker._strategy_handlers = None
    worker.cpu_saved_models = {}
    return worker


def _engine(args):
    class PrefixEngine(PPODiffusersFSDPEngine):
        def _build_module(self):
            module = super()._build_module()
            if not 0 < args.layers <= len(module.transformer_blocks):
                raise ValueError("Requested depth exceeds pretrained transformer depth")
            module.transformer_blocks = module.transformer_blocks[: args.layers]
            module.register_to_config(num_layers=args.layers)
            return module

    torch.manual_seed(args.seed)
    model = DiffusionModelConfig(
        path=str(args.model),
        algorithm="flow_grpo",
        load_tokenizer=False,
        attn_backend="native",
        enable_gradient_checkpointing=False,
        lora_rank=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dtype="float32",
        target_modules=TARGETS,
        policy_state_adapters=("default",),
    )
    config = FSDPEngineConfig(
        strategy="fsdp2",
        fsdp_size=dist.get_world_size(),
        ulysses_sequence_parallel_size=1,
        model_dtype="bfloat16",
        dtype="bfloat16",
        param_offload=args.offload,
        optimizer_offload=False,
        forward_only=False,
        use_orig_params=True,
        mixed_precision={"param_dtype": "bfloat16", "reduce_dtype": "float32", "buffer_dtype": "float32"},
    )
    engine = PrefixEngine(model, config, FSDPOptimizerConfig(total_training_steps=10), CheckpointConfig())
    engine.initialize()
    return engine


def _validate_parameter_contract(engine, args):
    """Require exactly one default-adapter A/B expansion for every requested block target."""
    modules = dict(engine.module.named_modules())
    expected_ids = set()
    expansions = []
    for block in range(args.layers):
        for target in TARGETS:
            prefix = f"transformer_blocks.{block}."
            module_name = f"{prefix}attn.{target}" if target in _ATTENTION_TARGETS else f"{prefix}{target}"
            if module_name not in modules:
                raise RuntimeError(f"Missing effective LoRA target at {module_name!r}")
            module = modules[module_name]
            if set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}:
                raise RuntimeError(f"Unexpected adapter expansion at {module_name}")
            for side in ("lora_A", "lora_B"):
                adapter = getattr(module, side)["default"]
                parameters = list(adapter.named_parameters())
                if len(parameters) != 1 or parameters[0][0] != "weight":
                    raise RuntimeError(f"Ambiguous {side} parameter structure at {module_name}")
                expected_ids.add(id(parameters[0][1]))
            expansions.append(module_name)
    named = list(engine.module.named_parameters())
    trainable = [(name, param) for name, param in named if param.requires_grad]
    if {id(param) for _, param in trainable} != expected_ids or len(trainable) != 2 * args.layers * len(TARGETS):
        raise RuntimeError("Trainable parameters differ from the exact requested default-adapter LoRA expansion")
    if any(_local(param).dtype != torch.float32 for _, param in trainable):
        raise RuntimeError("Every trainable LoRA parameter must be FP32")
    frozen = [(name, param) for name, param in named if not param.requires_grad]
    if not frozen or any(_local(param).dtype != torch.bfloat16 for _, param in frozen):
        raise RuntimeError("Every frozen model parameter must be BF16")

    def histogram(values):
        result = {}
        for _, param in values:
            key = str(_local(param).dtype)
            result[key] = result.get(key, 0) + 1
        return result

    return {
        "effective_target_modules": expansions,
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": len(trainable),
        "trainable_element_count": sum(_local(param).numel() for _, param in trainable),
        "frozen_parameter_count": len(frozen),
        "frozen_element_count": sum(_local(param).numel() for _, param in frozen),
        "trainable_dtype_histogram": histogram(trainable),
        "frozen_dtype_histogram": histogram(frozen),
    }


def _correctness(worker, arm):
    module = worker.actor.engine.module
    identities = tuple(id(param) for param in module.parameters())
    original = _fingerprints(module)
    try:
        worker.save_model_to_cpu(0)
        assert isinstance(worker.cpu_saved_models[0], _TrainableSnapshot) == (arm == "trainable")
        for param in module.parameters():
            local = _local(param)
            if param.requires_grad and local.numel():
                local.reshape(-1)[0].add_(1)
        changed = _fingerprints(module)
        assert changed["False"] == original["False"] and changed["True"] != original["True"]
        worker.save_model_to_cpu(1)
        retained_bytes = sum(
            _payload(snapshot.state if isinstance(snapshot, _TrainableSnapshot) else snapshot)
            for snapshot in worker.cpu_saved_models.values()
        )
        worker.restore_model_from_cpu(0)
        assert _fingerprints(module) == original
        worker.restore_model_from_cpu(1)
        assert _fingerprints(module) == changed
        worker.restore_model_from_cpu(0)
        assert _fingerprints(module) == original
        assert tuple(id(param) for param in module.parameters()) == identities
    finally:
        worker.clear_cpu_model(1)
        worker.clear_cpu_model(0)
    assert not worker.cpu_saved_models
    return retained_bytes


def _rss():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class _RssSampler:
    def __init__(self):
        self.baseline = _rss()
        self.peak = self.baseline
        self.samples = 1
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(0.005):
            self.peak = max(self.peak, _rss())
            self.samples += 1

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, _rss())


def _reclaim():
    gc.collect()
    torch.cuda.empty_cache()
    # Equal, untimed allocator preparation for both arms; never drop OS page cache.
    libc = ctypes.CDLL(None)
    if not hasattr(libc, "malloc_trim"):
        raise RuntimeError("Memory comparison requires glibc malloc_trim on Linux")
    libc.malloc_trim(0)
    torch.cuda.synchronize()
    dist.barrier()


def _cycle(worker, sync_steps):
    save_seconds = 0.0
    restore_seconds = 0.0

    def call(method, snapshot_id):
        start = time.perf_counter()
        method(snapshot_id)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    try:
        save_seconds += call(worker.save_model_to_cpu, 0)
        for _ in range(1, sync_steps):
            save_seconds += call(worker.save_model_to_cpu, 1)
            restore_seconds += call(worker.restore_model_from_cpu, 0)
            restore_seconds += call(worker.restore_model_from_cpu, 1)
            worker.clear_cpu_model(1)
    finally:
        worker.clear_cpu_model(1)
        worker.clear_cpu_model(0)
    torch.cuda.synchronize()
    return {
        "cycle_seconds": time.perf_counter() - start,
        "save_seconds": save_seconds,
        "restore_seconds": restore_seconds,
        "save_calls": sync_steps,
        "restore_calls": 2 * (sync_steps - 1),
    }


def _run(args, manifest_sha256, provenance, verified):
    rank = dist.get_rank()
    engine = _engine(args)
    parameter_contract = _validate_parameter_contract(engine, args)
    workers = {arm: _worker(engine, arm) for arm in ("full", "trainable")}
    sizes = {"trainable": 0, "frozen": 0}
    for param in engine.module.parameters():
        sizes["trainable" if param.requires_grad else "frozen"] += _payload(_local(param))
    expected = {"full": 2 * sum(sizes.values()), "trainable": 2 * sizes["trainable"]}
    _write(
        args.output / f"progress-rank{rank}.json",
        {
            "manifest_sha256": manifest_sha256,
            "phase": "model_loaded",
            "local_bytes": sizes,
            "parameter_contract": parameter_contract,
        },
    )
    for arm, worker in workers.items():
        assert _correctness(worker, arm) == expected[arm]
    original_fingerprints = _fingerprints(engine.module)
    original_identities = tuple(id(param) for param in engine.module.parameters())
    _write(
        args.output / f"progress-rank{rank}.json",
        {"manifest_sha256": manifest_sha256, "phase": "correctness_passed"},
    )
    print(f"BENCH_CORRECTNESS_PASS rank={rank} layers={args.layers}", flush=True)
    for pair in range(args.warmup):
        for arm in arm_order(pair):
            _cycle(workers[arm], args.sync_steps)
            dist.barrier()
    rows = []
    for pair in range(args.repeats):
        for arm in arm_order(pair):
            _reclaim()
            row = {
                "pair": pair,
                "arm": arm,
                "rank": rank,
                "retained_snapshot_bytes": expected[arm],
                **_cycle(workers[arm], args.sync_steps),
            }
            # Untimed: prevent faster-rank cleanup or JSON I/O from contaminating the next observation.
            dist.barrier()
            rows.append(row)
            _write(
                args.output / f"timing-rank{rank}.json",
                {"manifest_sha256": manifest_sha256, "timings": rows},
            )
        print(f"BENCH_PAIR_DONE rank={rank} pair={pair}", flush=True)
    memory = []
    for pair in range(2):
        for arm in arm_order(pair):
            _reclaim()
            torch.cuda.reset_peak_memory_stats()
            cuda_allocated = torch.cuda.memory_allocated()
            cuda_reserved = torch.cuda.memory_reserved()
            with _RssSampler() as sampler:
                cycle = _cycle(workers[arm], args.sync_steps)
            # Stop RSS sampling before the untimed synchronization wait.
            dist.barrier()
            memory.append(
                {
                    "pair": pair,
                    "arm": arm,
                    "rank": rank,
                    "cycle_seconds": cycle["cycle_seconds"],
                    "rss_baseline_bytes": sampler.baseline,
                    "rss_peak_sampled_bytes": sampler.peak,
                    "rss_peak_delta_bytes": sampler.peak - sampler.baseline,
                    "rss_samples": sampler.samples,
                    "rss_sample_interval_seconds": 0.005,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "cuda_baseline_allocated_bytes": cuda_allocated,
                    "cuda_baseline_reserved_bytes": cuda_reserved,
                    "cuda_peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - cuda_allocated,
                    "cuda_peak_reserved_delta_bytes": torch.cuda.max_memory_reserved() - cuda_reserved,
                }
            )
    # Validate state again after all measurements without retaining a full-model copy.
    assert _fingerprints(engine.module) == original_fingerprints
    assert tuple(id(param) for param in engine.module.parameters()) == original_identities
    for worker in workers.values():
        assert not worker.cpu_saved_models
    result = {
        "manifest_sha256": manifest_sha256,
        "rank": rank,
        "local_parameter_bytes": sizes,
        "parameter_contract": parameter_contract,
        "timings": rows,
        "memory": memory,
    }
    _write(args.output / f"result-rank{rank}.json", result)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, result)
    if rank == 0:
        summary = summarize_pairs(
            [row for result in gathered for row in result["timings"]], dist.get_world_size(), args.repeats
        )
        summary.update(
            {
                "scope": "snapshot bookkeeping only; no forward/backward/rollout/reward timing",
                "model_depth": args.layers,
                "offload": args.offload,
                "lora_rank": args.lora_rank,
                "lora_alpha": 2 * args.lora_rank,
                "target_modules": TARGETS,
                "base_dtype": "bfloat16",
                "lora_dtype": "float32",
                "seed": args.seed,
                "warmup_pairs": args.warmup,
                "timed_pairs": args.repeats,
                "sync_steps": args.sync_steps,
                "world_size": dist.get_world_size(),
                "aggregate_parameter_bytes": {
                    key: sum(item["local_parameter_bytes"][key] for item in gathered) for key in sizes
                },
                "training_cycle_fraction": None,
                "memory_measurement": "separate sampled pass, not exact RSS peak",
                "runtime": {
                    "torch": torch.__version__,
                    "diffusers": provenance["versions"]["diffusers"],
                    "peft": provenance["versions"]["peft"],
                    "device": torch.cuda.get_device_name(),
                    "python": platform.python_version(),
                },
                "source_revision": args.source_revision,
                "checkpoint_revision": args.checkpoint_revision,
                "manifest_sha256": manifest_sha256,
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "checkpoint_manifest_sha256": verified["checkpoint_manifest_sha256"],
                "checkpoint_metadata_sha256": verified["metadata_sha256"],
                "parameter_contract_by_rank": [item["parameter_contract"] for item in gathered],
                **provenance,
            }
        )
        _write(args.output / "summary.json", summary)
    print(f"BENCH_COMPLETE rank={rank}", flush=True)


def main():
    """Run one pretrained depth/offload stratum in an isolated distributed process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True, help="Unique immutable identifier shared by both ranks")
    parser.add_argument("--layers", type=int, default=60)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--sync-steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()
    if (
        args.layers not in (10, 30, 60)
        or args.sync_steps != 3
        or args.warmup != 2
        or args.repeats != 10
        or args.lora_rank != 64
        or args.seed != 20260911
    ):
        parser.error("Frozen matrix requires depth10/30/60, sync3, warmup2, repeats10, rank64 and seed20260911")
    if not all(re.fullmatch(r"[a-f0-9]{40}", value) for value in (args.source_revision, args.checkpoint_revision)):
        parser.error("Source and checkpoint revisions must be full commit IDs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", args.run_id):
        parser.error("run-id must be 8-128 portable identifier characters")
    rank = int(os.environ["RANK"])
    manifest_sha256, provenance, verified = _preflight(args, rank)
    if not 0 < args.layers <= verified["transformer_config"]["num_layers"]:
        parser.error("layers must be within the pretrained depth")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 2:
            raise ValueError("Frozen benchmark matrix requires exactly two GPU ranks")
        if dist.get_rank() == 0:
            assert rank == 0
        with torch.no_grad():
            _run(args, manifest_sha256, provenance, verified)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
