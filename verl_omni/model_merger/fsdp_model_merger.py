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
"""Publish complete, audited Diffusers transformers with unchanged base components."""

import inspect
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor, Replicate, Shard

from .architectures import _PIPELINES, _TRANSFORMERS
from .base_model_merger import BaseModelMerger, MergeResult
from .utils import (
    MANIFEST_NAME,
    fingerprint,
    inventory,
    publication_directory,
    read_json,
    tree_files,
    weight_files,
    write_json,
    write_weights,
)

_H3_NATIVE_CLASS = "MiniMaxH3DiTModel"
_H3_CONFIG_RENAMES = {
    "num_refiner_layers": "token_refiner_num_layers",
    "ffn_dim": "ffn_hidden_size",
    "in_channels": "latents_dim",
    "audio_in_channels": "audio_latents_dim",
    "freq_dim": "timestep_input_dim",
    "time_embed_hidden_dim": "time_embed_hidden_size",
    "rope_freq_dim": "rope_inv_freq_len",
}
_H3_SHARED_CONFIG_FIELDS = (
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "attention_head_dim",
    "patch_size",
    "text_dim",
    "time_embed_dim",
    "norm_eps",
    "qk_norm_eps",
    "final_norm_eps",
)
_BOOGU_MODULE_ALIASES = {
    "transformer_boogu": "boogu.models.transformers.transformer_boogu",
    "scheduling_flow_match_euler_discrete_time_shifting": (
        "boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting"
    ),
}
_H3_TOPLEVEL_RENAMES = (
    ("audio_proj_in", "audio_patch_proj"),
    ("audio_proj_out", "final_layer.audio_out"),
    ("proj_in", "video_patch_proj"),
    ("proj_out", "final_layer.video_out"),
    ("context_embedder", "condition_proj"),
    ("time_embedder.linear_1", "time_embedder.proj_in"),
    ("time_embedder.linear_2", "time_embedder.proj_out"),
    ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ("norm_out.norm", "final_layer.norm"),
)


def _transformer_class(architecture: str):
    if architecture not in _TRANSFORMERS:
        raise ValueError(f"Unsupported publishing architecture: {architecture}")
    module = "boogu.models.transformers.transformer_boogu" if architecture == "BooguImagePipeline" else "diffusers"
    from importlib import import_module

    return getattr(import_module(module), _TRANSFORMERS[architecture])


def _pipeline_class(architecture: str):
    if architecture == "BooguImagePipeline":
        from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline

        return BooguImagePipeline
    import diffusers

    return getattr(diffusers, architecture)


def _resolve_architecture(index: dict | None, config: dict) -> str:
    """Infer the architecture from the base artifact without a user-selected model name."""
    if index is not None:
        architecture = index.get("_class_name")
    else:
        model_class = config.get("_class_name")
        # Qwen Image and Qwen Image Edit share one standalone transformer class;
        # the component artifact does not need to guess which parent pipeline it came from.
        architecture = next((key for key, value in _TRANSFORMERS.items() if value == model_class), None)
    if architecture not in _TRANSFORMERS:
        raise ValueError(f"Unsupported publishing architecture: {architecture}")
    expected = (
        _H3_NATIVE_CLASS if index is not None and architecture == "MiniMaxH3Pipeline" else _TRANSFORMERS[architecture]
    )
    if config.get("_class_name") != expected:
        raise ValueError(f"Expected {expected} for {architecture}, got {config.get('_class_name')}")
    return architecture


def _h3_native_name(name: str) -> str:
    """Map one unfused Diffusers H3 parameter name to its native equivalent."""
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    name = name.replace("transformer_blocks.", "blocks.")
    name = name.replace(".attn.norm_q.", ".attn.q_norm.")
    name = name.replace(".attn.norm_k.", ".attn.k_norm.")
    name = name.replace(".attn.to_out.0.", ".attn.out_proj.")
    name = name.replace(".ff.net.2.", ".mlp.fc2.")
    for source, target in _H3_TOPLEVEL_RENAMES:
        if name.startswith(source + "."):
            return target + name[len(source) :]
    return name


def model_rank_files(root: Path) -> list[Path]:
    """Require exactly one model shard per declared rank, ignoring optimizer state."""
    metadata = read_json(root / "fsdp_config.json")
    world_size = metadata.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("fsdp_config.json must declare a positive integer world_size")
    if type(metadata.get("FSDP_version")) is not int or metadata["FSDP_version"] != 2:
        raise ValueError("Only FSDP2 checkpoints are supported")
    expected = [root / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    if set(root.glob("model_world_size_*_rank_*.pt")) != set(expected) or not all(p.is_file() for p in expected):
        raise ValueError("Missing or unexpected model rank files")
    return expected


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    local = value.to_local() if isinstance(value, DTensor) else value
    if type(local) is not torch.Tensor:
        raise ValueError("Only plain tensors and one-dimensional DTensors are supported; ShardedTensor is unsupported")
    if local.device.type != "cpu" or local.layout != torch.strided or local.is_quantized or local.is_complex():
        raise ValueError("Expected a dense, real, non-quantized CPU tensor")
    if local.is_floating_point() and not torch.isfinite(local).all():
        raise ValueError("Checkpoint contains non-finite weights")
    return local


def reconstruct_tensor(values: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    """Reconstruct a tensor and verify exact pre-cast round-trip to every local shard."""
    local = [_local_tensor(value) for value in values]
    if len({tensor.dtype for tensor in local}) != 1:
        raise ValueError("Dtype disagreement across ranks")
    distributed = [isinstance(value, DTensor) for value in values]
    if any(distributed) and not all(distributed):
        raise ValueError("Mixed DTensor/plain representation across ranks")
    if not any(distributed):
        if any(tuple(tensor.shape) != shape for tensor in local):
            raise ValueError("Plain tensors must have the full schema shape; ambiguous plain sharding is unsupported")
        if any(not torch.equal(local[0], tensor) for tensor in local[1:]):
            raise ValueError("Plain tensor replicas disagree")
        return local[0].clone().contiguous()

    first = values[0]
    mesh = first.device_mesh.mesh
    placements = first.placements
    if mesh.ndim != 1 or len(placements) != 1 or sorted(mesh.tolist()) != list(range(len(values))):
        raise ValueError("Only a one-dimensional FSDP mesh covering every rank is supported")
    if first.device_mesh.mesh_dim_names and "tp" in first.device_mesh.mesh_dim_names:
        raise ValueError("Tensor-parallel checkpoints are unsupported")
    placement = placements[0]
    if type(placement) not in (Shard, Replicate):
        raise ValueError(f"Unsupported DTensor placement: {placement}")
    for value in values:
        if (
            tuple(value.shape) != shape
            or value.dtype != local[0].dtype
            or value.placements != placements
            or not torch.equal(value.device_mesh.mesh, mesh)
            or value.device_mesh.mesh_dim_names != first.device_mesh.mesh_dim_names
            or value.device_mesh.device_type != first.device_mesh.device_type
            or value.stride() != first.stride()
        ):
            raise ValueError("DTensor shape, mesh or placement disagreement")

    if isinstance(placement, Replicate):
        if any(tuple(tensor.shape) != shape or not torch.equal(local[0], tensor) for tensor in local):
            raise ValueError("DTensor replicas disagree with shape or values")
        return local[0].clone().contiguous()

    dim = placement.dim
    if not 0 <= dim < len(shape):
        raise ValueError("Invalid sharding dimension")
    ordered = [local[rank] for rank in mesh.tolist()]
    chunk = (shape[dim] + len(values) - 1) // len(values)
    extents = []
    for coordinate, tensor in enumerate(ordered):
        expected = list(shape)
        expected[dim] = max(0, min(chunk, shape[dim] - coordinate * chunk))
        if tuple(tensor.shape) != tuple(expected):
            raise ValueError("Invalid local shard extent (including uneven/empty shards)")
        extents.append(expected[dim])
    merged = torch.cat(ordered, dim=dim).contiguous()
    offset = 0
    for tensor, extent in zip(ordered, extents, strict=True):
        if not torch.equal(merged.narrow(dim, offset, extent), tensor):
            raise ValueError("Source shard round-trip failed")
        offset += extent
    return merged


def _canonical_config_value(value):
    """Normalize JSON-equivalent containers before comparing model behavior."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, _canonical_config_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_canonical_config_value(item) for item in value)
    return value


def _portable_config(config: dict) -> dict:
    """Remove only location metadata; never rewrite behavior-affecting values."""
    result = {}
    for key, value in config.items():
        if key in {"_name_or_path", "name_or_path"}:
            continue
        result[key] = _portable_config(value) if isinstance(value, dict) else value
    return result


def _transformer_schema(base: Path, source_config_path: Path, architecture: str):
    cls = _transformer_class(architecture)
    base_config = read_json(base / "config.json")
    source_config = read_json(source_config_path)
    fields = set(inspect.signature(cls.__init__).parameters) - {"self"}
    for config in (base_config, source_config):
        if config.get("_class_name") != cls.__name__:
            raise ValueError(f"Expected {cls.__name__} config")
        if any(key not in fields and not key.startswith("_") for key in config):
            raise ValueError("Unrecognized transformer config fields")
    with torch.device("meta"):
        base_module = cls.from_config(base_config)
        source_module = cls.from_config(source_config)
    if any(
        _canonical_config_value(base_module.config[key]) != _canonical_config_value(source_module.config[key])
        for key in fields
    ):
        raise ValueError("Checkpoint/base transformer configuration mismatch")
    shapes = {key: tuple(value.shape) for key, value in base_module.state_dict().items()}
    keep_fp32 = tuple(getattr(base_module, "_keep_in_fp32_modules", None) or ())
    del base_module, source_module
    mapping = weight_files(base)
    if set(mapping) != set(shapes):
        raise ValueError("Base transformer weights do not match its config schema")
    for path in set(mapping.values()):
        with safe_open(path, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                if tuple(archive.get_slice(key).get_shape()) != shapes[key]:
                    raise ValueError(f"Base tensor shape mismatch: {key}")
    return shapes, keep_fp32


def _weight_shapes(mapping: Mapping[str, Path]) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for path in sorted(set(mapping.values())):
        with safe_open(path, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                shapes[key] = tuple(archive.get_slice(key).get_shape())
    return shapes


def _read_weight(mapping: Mapping[str, Path], key: str) -> torch.Tensor:
    with safe_open(mapping[key], framework="pt", device="cpu") as archive:
        return archive.get_tensor(key)


def _h3_source_schema(source_config_path: Path, native_root: Path):
    """Validate equivalent Diffusers/native H3 configs and derive both tensor schemas."""
    cls = _transformer_class("MiniMaxH3Pipeline")
    source_config = read_json(source_config_path)
    native_config = read_json(native_root / "config.json")
    if source_config.get("_class_name") != cls.__name__ or native_config.get("_class_name") != _H3_NATIVE_CLASS:
        raise ValueError("MiniMax H3 requires a Diffusers actor config and native MiniMaxH3DiTModel base")
    fields = set(inspect.signature(cls.__init__).parameters) - {"self"}
    if any(key not in fields and not key.startswith("_") for key in source_config):
        raise ValueError("Unrecognized MiniMax H3 actor config fields")
    with torch.device("meta"):
        module = cls.from_config(source_config)
    source_shapes = {key: tuple(value.shape) for key, value in module.state_dict().items()}
    keep_fp32 = tuple(getattr(module, "_keep_in_fp32_modules", None) or ())
    del module

    for name in _H3_SHARED_CONFIG_FIELDS:
        if _canonical_config_value(native_config.get(name)) != _canonical_config_value(source_config.get(name)):
            raise ValueError(f"MiniMax H3 native/base config mismatch: {name}")
    for source_name, native_name in _H3_CONFIG_RENAMES.items():
        if _canonical_config_value(native_config.get(native_name)) != _canonical_config_value(
            source_config.get(source_name)
        ):
            raise ValueError(f"MiniMax H3 native/base config mismatch: {native_name}")
    hidden_size = source_config.get("hidden_size")
    if native_config.get("adaln_out_features") != 18 * hidden_size:
        raise ValueError("MiniMax H3 native adaln_out_features mismatch")
    if native_config.get("final_adaln_out_features") != 2 * hidden_size:
        raise ValueError("MiniMax H3 native final_adaln_out_features mismatch")

    native_mapping = weight_files(native_root, "model.safetensors")
    native_shapes = _weight_shapes(native_mapping)
    return source_shapes, keep_fp32, source_config, native_mapping, native_shapes


def _h3_conversion_plan(source_shapes: Mapping[str, tuple[int, ...]]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Plan the complete Diffusers-to-native H3 tensor conversion."""
    plan: dict[str, tuple[str, tuple[str, ...]]] = {}
    qkv: dict[str, dict[str, str]] = {}
    for name in source_shapes:
        if name.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight")):
            block, projection = name.rsplit(".attn.to_", 1)
            target = f"{_h3_native_name(block)}.attn.qkv_proj.weight"
            qkv.setdefault(target, {})[projection[0]] = name
            continue
        if name.endswith(".ff.net.0.proj.weight"):
            target = _h3_native_name(name).replace(".ff.net.0.proj.", ".mlp.fc1.")
            kind = "geglu"
        else:
            target = _h3_native_name(name)
            kind = "identity"
        if target in plan:
            raise ValueError(f"Duplicate MiniMax H3 native target: {target}")
        plan[target] = (kind, (name,))
    for target, parts in qkv.items():
        if set(parts) != {"q", "k", "v"} or target in plan:
            raise ValueError(f"Incomplete MiniMax H3 QKV group: {target}")
        plan[target] = ("qkv", tuple(parts[name] for name in ("q", "k", "v")))
    return plan


def _check_pipeline(base: Path, architecture: str, component: str) -> None:
    import diffusers
    import transformers
    from diffusers import ModelMixin, SchedulerMixin
    from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

    if architecture not in _PIPELINES or architecture == "MiniMaxH3Pipeline":
        raise ValueError(f"Unsupported standard Diffusers pipeline: {architecture}")
    config = read_json(base / "model_index.json")
    cls = _pipeline_class(architecture)
    signature = inspect.signature(cls.__init__).parameters
    types = cls._get_signature_types()
    optional = cls._optional_components
    if config.get("_module") or any(not key.startswith("_") and key not in signature for key in config):
        raise ValueError("Unknown pipeline components or custom pipeline module")
    for key, parameter in signature.items():
        if key == "self":
            continue
        if key not in config:
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(f"Missing pipeline component: {key}")
            continue
        value = config[key]
        expected = types[key]
        # Pipeline options (e.g. Wan boundary_ratio) are not component descriptors.
        if all(kind in (bool, int, float, str, type(None)) for kind in expected):
            if not any(type(value) is kind for kind in expected):
                raise ValueError(f"Invalid pipeline option: {key}")
            continue
        if value == [None, None] and key in optional and key != component:
            continue
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, str) for v in value):
            raise ValueError(f"Invalid pipeline component: {key}")
        library, name = value
        modules = {"diffusers": diffusers, "transformers": transformers, "ltx2": diffusers.pipelines.ltx2}
        module = modules.get(library)
        if module is None and architecture == "BooguImagePipeline":
            from importlib import import_module

            module_name = _BOOGU_MODULE_ALIASES.get(library, library if library.startswith("boogu.") else None)
            if module_name is not None:
                module = import_module(module_name)
        if module is None or not isinstance(actual := getattr(module, name, None), type):
            raise ValueError(f"Unsupported component class: {key}")
        tokenizer_match = issubclass(actual, PreTrainedTokenizerBase) and (
            AutoTokenizer in expected
            or any(name.removesuffix("Fast") == kind.__name__.removesuffix("Fast") for kind in expected)
        )
        boogu_mllm_match = (
            architecture == "BooguImagePipeline"
            and key == "mllm"
            and name in {"Qwen3VLModel", "Qwen3VLForConditionalGeneration"}
        )
        if not issubclass(actual, expected) and not tokenizer_match and not boogu_mllm_match:
            raise ValueError(f"Component class conflicts with pipeline: {key}")
        root = base / key
        if issubclass(actual, ModelMixin | PreTrainedModel):
            model_config = read_json(root / "config.json")
            if model_config.get("quantization_config") or model_config.get("auto_map"):
                raise ValueError("Quantized/custom-code components are unsupported")
            weight_files(root, "model.safetensors" if issubclass(actual, PreTrainedModel) else None)
        elif issubclass(actual, SchedulerMixin):
            if read_json(root / "scheduler_config.json").get("_class_name") != name:
                raise ValueError("Scheduler config conflicts with pipeline index")
        elif issubclass(actual, PreTrainedTokenizerBase):
            read_json(root / "tokenizer_config.json")
            if not any(
                (root / asset).is_file() for asset in ("tokenizer.json", "spiece.model", "tokenizer.model")
            ) and not all((root / asset).is_file() for asset in ("vocab.json", "merges.txt")):
                raise ValueError("Missing tokenizer vocabulary assets")
        else:
            # Processor / image processor assets are copied unchanged, never reconstructed.
            assets = [root / name for name in ("config.json", "processor_config.json", "preprocessor_config.json")]
            if not any(path.is_file() for path in assets):
                raise ValueError(f"Missing processor config: {key}")
            for path in assets:
                if path.is_file():
                    read_json(path)
    if config.get(component) in (None, [None, None]):
        raise ValueError(f"Selected trained component is absent: {component}")
    if architecture == "WanPipeline" and config.get("transformer_2") not in (None, [None, None]):
        ratio = config.get("boundary_ratio")
        if not isinstance(ratio, float | int) or isinstance(ratio, bool) or not 0 < ratio < 1:
            raise ValueError("Dual-transformer Wan requires a boundary_ratio in (0, 1)")


def _check_h3_pipeline(base: Path) -> None:
    """Validate the fixed native MiniMax H3 package boundary without importing GPU runtime code."""
    index = read_json(base / "model_index.json")
    expected = {
        "transformer": "MiniMaxH3DiTModel",
        "text_encoder": "MiniMaxH3Qwen3VLHFEncoder",
        "video_vae": "MiniMaxH3VideoVAE",
        "audio_vae": "MiniMaxH3AudioVAE",
        "processor": "Qwen3VLProcessor",
    }
    for component, class_name in expected.items():
        value = index.get(component)
        if not isinstance(value, list) or len(value) != 2 or value[1] != class_name:
            raise ValueError(f"Invalid MiniMax H3 component declaration: {component}")
        root = base / component
        if root.is_symlink() or not root.is_dir() or not (root / "config.json").is_file():
            raise ValueError(f"Missing MiniMax H3 component: {component}")
        read_json(root / "config.json")
    tokenizer = index.get("tokenizer")
    if (
        not isinstance(tokenizer, list)
        or len(tokenizer) != 2
        or tokenizer[1]
        not in {
            "Qwen2Tokenizer",
            "Qwen2TokenizerFast",
        }
    ):
        raise ValueError("Invalid MiniMax H3 tokenizer declaration")
    if not (base / "tokenizer/tokenizer_config.json").is_file():
        raise ValueError("Missing MiniMax H3 tokenizer assets")
    if index.get("scheduler") not in (None, [None, None]):
        raise ValueError("MiniMax H3 native pipeline must not declare an external scheduler")
    if not isinstance(index.get("_minimax_h3"), dict):
        raise ValueError("Missing MiniMax H3 release metadata")
    weight_files(base / "transformer", "model.safetensors")


class FSDPModelMerger(BaseModelMerger):
    """Recover FSDP checkpoints and publish canonical Diffusers or native H3 artifacts."""

    @contextmanager
    def _rank_states(self, expected_shapes: Mapping[str, tuple[int, ...]]):
        states = []
        try:
            for path in model_rank_files(Path(self.config.local_dir)):
                # Pickle is explicitly trusted by ModelMergerConfig. Do not silently fall back
                # to eager loading if mmap or this installed torch's decoder is unsupported.
                state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
                    raise ValueError("Expected a flat state dict in each model rank file")
                if any("lora_" in key or ".base_layer." in key for key in state):
                    raise ValueError("Adapter-bearing checkpoints require the future LoRA export mode")
                if set(state) != set(expected_shapes):
                    raise ValueError(
                        f"Incomplete transformer state: missing={sorted(set(expected_shapes) - set(state))[:8]}, "
                        f"unexpected={sorted(set(state) - set(expected_shapes))[:8]}"
                    )
                states.append(state)
            yield states
        finally:
            states.clear()

    @staticmethod
    def _merged_tensor(states, key: str, shape: tuple[int, ...]) -> torch.Tensor:
        try:
            return reconstruct_tensor([state[key] for state in states], shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot reconstruct {key}: {exc}") from exc

    def iter_merged_weights(self, expected_shapes: Mapping[str, tuple[int, ...]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Mmap rank files and yield complete schema-checked weights without initializing distributed."""
        with self._rank_states(expected_shapes) as states:
            for key, shape in sorted(expected_shapes.items()):
                yield key, self._merged_tensor(states, key, shape)

    def iter_h3_native_weights(
        self,
        source_shapes: Mapping[str, tuple[int, ...]],
        source_config: Mapping,
        native_mapping: Mapping[str, Path],
        native_shapes: Mapping[str, tuple[int, ...]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Convert complete Diffusers H3 tensors to the native fused checkpoint layout."""
        plan = _h3_conversion_plan(source_shapes)
        if set(plan) | {"rope.inv_freq"} != set(native_shapes):
            raise ValueError(
                "MiniMax H3 conversion does not cover the native schema: "
                f"missing={sorted(set(native_shapes) - set(plan) - {'rope.inv_freq'})[:8]}, "
                f"unexpected={sorted(set(plan) - set(native_shapes))[:8]}"
            )
        heads = int(source_config["num_attention_heads"])
        head_dim = int(source_config["attention_head_dim"])
        ff_half = int(source_config["ffn_dim"])
        rope_len = int(source_config["rope_freq_dim"])
        rope_theta = float(source_config.get("rope_theta", 10000.0))
        with self._rank_states(source_shapes) as states:
            for target in sorted(native_shapes):
                if target == "rope.inv_freq":
                    value = rope_theta ** (-(torch.arange(0, 2 * rope_len, 2, dtype=torch.float32) / (2 * rope_len)))
                    if not torch.equal(value, _read_weight(native_mapping, target)):
                        raise ValueError("MiniMax H3 base rope.inv_freq conflicts with the actor config")
                else:
                    kind, names = plan[target]
                    values = [self._merged_tensor(states, name, source_shapes[name]) for name in names]
                    if kind == "qkv":
                        if any(value.shape[0] != heads * head_dim for value in values):
                            raise ValueError(f"MiniMax H3 QKV shape mismatch: {target}")
                        value = torch.stack([tensor.reshape(heads, head_dim, -1) for tensor in values], dim=1).reshape(
                            heads * 3 * head_dim, -1
                        )
                    elif kind == "geglu":
                        if values[0].shape[0] != 2 * ff_half:
                            raise ValueError(f"MiniMax H3 GEGLU shape mismatch: {target}")
                        up, gate = values[0].split(ff_half, dim=0)
                        value = torch.cat([gate, up], dim=0)
                    else:
                        value = values[0]
                value = value.contiguous()
                if tuple(value.shape) != native_shapes[target]:
                    raise ValueError(f"MiniMax H3 native tensor shape mismatch: {target}")
                yield target, value

    def merge_and_save(self) -> MergeResult | dict:
        if self.config.operation == "test":
            from .output_validation import validate_artifact

            if not self.config.test_hf_dir:
                raise ValueError("test operation requires test_hf_dir")
            return validate_artifact(self.config.test_hf_dir)
        if not self.config.local_dir or not self.config.base_model or not self.config.target_dir:
            raise ValueError("merge operation requires local_dir, base_model and target_dir")
        if not self.config.hf_model_config_path:
            raise ValueError("merge operation requires hf_model_config_path")
        source = Path(self.config.local_dir).resolve(strict=True)
        base = Path(self.config.base_model).resolve(strict=True)
        source_config_path = Path(self.config.hf_model_config_path).resolve(strict=True) / "config.json"
        if not source_config_path.is_file():
            raise FileNotFoundError(source_config_path)
        raw_target = Path(self.config.target_dir)
        if os.path.lexists(raw_target):
            raise FileExistsError(raw_target)
        target = raw_target.resolve()
        if not target.parent.is_dir():
            raise ValueError("Output parent directory must already exist")
        roots = (source, base, target)
        for i, left in enumerate(roots):
            for right in roots[i + 1 :]:
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError("Source, base and target directories must not overlap")
        config_root = source_config_path.parent
        if source_config_path.is_relative_to(base):
            raise ValueError("Actor hf_model_config_path must not come from the base model")
        if target.is_relative_to(config_root) or config_root.is_relative_to(target):
            raise ValueError("Actor config and target directories must not overlap")
        if (source / "merge_source.json").exists():
            raise ValueError("Save-time merge_source schemas are not supported by this initial exporter")
        if (source / "lora_train_meta.json").exists():
            raise ValueError("LoRA checkpoint metadata requires the future adapter export mode")

        source_files = model_rank_files(source) + [source / "fsdp_config.json"]
        index_path = base / "model_index.json"
        index = read_json(index_path) if index_path.is_file() else None
        pipeline_output = self.config.output_format == "pipeline"
        if pipeline_output and index is None:
            raise ValueError("Pipeline export requires a complete base pipeline; use output_format=transformer")
        component = "transformer"
        model_root = base / component if index is not None else base
        if model_root.is_symlink():
            raise ValueError("Component directory symlinks are unsupported")
        component_config = read_json(model_root / "config.json")
        architecture = _resolve_architecture(index, component_config)
        native_h3 = pipeline_output and architecture == "MiniMaxH3Pipeline"
        if not pipeline_output and architecture == "MiniMaxH3Pipeline" and index is not None:
            raise ValueError(
                "Standalone MiniMax H3 export requires a canonical Diffusers transformer base, not a native pipeline"
            )

        if pipeline_output:
            base_files = tree_files(base)
            if any(path.suffix == ".py" for path in base_files) and not self.config.trust_remote_code:
                raise ValueError(
                    "Pipeline contains Python assets; pass --trust-remote-code for this audited local base"
                )
        else:
            base_files = sorted(set(weight_files(model_root).values()) | {model_root / "config.json"})
            if index is not None:
                base_files.append(index_path)
        source_inventory = inventory(source, source_files)
        source_config_inventory = inventory(source_config_path.parent, [source_config_path])
        base_inventory = inventory(base, base_files)

        native_weight_name = None
        if native_h3:
            _check_h3_pipeline(base)
            source_shapes, keep_fp32, source_config, native_mapping, native_shapes = _h3_source_schema(
                source_config_path, model_root
            )
            raw_weights = self.iter_h3_native_weights(source_shapes, source_config, native_mapping, native_shapes)
            trained_weight_files = set(native_mapping.values())
            native_weight_name = "model.safetensors"
        else:
            if pipeline_output:
                _check_pipeline(base, architecture, component)
            source_shapes, keep_fp32 = _transformer_schema(model_root, source_config_path, architecture)
            raw_weights = self.iter_merged_weights(source_shapes)
            trained_weight_files = set(weight_files(model_root).values())

        for suffix in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
            path = model_root / suffix
            if path.is_file():
                trained_weight_files.add(path)
        dtype = None if self.config.dtype == "preserve" else getattr(torch, self.config.dtype)

        def weights():
            for key, value in raw_weights:
                if dtype is not None and value.is_floating_point() and key != "rope.inv_freq":
                    effective_dtype = torch.float32 if any(part in key.split(".") for part in keep_fp32) else dtype
                    value = value.to(effective_dtype)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Requested cast produced non-finite weights: {key}")
                yield key, value

        with publication_directory(target) as staging:
            tensor_directory = component if pipeline_output else "."
            specs = write_weights(
                staging / tensor_directory,
                weights(),
                self.config.max_shard_size,
                weights_name=native_weight_name,
            )
            rewritten = []
            copy_files = base_files if pipeline_output else [model_root / "config.json"]
            copied = {}
            for path in copy_files:
                relative = path.relative_to(base) if pipeline_output else Path("config.json")
                if pipeline_output and path in trained_weight_files:
                    continue
                copied[relative.as_posix()] = base_inventory[path.relative_to(base).as_posix()]
                destination = staging / relative
                if destination.exists():
                    raise ValueError(f"Output path collision: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json" and ("config" in path.name or path.name == "model_index.json"):
                    original = read_json(path)
                    portable = _portable_config(original)
                    if portable != original:
                        write_json(destination, portable)
                        rewritten.append(relative.as_posix())
                        continue
                shutil.copyfile(path, destination)
            output_inventory = inventory(staging, tree_files(staging))
            for name, digest in copied.items():
                if name not in rewritten and output_inventory.get(name) != digest:
                    raise ValueError(f"Copied base asset verification failed: {name}")
            if (
                inventory(source, source_files) != source_inventory
                or inventory(source_config_path.parent, [source_config_path]) != source_config_inventory
                or inventory(base, tree_files(base) if pipeline_output else base_files) != base_inventory
                or read_json(model_root / "config.json") != component_config
                or (index is not None and read_json(index_path) != index)
            ):
                raise ValueError("Source/base inputs changed during export")
            if model_rank_files(source) != source_files[:-1]:
                raise ValueError("Model rank inventory changed during export")
            import diffusers

            artifact_type = (
                "minimax_h3_pipeline"
                if native_h3
                else ("diffusers_pipeline" if pipeline_output else "diffusers_transformer")
            )
            manifest = {
                "schema_version": 1,
                "artifact_type": artifact_type,
                "architecture": architecture,
                "backend": "fsdp",
                "trained_components": [component],
                "tensor_directory": tensor_directory,
                "dtype": self.config.dtype,
                "max_shard_size_bytes": self.config.max_shard_size,
                "source_fingerprint": fingerprint(
                    source_inventory | {"huggingface/config.json": next(iter(source_config_inventory.values()))}
                ),
                "base_fingerprint": fingerprint(base_inventory),
                "source_files": source_inventory,
                "source_model_config": {
                    "sha256": next(iter(source_config_inventory.values())),
                    "logical_path": "huggingface/config.json",
                },
                "base_files": base_inventory,
                "files": output_inventory,
                "tensors": specs,
                "config_transform": {"id": "remove_location_metadata_v1", "files": rewritten},
                "producer": {"torch": torch.__version__, "diffusers": diffusers.__version__, "training": "unknown"},
                "verification": {
                    "structure": "passed",
                    "source_round_trip": "passed",
                    "artifact_round_trip": "passed",
                    "integrity": "passed",
                    "runtime": "not_run",
                },
            }
            write_json(staging / MANIFEST_NAME, manifest)
            from .output_validation import validate_artifact

            validate_artifact(staging)
        result = MergeResult(target, target / MANIFEST_NAME)
        self.upload_to_huggingface(result.output_dir)
        return result
