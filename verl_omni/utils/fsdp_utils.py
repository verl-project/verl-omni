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
"""
FSDP utilities for verl-omni
"""

import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path

import peft
import torch
from peft.utils.save_and_load import get_peft_model_state_dict
from safetensors.torch import save_file
from verl.utils.fsdp_utils import collect_lora_params as _upstream_collect_lora_params
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.fsdp_utils import layered_summon_lora_params as _upstream_layered_summon_lora_params

__all__ = ["collect_lora_params", "export_fsdp_lora_adapter", "fsdp_summon_full_params"]


def _get_fsdp_module_cls():
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        from torch.distributed._composable.fsdp import FSDPModule
    return FSDPModule


def _iter_fsdp2_submodules(module):
    fsdp_module_cls = _get_fsdp_module_cls()
    for name, submodule in module.named_modules():
        if isinstance(submodule, fsdp_module_cls) and name != "":
            yield name, submodule


@contextmanager
def fsdp_summon_full_params(module, *, writeback: bool = False, with_grads: bool = False, recurse: bool = True):
    """Summon unsharded params for FSDP1/FSDP2, matching verl's fsdp_merge_unmerge pattern."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    version = fsdp_version(module)
    if version == 0:
        yield
        return
    if version == 1:
        with FSDP.summon_full_params(module, writeback=writeback, recurse=recurse, with_grads=with_grads):
            yield
        return

    submodules = list(_iter_fsdp2_submodules(module))
    if not submodules:
        yield
        return

    with ExitStack() as stack:
        for _, submodule in submodules:
            stack.enter_context(FSDP.summon_full_params(submodule, writeback=writeback, with_grads=with_grads))
        yield


def _param_to_cpu(param):
    if hasattr(param, "full_tensor"):
        return param.full_tensor().detach().cpu()
    return param.detach().cpu()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "_local_tensor"):
        tensor = tensor._local_tensor
    return tensor.detach().cpu().contiguous()


def _normalize_peft_config(config: dict) -> dict:
    for key in ("task_type", "peft_type"):
        if key in config and hasattr(config[key], "value"):
            config[key] = config[key].value
    if config.get("target_modules") is not None:
        config["target_modules"] = sorted(config["target_modules"])
    return config


def _discover_fsdp_rank_paths(input_dir: Path, world_size: int) -> list[Path]:
    rank_paths = [input_dir / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    missing = [str(path) for path in rank_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rank checkpoint(s): {missing}")
    return rank_paths


def _merge_fsdp_lora_tensors(rank_paths: list[Path]) -> tuple[OrderedDict[str, torch.Tensor], list[str]]:
    print(f"Loading rank 0/{len(rank_paths) - 1}: {rank_paths[0].name}")
    rank0_state = torch.load(rank_paths[0], map_location="cpu", weights_only=False, mmap=True)
    lora_keys = sorted(key for key in rank0_state.keys() if "lora_" in key)
    if not lora_keys:
        raise RuntimeError(f"No lora_ keys found in {rank_paths[0]}")

    print(f"Found {len(lora_keys)} LoRA tensors")
    lora_shards = {key: [_local_tensor(rank0_state[key])] for key in lora_keys}
    placements = {key: getattr(rank0_state[key], "placements", None) for key in lora_keys}
    del rank0_state

    for rank, rank_path in enumerate(rank_paths[1:], start=1):
        print(f"Loading rank {rank}/{len(rank_paths) - 1}: {rank_path.name}")
        rank_state = torch.load(rank_path, map_location="cpu", weights_only=False, mmap=True)
        for key in lora_keys:
            lora_shards[key].append(_local_tensor(rank_state[key]))
        del rank_state

    lora_params = OrderedDict()
    target_modules = set()
    for key in lora_keys:
        placement = placements[key]
        if placement and len(placement) == 1 and placement[0].is_shard():
            merged = torch.cat(lora_shards[key], dim=placement[0].dim).contiguous()
        else:
            merged = lora_shards[key][0].contiguous()

        module_key = key.rsplit(".lora_", maxsplit=1)[0]
        peft_key = "base_model.model." + key.replace(".default.weight", ".weight")
        lora_params[peft_key] = merged
        target_modules.add(module_key)

    return lora_params, sorted(target_modules)


def _build_peft_lora_config(meta: dict, target_modules: list[str], base_model_name_or_path: str | None) -> dict:
    peft_dict = {
        "r": int(meta["r"]),
        "lora_alpha": int(meta["lora_alpha"]),
        "target_modules": target_modules,
    }
    if meta.get("task_type") is not None:
        peft_dict["task_type"] = meta["task_type"]

    config = peft.LoraConfig(**peft_dict).to_dict()
    config = _normalize_peft_config(config)
    if base_model_name_or_path is not None:
        config["base_model_name_or_path"] = base_model_name_or_path
    return config


def export_fsdp_lora_adapter(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    base_model_name_or_path: str | None = None,
) -> dict:
    """Export PEFT LoRA adapter weights from a verl FSDP checkpoint directory.

    This helper is intended for FSDP checkpoints that contain LoRA weights
    inside sharded model state dicts. It reads only tensors whose names contain
    ``lora_``, merges their DTensor/local shards across ranks, and writes a
    PEFT-compatible ``adapter_config.json`` plus ``adapter_model.safetensors``.
    The full model is not instantiated.

    Args:
        input_dir: Directory containing ``fsdp_config.json``,
            ``lora_train_meta.json``, and
            ``model_world_size_<world_size>_rank_<rank>.pt`` files.
        output_dir: Directory to write the PEFT adapter files. Defaults to
            ``<input_dir>/lora_adapter``.
        base_model_name_or_path: Optional value to write into the PEFT
            ``adapter_config.json`` as ``base_model_name_or_path``.

    Returns:
        A summary dictionary with:
        ``output_dir`` (str), ``target_modules`` (list[str]),
        ``adapter_tensors`` (int), ``adapter_size_mib`` (float), and
        ``world_size`` (int).
    """
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else input_dir / "lora_adapter"

    fsdp_config = _load_json(input_dir / "fsdp_config.json")
    lora_meta = _load_json(input_dir / "lora_train_meta.json")
    world_size = int(fsdp_config["world_size"])
    rank_paths = _discover_fsdp_rank_paths(input_dir, world_size)

    print(f"Exporting LoRA adapter from {world_size} FSDP ranks")
    print(f"Input directory: {input_dir}")
    print(f"Output: {output_dir}")

    lora_params, target_modules = _merge_fsdp_lora_tensors(rank_paths)
    peft_config = _build_peft_lora_config(lora_meta, target_modules, base_model_name_or_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(peft_config, f, ensure_ascii=False, indent=4)
    save_file(lora_params, output_dir / "adapter_model.safetensors")

    adapter_size = (output_dir / "adapter_model.safetensors").stat().st_size / (1024**2)
    return {
        "output_dir": str(output_dir),
        "target_modules": target_modules,
        "adapter_tensors": len(lora_params),
        "adapter_size_mib": adapter_size,
        "world_size": world_size,
    }


def _peft_lora_params_to_cpu(peft_model, adapter_name: str) -> OrderedDict:
    lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
    return OrderedDict((name, _param_to_cpu(param)) for name, param in lora_params.items())


def _collect_base_weights_to_cpu(peft_model) -> OrderedDict:
    from verl.utils.device import get_device_name

    model = peft_model.base_model.model
    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
    model = model.to("cpu")
    lora_params = OrderedDict()
    for name, param in model.state_dict().items():
        if any(x in name for x in ["_flat_param", "lora_"]):
            continue
        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
        lora_params[name] = _param_to_cpu(param)
    model = model.to(orig_dev)
    return lora_params


def _collect_base_weights_from_state_dict(state_dict) -> OrderedDict:
    lora_params = OrderedDict()
    for name, param in state_dict.items():
        if any(x in name for x in ["_flat_param", "lora_"]):
            continue
        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
        lora_params[name] = _param_to_cpu(param)
    return lora_params


def _collect_lora_params_non_layered(module, peft_model, adapter_name: str, base_sync_done: bool) -> OrderedDict:
    """Collect LoRA/base params without layered summon for FSDP1/FSDP2/non-FSDP modules."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.device import get_torch_device

    version = fsdp_version(module)
    if version == 0:
        if base_sync_done:
            return _peft_lora_params_to_cpu(peft_model, adapter_name)
        return _collect_base_weights_to_cpu(peft_model)

    if version == 1:
        with FSDP.summon_full_params(module, writeback=False):
            if base_sync_done:
                lora_params = _peft_lora_params_to_cpu(peft_model, adapter_name)
            else:
                lora_params = _collect_base_weights_to_cpu(peft_model)
        get_torch_device().empty_cache()
        return lora_params

    lora_params = OrderedDict()
    for name, submodule in _iter_fsdp2_submodules(module):
        with FSDP.summon_full_params(submodule, writeback=False):
            if base_sync_done:
                sub_lora_params = get_peft_model_state_dict(
                    peft_model, state_dict=submodule.state_dict(), adapter_name=adapter_name
                )
                block_prefix = name.replace("_fsdp_wrapped_module.", "")
                for param_name, param in sub_lora_params.items():
                    full_name = f"{block_prefix}.{param_name}" if block_prefix else param_name
                    lora_params[full_name] = _param_to_cpu(param)
            else:
                lora_params.update(_collect_base_weights_from_state_dict(submodule.state_dict()))
    get_torch_device().empty_cache()
    return lora_params


def _collect_lora_params_with_adapter(
    module,
    layered_summon: bool,
    base_sync_done: bool,
    adapter_name: str,
    layered_summon_fn: Callable,
) -> OrderedDict:
    """Verl-style LoRA collection with explicit ``adapter_name`` for PEFT state."""
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        if layered_summon:
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            if layered_summon_fn is _upstream_layered_summon_lora_params:
                return layered_summon_fn(module)
            return layered_summon_fn(module, adapter_name=adapter_name)
        return _collect_lora_params_non_layered(module, peft_model, adapter_name, base_sync_done)

    if base_sync_done:
        return _peft_lora_params_to_cpu(peft_model, adapter_name)
    return _collect_base_weights_to_cpu(peft_model)


def _layered_summon_lora_params_diffusers(
    fsdp_module, adapter_name: str = "default", layer_prefixes: Sequence[str] = ("transformer_blocks.",)
) -> OrderedDict:
    """Layered LoRA param collection for diffusers transformer-block models.

    Args:
        fsdp_module: The FSDP-wrapped module.
        adapter_name: LoRA adapter name.
        layer_prefixes: FSDP layer name prefixes.  Defaults to
            ``["transformer_blocks."]``.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.device import get_torch_device

    def _prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = []
    for lp in layer_prefixes:
        # FSDP1
        prefix_list.append(f"_fsdp_wrapped_module.{lp}")
        # FSDP2
        prefix_list.append(lp)
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in _prefix_submodules(fsdp_module, prefix):
            block_prefix = name.replace("_fsdp_wrapped_module.", "")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(
                        peft_model, state_dict=submodule.state_dict(), adapter_name=adapter_name
                    )
                    sub_lora_params = {
                        f"{block_prefix}.{param_name}": _param_to_cpu(param)
                        for param_name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()
    return lora_params


def collect_lora_params(
    module,
    layered_summon: bool,
    base_sync_done: bool,
    is_diffusers: bool = False,
    adapter_name: str = "default",
    layer_prefixes: Sequence[str] = ("transformer_blocks.",),
) -> OrderedDict:
    """Collect LoRA or base parameters for weight sync to the rollout worker.

    Raises ``RuntimeError`` when no parameters were collected
    (e.g. mismatched ``layer_prefixes``).

    Args:
        module: The FSDP-wrapped or plain module.
        layered_summon: Summon one FSDP unit at a time instead of the full model.
        base_sync_done: If ``True``, collect only LoRA weights; else full base weights.
        is_diffusers: Use the diffusers-specific layered summon helper.
        adapter_name: LoRA adapter name (usually ``"default"``).
        layer_prefixes: FSDP layer name prefixes (``["transformer_blocks."]``
    """
    use_diffusers_layered = is_diffusers and layered_summon and fsdp_version(module) > 0
    if adapter_name == "default" and not use_diffusers_layered and fsdp_version(module) != 2:
        return _upstream_collect_lora_params(module, layered_summon=layered_summon, base_sync_done=base_sync_done)

    if is_diffusers:
        layered_summon_fn = partial(
            _layered_summon_lora_params_diffusers, adapter_name=adapter_name, layer_prefixes=layer_prefixes
        )
    else:
        layered_summon_fn = _upstream_layered_summon_lora_params
    lora_params = _collect_lora_params_with_adapter(
        module,
        layered_summon=layered_summon,
        base_sync_done=base_sync_done,
        adapter_name=adapter_name,
        layered_summon_fn=layered_summon_fn,
    )
    if not lora_params:
        raise RuntimeError(
            f"collect_lora_params collected 0 parameters with prefixes={layer_prefixes}. "
            "Check ``fsdp_layer_prefixes`` in the model config matches the model's "
            "FSDP layer naming (e.g. ``['transformer_blocks.']`` for DiT models)."
        )
    return lora_params
