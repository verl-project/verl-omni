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
"""msgspec-safe wire encoding for the engine-side-merge CUDA-IPC payload.

The merged LoRA payload rides ``collective_rpc`` to the rollout engine
workers. Diffusion workers execute in the server process, but AR workers live
in separate processes, so the kwargs cross vLLM's msgspec serializer, which
accepts only primitives (plus tensors and ndarrays). The
``torch.multiprocessing.reductions.reduce_tensor`` args the engine rebuilds
every tensor from carry class objects (``type(tensor)``, ``type(storage)``),
``torch.dtype`` instances and tuple types — raw, they fail that RPC with
``Object of type <class 'torch._C._TensorMeta'> is not serializable``.

The encoding is position-preserving and generic over the args layout: every
element becomes either a pass-through primitive or a two-item ``[tag, value]``
marker list, so torch may reorder its reduce-args layout (the engine
overwrites the device index at a fixed position) without touching this module.
The same pass normalizes the PEFT config: PEFT resolves ``target_modules`` to a
``set``, which the engine's target matcher only accepts as ``list`` or ``str``
— a set silently degrades to match-everything and merged mode then rejects the
unquantized component root (``transformer``).
"""

from collections.abc import Mapping, Sequence

import torch

__all__ = [
    "IPC_WIRE_KEY",
    "IPC_WIRE_VERSION",
    "decode_ipc_handle_args",
    "encode_ipc_handle_args",
    "normalize_peft_config_for_wire",
]

# Payload key carrying the wire version. Popped by the worker extension before
# the dict reaches the engine's update_info dataclass, which rejects extras.
IPC_WIRE_KEY = "ipc_handles_wire"
IPC_WIRE_VERSION = "verl-omni-msgspec-v1"

_TYPE_TAG = "type"  # importable class object, e.g. torch.Tensor
_DTYPE_TAG = "dtype"  # torch.dtype instance
_TUPLE_TAG = "tuple"  # tuple (incl. torch.Size); msgspec arrays decode as lists

_PRIMITIVES = str | int | float | bool | bytes

# The only classes reduce_tensor args ever carry: type(tensor) and
# type(storage). An allowlist beats importlib resolution: it fails closed on
# anything else and never imports by payload-provided name.
_WIRE_TYPES = {
    "torch.Tensor": torch.Tensor,
    "torch.nn.parameter.Parameter": torch.nn.parameter.Parameter,
    "torch.storage.TypedStorage": torch.storage.TypedStorage,
    "torch.UntypedStorage": torch.UntypedStorage,
}


def encode_ipc_handle_args(args: Sequence) -> list:
    """Encode one ``reduce_tensor`` args tuple into msgspec primitives.

    Raises:
        TypeError: if an element is neither a primitive nor one of the tagged
            kinds — fail closed rather than let the worker RPC die mid-session.
    """
    return [_encode_element(element) for element in args]


def decode_ipc_handle_args(encoded: Sequence) -> tuple:
    """Rebuild the ``reduce_tensor`` args tuple the engine consumes."""
    return tuple(_decode_element(element) for element in encoded)


def _encode_element(element):
    if element is None or isinstance(element, _PRIMITIVES):
        return element
    if isinstance(element, torch.dtype):
        return [_DTYPE_TAG, str(element)]
    if isinstance(element, tuple):
        return [_TUPLE_TAG, [_encode_element(item) for item in element]]
    if isinstance(element, type):
        return [_TYPE_TAG, f"{element.__module__}.{element.__qualname__}"]
    raise TypeError(
        f"CUDA-IPC handle args contain {element!r} of type {type(element).__name__}, "
        "which has no msgspec-safe wire encoding for the engine-merge payload."
    )


def _decode_element(element):
    if isinstance(element, list) and len(element) == 2 and element[0] in (_TYPE_TAG, _DTYPE_TAG, _TUPLE_TAG):
        tag, value = element
        if tag == _DTYPE_TAG:
            module, _, name = value.partition(".")
            dtype = getattr(torch, name, None) if module == "torch" else None
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"engine-merge payload carries unknown dtype {value!r}")
            return dtype
        if tag == _TYPE_TAG:
            resolved = _WIRE_TYPES.get(value)
            if resolved is None:
                raise ValueError(
                    f"engine-merge payload carries type {value!r}, which is not one of "
                    f"the reduce_tensor arg classes {sorted(_WIRE_TYPES)}."
                )
            return resolved
        return tuple(_decode_element(item) for item in value)
    if element is None or isinstance(element, _PRIMITIVES):
        return element
    raise ValueError(
        f"engine-merge payload handle args contain {element!r} of type {type(element).__name__}, "
        "which the wire decoder does not produce; the payload is corrupt or misrouted."
    )


def normalize_peft_config_for_wire(peft_config: Mapping | None) -> dict | None:
    """Return a copy with every set/tuple turned into a list, sets sorted.

    The engine's ``PEFTHelper`` declares ``target_modules: list[str] | str``;
    PEFT's ``to_dict()`` ships a ``set``, which its matcher skips entirely
    (match-everything). Lists make the selection exact on both engines.
    """
    if peft_config is None:
        return None
    normalized = {}
    for key, value in peft_config.items():
        if isinstance(value, set | frozenset):
            normalized[key] = sorted(value)
        elif isinstance(value, tuple):
            normalized[key] = list(value)
        elif isinstance(value, Mapping):
            normalized[key] = normalize_peft_config_for_wire(value)
        else:
            normalized[key] = value
    return normalized
