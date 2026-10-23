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

"""Preserve input row identity when returning one optimizer update's outputs."""

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

_ROW_KEY = "__omni_training_output_row__"


def prepare_single_update_batch(
    data: TensorDict, *, mini_batch_size: int, epochs: int, seed: int, dataloader_kwargs: dict
) -> tuple[TensorDict, torch.Tensor]:
    """Validate a complete local update before any model or optimizer work.

    Outputs from repeated epochs or several optimizer updates cannot share one
    pre-update policy. This opt-in return contract therefore accepts exactly
    one complete local batch, independently of the data-parallel shard size.
    A shuffled permutation is allowed; dropped or repeated rows are not.
    """
    size = data.batch_size[0]
    if size <= 0 or mini_batch_size != size or epochs != 1:
        raise ValueError("Returning model outputs requires one complete local batch and one optimizer update")
    if _ROW_KEY in data:
        raise ValueError(f"Reserved training output key already present: {_ROW_KEY}")

    view = data.clone(recurse=False)
    view[_ROW_KEY] = torch.arange(size, dtype=torch.long)
    batches = list(
        tu.make_iterator(view, mini_batch_size=size, epochs=1, seed=seed, dataloader_kwargs=dataloader_kwargs)
    )
    if len(batches) != 1:
        raise ValueError("The training iterator must return exactly one complete batch")
    batch = batches[0]
    order = batch.pop(_ROW_KEY).cpu()
    if order.dtype != torch.long or order.shape != (size,) or not torch.equal(order.sort().values, torch.arange(size)):
        raise ValueError("The training iterator dropped or duplicated input rows")
    return batch, order


def restore_single_update_output(output: TensorDict, order: torch.Tensor) -> TensorDict:
    """Detach outputs and restore the original local input order, including jagged tensors."""
    size = len(order)
    if (
        order.dtype != torch.long
        or order.ndim != 1
        or output.batch_size != (size,)
        or not torch.equal(order.sort().values, torch.arange(size))
    ):
        raise ValueError("Training output rows do not match the original batch")
    return tu.index_select_tensor_dict(output.detach(), order.argsort())


def detach_microbatch_outputs(result: tuple) -> tuple:
    """Release diagnostic graphs while preserving the exact backward loss.

    Megatron retains each microbatch's reduced outputs until batch assembly.
    Unused diagnostics can otherwise keep full-vocabulary activations alive
    across all microbatches, even after the policy loss has been backpropagated.
    Both ordinary and per-token-loss schedule return layouts are preserved.
    """
    if len(result) not in (2, 3) or not isinstance(result[-1], dict):
        raise ValueError("Unexpected Megatron loss/output return layout")
    reduced = result[-1]
    if "model_output" not in reduced:
        return result
    detached = torch.utils._pytree.tree_map_only(torch.Tensor, torch.Tensor.detach, reduced["model_output"])
    return (*result[:-1], {**reduced, "model_output": detached})
