# Copyright 2026 verl-omni contributors
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

from collections.abc import Mapping

from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import KVBatchMeta


def get_training_output_metrics(output: TensorDict | KVBatchMeta, batch: KVBatchMeta) -> dict:
    """Read training metrics without changing the scalar-only collect semantics.

    Returning model outputs makes ``tqbridge`` write tensors to storage and
    return metadata. ND collection then keeps a list of source-rank metrics,
    whereas the scalar-only TensorDict route selects its first source. Worker
    metrics already include the engine's DP aggregation; averaging that list
    again would change the existing behavior. Reuse native scalar collection.
    """
    if not isinstance(output, KVBatchMeta):
        return output["metrics"]
    if output.partition_id != batch.partition_id or list(output.keys) != list(batch.keys):
        raise ValueError("Training output partition and ordered keys must match the input batch")
    metrics = output.extra_info.get("metrics")
    if isinstance(metrics, list | tuple):
        if not metrics or not all(isinstance(item, Mapping) for item in metrics):
            raise ValueError("Training output must include nonempty source-rank metrics mappings")
        scalar_outputs = [tu.get_tensordict({}, non_tensor_dict={"metrics": item}) for item in metrics]
        return tu.get(tu.concat_tensordict(scalar_outputs), "metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Training output is missing its metrics mapping")
    return metrics
