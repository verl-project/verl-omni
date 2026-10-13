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
"""Phase-local dispatch across complete reward replicas."""

import asyncio

from verl.protocol import DataProto, pad_dataproto_to_divisor


async def dispatch_reward_groups(data: DataProto, worker_groups: dict, batch_sizes: dict[str, int]) -> dict[str, list]:
    """Score groups, restore sample order and drain accepted RPCs before returning.

    Groups absent from ``batch_sizes`` retain static padded splitting. Other
    groups dispatch at most one microbatch per replica, replenishing on completion.
    Failure or caller cancellation stops new dispatch, but does not cancel an RPC.
    """
    outputs = {}
    next_index = {}
    static_chunks = {}
    stopped = False
    for name, workers in worker_groups.items():
        if not workers:
            raise ValueError(f"Reward worker group {name!r} has no replicas")
        batch_size = batch_sizes.get(name)
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError(f"Reward worker group {name!r} dispatch_batch_size must be a positive integer")
        next_index[name] = 0
        if batch_size is None and len(data):
            padded, _ = pad_dataproto_to_divisor(data, len(workers))
            static_chunks[name] = padded.chunk(len(workers))
            outputs[name] = [None] * len(padded)
        else:
            outputs[name] = [None] * len(data)

    async def run_replica(name, worker, replica_index):
        nonlocal stopped
        try:
            while not stopped and next_index[name] < len(data):
                if name in static_chunks:
                    chunk = static_chunks[name][replica_index]
                    start = replica_index * len(chunk)
                else:
                    start = next_index[name]
                    stop = min(start + batch_sizes[name], len(data))
                    next_index[name] = stop
                    chunk = data[start:stop]
                result = await worker.compute_score_batch.remote(chunk)
                if len(result) != len(chunk):
                    raise ValueError(
                        f"Reward worker group {name!r} returned {len(result)} results for {len(chunk)} samples"
                    )
                outputs[name][start : start + len(chunk)] = result
                if name in static_chunks:
                    break
        except BaseException:
            stopped = True
            raise

    pending = asyncio.gather(
        *(
            run_replica(name, worker, index)
            for name, workers in worker_groups.items()
            for index, worker in enumerate(workers)
        ),
        return_exceptions=True,
    )
    cancellation = None
    while True:
        try:
            results = await asyncio.shield(pending)
            break
        except asyncio.CancelledError as exc:
            # Remote work can outlive a cancelled await; keep its ownership until completion.
            stopped = True
            cancellation = exc
    if cancellation is not None:
        raise cancellation
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return {name: values[: len(data)] for name, values in outputs.items()}
