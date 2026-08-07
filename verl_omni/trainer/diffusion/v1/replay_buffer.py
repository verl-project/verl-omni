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

import logging
import os
import time
from collections import Counter

import transfer_queue as tq
from transfer_queue import KVBatchMeta
from verl.trainer.ppo.v1.replay_buffer import VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS, ReplayBuffer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class DiffusionReplayBuffer(ReplayBuffer):
    """Replay buffer that can replace incomplete diffusion rollout groups.

    A prompt is marked ``failure`` after all of its ``rollout.n`` sessions
    settle and at least one session failed. When replacement is enabled, the
    whole prompt group is evicted before sampling and an exact number of fresh
    prompts is submitted. Validation keeps the upstream sampling behavior.
    """

    def __init__(
        self,
        *args,
        refill_fn=None,
        drop_incomplete_groups: bool = False,
        max_incomplete_group_refill_rounds: int = 3,
        train_batch_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.refill_fn = refill_fn
        self.drop_incomplete_groups = drop_incomplete_groups
        self.max_incomplete_group_refill_rounds = max_incomplete_group_refill_rounds
        self.train_batch_size = train_batch_size

        if self.drop_incomplete_groups and self.refill_fn is None:
            raise ValueError("drop_incomplete_groups requires refill_fn")
        if (
            isinstance(self.max_incomplete_group_refill_rounds, bool)
            or not isinstance(self.max_incomplete_group_refill_rounds, int)
            or (self.max_incomplete_group_refill_rounds <= 0)
        ):
            raise ValueError("max_incomplete_group_refill_rounds must be a positive integer")
        if self.drop_incomplete_groups and (
            isinstance(self.train_batch_size, bool)
            or not isinstance(self.train_batch_size, int)
            or self.train_batch_size <= 0
        ):
            raise ValueError("drop_incomplete_groups requires a positive train_batch_size")

    @staticmethod
    def _trajectory_uid(key: str) -> str:
        parts = key.rsplit("_", 2)
        return parts[0] if len(parts) == 3 else key

    def _evict_incomplete_groups(self, partition_id: str) -> tuple[int, dict]:
        failed_uids = set(self.failure_keys[partition_id])
        trajectory_keys = [key for key in self.partitions[partition_id] if self._trajectory_uid(key) in failed_uids]

        metadata = tq.kv_list() or {}
        partition_metadata = metadata.get(partition_id, {})
        reason_counts = Counter(partition_metadata.get(uid, {}).get("failure_reason", "unknown") for uid in failed_uids)

        tq.kv_clear(partition_id=partition_id, keys=[*failed_uids, *trajectory_keys])

        prefix = "training" if partition_id == "train" else "validation"
        metrics = {
            f"{prefix}/rollout_failure/evicted_groups": len(failed_uids),
            f"{prefix}/rollout_failure/evicted_trajectories": len(trajectory_keys),
        }
        metrics.update(
            {f"{prefix}/rollout_failure/reason/{reason}_groups": count for reason, count in reason_counts.items()}
        )
        return len(failed_uids), metrics

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample complete groups, replacing failed training groups within a bounded budget."""
        last_debug_time = time.time()
        refill_rounds = 0
        refilled_prompts = 0
        failure_metrics: Counter = Counter()

        while True:
            self._sync_metadata_from_transfer_queue()

            if partition_id == "train" and self.drop_incomplete_groups and self.failure_keys[partition_id]:
                if refill_rounds >= self.max_incomplete_group_refill_rounds:
                    raise RuntimeError(
                        "Exceeded max_incomplete_group_refill_rounds="
                        f"{self.max_incomplete_group_refill_rounds} while replacing failed rollout groups"
                    )

                num_failed = len(self.failure_keys[partition_id])
                refill_budget = self.train_batch_size * self.max_incomplete_group_refill_rounds
                if refilled_prompts + num_failed > refill_budget:
                    raise RuntimeError(
                        f"Incomplete-group refill requires {refilled_prompts + num_failed} prompts, "
                        f"exceeding the bounded budget {refill_budget}"
                    )

                num_failed, metrics = self._evict_incomplete_groups(partition_id)
                failure_metrics.update(metrics)
                refilled = self.refill_fn(num_failed)
                if refilled != num_failed:
                    raise RuntimeError(f"refill_fn submitted {refilled} prompts, expected {num_failed}")

                refill_rounds += 1
                refilled_prompts += refilled
                failure_metrics["training/rollout_failure/refilled_prompts"] += refilled
                failure_metrics["training/rollout_failure/refill_rounds"] += 1
                logger.warning(
                    "Evicted %d incomplete rollout groups and submitted exact replacements (round %d/%d)",
                    num_failed,
                    refill_rounds,
                    self.max_incomplete_group_refill_rounds,
                )
                continue

            if self._has_enough_samples(global_steps, partition_id, batch_size):
                break

            time.sleep(self.poll_interval)
            if time.time() - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
                logger.info(
                    "pending: %d, running: %d, finished: %d, failure: %d",
                    len(self.pending_keys[partition_id]),
                    len(self.running_keys[partition_id]),
                    len(self.finished_keys[partition_id]),
                    len(self.failure_keys[partition_id]),
                )
                last_debug_time = time.time()

        finished_keys = self.finished_keys[partition_id]
        failure_keys = self.failure_keys[partition_id]
        prompt_global_steps = self.prompt_global_steps[partition_id]
        sampleable_keys = sorted(finished_keys | failure_keys, key=lambda key: prompt_global_steps.get(key, 0))
        selected_prompt_uids = sampleable_keys[:batch_size]
        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        selected = set(selected_prompt_uids)
        keys, tags = [], []
        for key, tag in self.partitions[partition_id].items():
            if self._trajectory_uid(key) in selected:
                keys.append(key)
                tags.append(tag)

        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        batch, off_policy_metrics = self._drop_max_off_policy_samples(global_steps, partition_id, batch)
        return batch, {**off_policy_metrics, **dict(failure_metrics)}
