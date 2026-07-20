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
import asyncio

from verl.experimental.reward_loop import RewardLoopManager


class OmniRewardLoopManager(RewardLoopManager):
    """RewardLoopManager that can start/stop the profiler on the reward-model rollout servers.

    The reward-model servers are the same ``RolloutReplica`` stack as the actor rollout
    servers, whose per-server profiler fan-out already exists (``RolloutReplica.start_profile``);
    upstream ``RewardLoopManager`` just exposes no caller for it. The trainer invokes these
    around the phase where the servers actually score: the generation phase when reward
    computation streams with rollout, or ``compute_rm_score`` in colocate mode. Configured
    via ``reward.reward_model.rollout.profiler``.
    """

    def start_profile(self, **kwargs) -> None:
        """Start profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("start_profile", **kwargs)

    def stop_profile(self) -> None:
        """Stop profiling on all reward-model rollout servers. No-op without a reward model."""
        self._run_on_replicas("stop_profile")

    def _run_on_replicas(self, method: str, **kwargs) -> None:
        if self.reward_model_manager is None:
            return
        replicas = self.reward_model_manager.rollout_replicas

        async def run_all():
            await asyncio.gather(*[getattr(replica, method)(**kwargs) for replica in replicas])

        asyncio.run(run_all())
