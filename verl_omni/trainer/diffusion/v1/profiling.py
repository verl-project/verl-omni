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
"""Step-based profiling windows for diffusion V1 trainers.

Async windows refer to trainer wall-clock steps, not to the policy version of
buffered samples. Sync rollout windows close before replicas sleep; actor and
async rollout windows can span consecutive selected steps.
"""

import logging

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def profiler_worker_group_kwargs(config) -> dict:
    """Pass Nsight launch options and selected steps to Ray worker groups."""
    steps = OmegaConf.select(config, "global_profiler.steps")
    if not steps:
        return {}
    kwargs = {"profile_steps": steps}
    if OmegaConf.select(config, "global_profiler.tool") == "nsys":
        options = OmegaConf.select(config, "global_profiler.global_tool_config.nsys.worker_nsight_options")
        if options is None:
            raise ValueError("global_profiler.global_tool_config.nsys.worker_nsight_options must be set")
        kwargs["worker_nsight_options"] = OmegaConf.to_container(options)
    return kwargs


class DiffusionV1Profiler:
    """Own paired start/stop calls, including partial-start failure cleanup."""

    def __init__(self, config, actor, rollout_managers, *, total_training_steps=None):
        self.steps = set(OmegaConf.select(config, "global_profiler.steps") or [])
        if total_training_steps is not None:
            # Out-of-range selections must not defer the finish hook forever.
            self.steps = {step for step in self.steps if 1 <= step <= total_training_steps}
        self.continuous = OmegaConf.select(config, "global_profiler.profile_continuous_steps", default=False)
        self.actor = (
            actor if OmegaConf.select(config, "actor_rollout_ref.actor.profiler.enable", default=False) else None
        )
        self.rollout_managers = (
            rollout_managers
            if OmegaConf.select(config, "actor_rollout_ref.rollout.profiler.enable", default=False)
            else []
        )
        self.controller = None
        if self.steps and OmegaConf.select(config, "global_profiler.tool") == "nsys":
            capture_range = OmegaConf.select(
                config, "global_profiler.global_tool_config.nsys.controller_nsight_options.capture-range"
            )
            if capture_range == "cudaProfilerApi":
                from verl.plugin.platform import get_platform

                self.controller = get_platform()
        self._active = {}

    def _start(self, key, start, stop, **kwargs):
        if key in self._active:
            return
        # A fan-out RPC can start some ranks before another rank fails. Include
        # this target in cleanup even when its start call raises.
        self._active[key] = stop
        start(**kwargs)

    def start_step(self, step):
        if step not in self.steps:
            return
        try:
            if self.controller is not None:
                self._start("controller", self.controller.profiler_start, self.controller.profiler_stop)
            if self.actor is not None:
                self._start("actor", self.actor.start_profile, self.actor.stop_profile, role="train", profile_step=step)
            for index, manager in enumerate(self.rollout_managers):
                # Engines expose their own start_profile API (not the worker
                # DistProfiler API), so do not forward worker-only kwargs.
                self._start(f"rollout_{index}", manager.start_profile, manager.stop_profile)
        except BaseException:
            self.close(suppress_errors=True)
            raise

    def stop_rollout(self):
        """Flush sync rollout traces before sleeping/offloading the replicas."""
        self._stop([key for key in self._active if key.startswith("rollout_")])

    def end_step(self, step, *, last_step=False):
        if last_step or not self.continuous or step + 1 not in self.steps:
            self.close(run_command=last_step or step == max(self.steps, default=-1))

    def _stop(self, keys, *, suppress_errors=False, run_command=False):
        error = None
        for key in reversed(keys):
            stop = self._active.pop(key)
            try:
                if key == "actor":
                    stop(run_command=run_command)
                else:
                    stop()
            except Exception as exc:
                logger.exception("Failed to stop diffusion V1 profiler %s", key)
                if error is None:
                    error = exc
        if error is not None and not suppress_errors:
            raise error

    def close(self, *, suppress_errors=False, run_command=False):
        """Attempt every stop even if a backend fails; repeated cleanup is safe."""
        self._stop(list(self._active), suppress_errors=suppress_errors, run_command=run_command)
