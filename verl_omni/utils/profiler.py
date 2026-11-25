# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Shared Ray launch options and controller capture state for Omni profiling."""

from omegaconf import OmegaConf
from verl.plugin.platform import get_platform
from verl.utils.device import is_cuda_available
from verl.utils.profiler import build_controller_nsight_launch_options


def build_controller_runner_options(config) -> dict:
    """Build the TaskRunner actor options for controller Nsight capture."""
    if (
        is_cuda_available
        and OmegaConf.select(config, "global_profiler.tool") == "nsys"
        and OmegaConf.select(config, "global_profiler.steps") is not None
        and len(OmegaConf.select(config, "global_profiler.steps")) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = build_controller_nsight_launch_options(config)
        return {"runtime_env": {"nsight": nsight_options}}
    return {}


class ControllerNsightProfiler:
    """Track controller CUDA-profiler API capture independently of worker lifecycles."""

    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.active = False

    def start(self) -> bool:
        """Start enabled capture, returning whether this call started it."""
        if self.enabled:
            if self.active:
                raise RuntimeError("Controller Nsight profiling is already active")
            get_platform().profiler_start()
            self.active = True
            return True
        return False

    def stop(self) -> None:
        """Stop active capture and clear its state even if stopping fails."""
        if self.active:
            try:
                get_platform().profiler_stop()
            finally:
                self.active = False
