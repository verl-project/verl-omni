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

"""Route actor reuse operations through the generic worker interface."""

from verl_omni.pipelines.proximal_reuse.worker import EXTENSION
from verl_omni.pipelines.step_extension import with_worker_extension_selector


def stage(worker_group, operation: str, batch):
    """Dispatch native actor shards and return the per-rank references."""
    selected = with_worker_extension_selector(batch, EXTENSION, operation)
    return worker_group.stage_step_extension(selected)


def control(worker_group, operation: str, payload=None):
    """Synchronously collect one control response per actor."""
    return worker_group.control_step_extension(EXTENSION, operation, payload)


def commit(worker_group, operation: str, payload):
    """Dispatch a commit without assuming atomic rollback or retry."""
    return worker_group.commit_step_extension(EXTENSION, operation, payload)
