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

"""Worker operations for qualified single-update actor reuse."""

from verl_omni.pipelines.proximal_reuse import rpc as proximal_rpc
from verl_omni.pipelines.step_extension import WorkerOperations, register_worker_extension
from verl_omni.workers.utils.parameter_revision import parameter_access

EXTENSION = "proximal_actor_reuse"


@parameter_access("actor.engine")
def _qualify(worker, payload):
    if payload is not None:
        raise ValueError("Qualification takes no payload")
    return proximal_rpc.qualify_actor(worker)


def _get_preparation(worker, payload):
    if payload is not None:
        raise ValueError("Preparation inspection takes no payload")
    return proximal_rpc.get_actor_preparation(worker)


def _get_update(worker, payload):
    if payload is not None:
        raise ValueError("Update inspection takes no payload")
    return proximal_rpc.get_staged_actor_update(worker)


register_worker_extension(
    EXTENSION,
    WorkerOperations(
        stage={"prepare": proximal_rpc.prepare_actor_locally, "update": proximal_rpc.stage_actor_update},
        control={
            "qualify": _qualify,
            "get_preparation": _get_preparation,
            "discard_preparation": proximal_rpc.discard_actor_preparation,
            "install": proximal_rpc.install_actor_preparation,
            "get_update": _get_update,
            "discard_update": proximal_rpc.discard_staged_actor_update,
        },
        commit={"update": proximal_rpc.commit_staged_actor_update},
    ),
)
