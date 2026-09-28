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

"""Worker entry points for actor-side (native) sampling."""

from tensordict import TensorDict
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.profiler import DistProfiler

from verl_omni.workers.engine_workers import ActorRolloutRefWorker, _with_routing_replay_flag


class NativeRolloutWorker(ActorRolloutRefWorker):
    """Run a colocated actor without constructing a separate rollout engine.

    The trainer groups this worker under its actor/rollout role. Internally, the parent
    uses the actor role: sampling runs on the actor engine, so no separate rollout
    engine is needed. Model initialization, updates and checkpoints are inherited.
    """

    def __init__(self, config, role, distillation_config=None, teacher_key=None, **kwargs):
        if role not in ("actor", "actor_rollout"):
            raise ValueError(f"NativeRolloutWorker requires an actor role without a reference policy, got {role!r}.")
        super().__init__(
            config=config,
            role="actor",
            distillation_config=distillation_config,
            teacher_key=teacher_key,
            **kwargs,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="green", role="generate")
    @_with_routing_replay_flag(enabled=True)
    def generate(self, data: TensorDict) -> TensorDict:
        """Dispatch a local generation batch to the engine hooks."""
        generate = getattr(self.actor.engine, "generate_rollout", None)
        if generate is None:
            raise NotImplementedError(f"{type(self.actor.engine).__name__} does not support actor-side generation")
        output = generate(data)
        return output.cpu() if output is not None else None
