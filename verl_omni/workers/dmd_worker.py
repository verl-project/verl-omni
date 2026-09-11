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
"""DMD2 specialization of the shared diffusion TrainingWorker."""

from functools import partial

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import TrainingWorkerConfig
from verl.workers.engine import EngineRegistry

from verl_omni.workers.engine_workers import TrainingWorker
from verl_omni.workers.utils.losses import diffusion_loss


class DMDTrainingWorker(TrainingWorker):
    """Pass DMD settings into the engine without duplicating worker initialization."""

    def __init__(self, config, *, dmd_config, role="actor", distillation_config=None):
        if role != "actor" or (distillation_config is not None and distillation_config.get("enabled", False)):
            raise ValueError("DMD2 uses one offline actor group, not OPD teacher workers.")
        self.dmd_config = omega_conf_to_dataclass(dmd_config)
        self.actor_config = omega_conf_to_dataclass(config.actor)
        model_config = omega_conf_to_dataclass(config.model)
        profiler = self.actor_config.profiler
        if profiler is not None and profiler.tool_config.get(profiler.tool) is not None:
            profiler.tool_config[profiler.tool] = omega_conf_to_dataclass(
                config.actor.profiler.tool_config[profiler.tool]
            )
        worker_config = TrainingWorkerConfig(
            model_type="diffusion_dmd_model",
            model_config=model_config,
            engine_config=self.actor_config.engine,
            optimizer_config=self.actor_config.optim,
            checkpoint_config=self.actor_config.checkpoint,
            profiler_config=profiler,
        )
        super().__init__(worker_config)
        self.loss_fn = partial(diffusion_loss, config=self.actor_config)

    def build_engine(self):
        """Use the registry with one additional typed DMD configuration."""
        from verl_omni.workers.engine.fsdp import dmd_impl  # noqa: F401

        return EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            dmd_config=self.dmd_config,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Reuse the standard worker reset/model initialization boundary."""
        self.reset()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=True)
    def update_actor(self, data):
        """Execute exactly one optimizer attempt using existing mini/microbatch machinery."""
        stage = tu.get_non_tensor_data(data, "dmd_stage", default="student")
        if stage not in {"student", "fake_score"}:
            raise ValueError(f"Invalid DMD2 stage {stage!r}.")
        if tu.get_non_tensor_data(data, "epochs", default=1) != 1:
            raise ValueError("DMD2 update_actor performs one optimizer attempt, not multiple epochs.")
        previous = self.engine.active_stage
        self.engine.select_stage(stage)
        tu.assign_non_tensor(
            data,
            global_token_num=None,
            num_mini_batch=1,
            mini_batch_size=None,
            epochs=1,
            dataloader_kwargs={"shuffle": False},
            micro_batch_size_per_gpu=getattr(self.dmd_config, f"{stage}_micro_batch_size_per_gpu"),
        )
        try:
            result = self.train_mini_batch(data)
            if result is not None:
                metrics = tu.get_non_tensor_data(result, "metrics", default=None)
                metrics["dmd/update_applied"] = float(self.engine.last_step_succeeded)
                metrics["dmd/skip_nonfinite"] = float(not self.engine.last_step_succeeded)
            return result
        finally:
            self.engine.select_stage(previous)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_model_provenance(self):
        """Read the resolved checkpoint identity used by this worker, not a moving hub alias."""
        from verl_omni.utils.fs import diffusion_model_provenance

        return diffusion_model_provenance(self.engine.model_config.local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_student(self, directory, role="student"):
        """Export the selected student adapter without exposing score-model weights."""
        self.engine.export_student(directory, role)
