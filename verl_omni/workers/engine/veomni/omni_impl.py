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

"""Shared VeOmni engine for registered omni training adapters."""

from tensordict import TensorDict
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead

from verl_omni.pipelines.model_base import OmniModelBase


@EngineRegistry.register(model_type="omni_model", backend="veomni", device="cuda")
class OmniVeOmniEngine(VeOmniEngineWithLMHead):
    """Delegate model differences to the same adapter registry used by FSDP.

    verl owns model loading, FSDP2/EP, optimization and checkpointing. The
    selected (architecture, stage) adapter opts into VeOmni and supplies its
    backend setup, packed inputs and trainable-parameter policy.
    """

    def _get_model_config_path(self):
        self.model_adapter_cls = OmniModelBase.get_class(self.model_config)
        self.model_adapter_cls.setup_veomni(self.model_config, self.engine_config)
        return super()._get_model_config_path()

    def _apply_veomni_input_transforms(self, model_inputs: dict, micro_batch: TensorDict):
        super()._apply_veomni_input_transforms(model_inputs, micro_batch)
        # Keep the mapping object used by verl, while allowing adapters to
        # return a fresh dictionary or remove backend-incompatible fields.
        adapted = self.model_adapter_cls.prepare_veomni_inputs(model_inputs, micro_batch, self.model_config)
        if not isinstance(adapted, dict):
            raise TypeError(f"OmniModelBase.prepare_veomni_inputs must return a dict, got {type(adapted).__name__}.")
        if adapted is not model_inputs:
            model_inputs.clear()
            model_inputs.update(adapted)

    def prepare_model_inputs(self, micro_batch):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        model_inputs = self.model_adapter_cls.prepare_model_inputs(model_inputs, micro_batch, self.model_config)
        if not isinstance(model_inputs, dict):
            raise TypeError(
                f"OmniModelBase.prepare_model_inputs must return a dict, got {type(model_inputs).__name__}."
            )
        return model_inputs, output_args

    def _build_optimizer(self, module):
        self.model_adapter_cls.configure_veomni_trainable_params(module, self.model_config)
        return super()._build_optimizer(module)
