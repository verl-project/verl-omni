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

import json
import os

from verl.utils import fs
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


class DiffusersFSDPCheckpointManager(FSDPCheckpointManager):
    """Preserve the origin of the base weights used by the Diffusers actor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        module = getattr(self.model, "_fsdp_wrapped_module", self.model)
        # Inactive adapters can also rewrite the shared base weights.
        initializations = (config.init_lora_weights for config in getattr(module, "peft_config", {}).values())
        self._base_weights_modified: bool | None = any(
            isinstance(initialization, str)
            and (initialization.startswith(("pissa", "corda")) or initialization.lower() in ("olora", "loftq"))
            for initialization in initializations
        )

    def _get_lora_train_meta(self, unwrap_model):
        metadata = super()._get_lora_train_meta(unwrap_model)
        if metadata is not None:
            metadata["base_weights_modified"] = self._base_weights_modified
        return metadata

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """Restore the base-weight marker when the model load replaces the base."""
        if not self.should_load_model or not self._has_lora():
            return super().load_checkpoint(local_path, hdfs_path, del_local_after_load)

        base_loaded = False

        def record_base_load(_module, incompatible_keys):
            nonlocal base_loaded
            # Adapter-only loads leave the base parameters in missing_keys.
            base_loaded = not incompatible_keys.missing_keys

        with self.model.register_load_state_dict_post_hook(record_base_load):
            super().load_checkpoint(local_path, hdfs_path, del_local_after_load)

        if base_loaded:
            self._base_weights_modified = None
            metadata_path = os.path.join(local_path, "lora_train_meta.json")
            if fs.exists(metadata_path):
                with open(fs.copy_to_local(metadata_path), encoding="utf-8") as file:
                    self._base_weights_modified = json.load(file).get("base_weights_modified")
