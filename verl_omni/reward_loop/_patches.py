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
"""Monkey-patches over ``verl.experimental.reward_loop`` for verl-omni.

Two upstream-verl behaviours need to be adapted for image (non-token) rollouts:

1. ``RewardLoopWorker`` does not import the verl-omni reward-manager package
   inside the Ray actor process, so ``@register("visual")`` never runs and
   ``load_reward_manager`` raises ``Unknown reward manager: visual``. We patch
   ``RewardLoopWorker.__init__`` to import ``verl_omni.reward_loop`` (whose
   side effect is the registration) before the original init runs. The import
   list is forwarded to actor processes via a Ray ``runtime_env`` env var
   because Ray pickles top-level classes by reference and would otherwise not
   observe driver-side mutations.

2. ``RewardLoopManager.compute_rm_score`` hard-codes a token-level rm_scores
   layout (``rm_scores[i, last_valid_token] = score``). For image rollouts the
   score is per-sample, so we patch ``compute_rm_score`` to assemble a
   ``(batch_size, 1)`` rm_scores tensor instead.

TODO(verl-omni): replace these monkey-patches by upstreaming proper extension
points into ``verl.experimental.reward_loop`` (e.g. a class-level
``score_assembler`` hook on ``RewardLoopManager`` plus an ``extra_imports``
hook on ``RewardLoopWorker`` that is forwarded to actor processes via
``runtime_env``). Then this module can be deleted.
"""

import importlib
import os

import numpy as np
import ray
from tensordict import TensorDict
from verl import DataProto
from verl.experimental.reward_loop import reward_loop as _reward_loop_module
from verl.experimental.reward_loop.reward_loop import RewardLoopManager, RewardLoopWorker

from .score_assembler import visual_score_assembler

_EXTRA_IMPORTS_ENV_VAR = "VERL_OMNI_REWARD_LOOP_EXTRA_IMPORTS"
_EXTRA_IMPORTS = ["verl_omni.reward_loop"]
_PATCHED_FLAG = "_verl_omni_patched"


def apply_patches() -> None:
    """Apply the verl-omni monkey-patches to the verl reward-loop classes.

    Idempotent: calling more than once is a no-op.
    """
    if getattr(_reward_loop_module, _PATCHED_FLAG, False):
        return

    orig_worker_init = RewardLoopWorker.__init__
    orig_init_workers = RewardLoopManager._init_reward_loop_workers

    def patched_worker_init(self, config, reward_router_address=None):
        env_modules = os.environ.get(_EXTRA_IMPORTS_ENV_VAR, "")
        modules = [m for m in env_modules.split(",") if m] or list(_EXTRA_IMPORTS)
        for module_path in modules:
            importlib.import_module(module_path)
        orig_worker_init(self, config, reward_router_address)

    def patched_init_reward_loop_workers(self):
        # Wrap the actor-class so every ``.options(...)`` call injects the
        # runtime_env env var that triggers verl-omni imports inside the actor.
        original_class = self.reward_loop_workers_class
        extra = ",".join(_EXTRA_IMPORTS)

        class _OptionsProxy:
            def options(_proxy, **opts):
                runtime_env = dict(opts.get("runtime_env") or {})
                env_vars = dict(runtime_env.get("env_vars") or {})
                env_vars.setdefault(_EXTRA_IMPORTS_ENV_VAR, extra)
                runtime_env["env_vars"] = env_vars
                opts["runtime_env"] = runtime_env
                return original_class.options(**opts)

        self.reward_loop_workers_class = _OptionsProxy()
        try:
            orig_init_workers(self)
        finally:
            self.reward_loop_workers_class = original_class

    def patched_compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]

        scores = [item["reward_score"] for item in outputs_flat]
        rm_scores = visual_score_assembler(data, scores)
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        non_tensor_batch = {}
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"reward_extra_keys": reward_extra_keys},
        )

    RewardLoopWorker.__init__ = patched_worker_init
    RewardLoopManager._init_reward_loop_workers = patched_init_reward_loop_workers
    RewardLoopManager.compute_rm_score = patched_compute_rm_score

    setattr(_reward_loop_module, _PATCHED_FLAG, True)


# Apply patches eagerly on import so any caller that imports
# ``verl_omni.reward_loop`` immediately gets the patched behaviour.
apply_patches()
