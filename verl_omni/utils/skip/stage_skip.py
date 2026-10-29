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
"""Per-stage skip classes for the diffusion RL pipeline.

verl's SkipManager protocol (``BaseSkip`` / ``SKIP_REGISTRY`` / ``annotate``)
is stage-agnostic but ships only rollout roles. This module registers the
pipeline stages verl-omni owns -- reward, old/ref log-prob, advantage, actor
update, weight sync -- plus a driver-side rollout replay for the v0 diffusion
trainer. Instances are created by :func:`verl_omni.utils.skip.init_skip_manager`
from the ``omni_skip`` config block; a decorated stage call whose global step is
listed in ``steps`` replays its output from
``{dump_dir}/{experiment}_{project}/{role}/{step}/{call_idx}/`` instead of
running the model forward. ``call_idx`` is the mini-batch ordinal within the
step (always 0 unless the v1 trainer splits the step, ``parameter_sync_step >
1``), mirroring upstream verl's ``rollout_tq`` cache layout.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf
from verl.protocol import DataProto
from verl.utils.skip import SkipManager
from verl.utils.skip.base_skip import BaseSkip, SkipAction, register_skip
from verl.utils.skip.rollout_skip import RolloutSkip

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE_SKIP_ROLES",
    "ActorUpdateSkip",
    "AdvantageSkip",
    "DiffusionRolloutSkip",
    "OldLogProbSkip",
    "RefLogProbSkip",
    "RewardSkip",
    "StageSkip",
    "WeightSyncSkip",
    "skip_stage_metrics",
]

STAGE_SKIP_ROLES = ("reward", "old_log_prob", "ref_log_prob", "advantage", "actor_update", "weight_sync")

PAYLOAD_NAME = "stage_payload.pt"
META_NAME = "meta.json"


class StageSkip(BaseSkip):
    """Replay one pipeline stage's cached output for configured steps.

    Cache-miss steps run the real stage and pickle its return value under the
    step directory; cache-hit steps return the pickled value without executing
    the stage. ``cache`` replays the exact step, ``repeat`` the closest dumped
    step, ``empty`` skips the stage without any cache (subclasses define the
    empty return). Skipped calls are counted loudly on the instance for the
    end-of-run summary.

    A step's cache holds one slot per stage call (``{step}/{call_idx}/``): the
    v1 separate-async trainer runs ``parameter_sync_step`` mini-batch calls per
    global step, and replaying mini-batch 0's output into mini-batch 1 would
    feed wrong samples to the update. ``repeat`` only considers a step whose
    slots are all dumped (complete cache).
    """

    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    print_mark = "[StageSkip] "
    role = ""

    def __init__(self, local_config: DictConfig, global_config: DictConfig) -> None:
        super().__init__(local_config, global_config)
        self.exp_name = global_config.trainer.get("experiment_name", "default_experiment_name")
        self.project_name = global_config.trainer.get("project_name", "default_project_name")
        trainer_mode = OmegaConf.select(global_config, "trainer.v1.trainer_mode", default="sync")
        self.parameter_sync_step = max(
            1, int(OmegaConf.select(global_config, f"trainer.v1.{trainer_mode}.parameter_sync_step", default=1))
        )
        # Driver-only counters: one StageSkip instance per role per process.
        self.skipped_calls = 0
        self.skipped_steps: set[int] = set()
        # Ordinal of the current call within the step; the decorator reaches
        # meet_precondition once per eligible (non-validation) call, so this
        # tracks mini-batch 0, 1, ... of a split step.
        self._call_step: int | None = None
        self._call_idx = 0

    def meet_precondition(self, step: int, func: Callable, *args, **kwargs) -> bool:
        if self.action == SkipAction.EMPTY:
            return True
        self._advance_call_idx(step)
        if self.action == SkipAction.CACHE:
            if self._payload_path(step, self._call_idx).is_file():
                return True
            logger.info(
                "%sNo cached %s output at step %s (call %s) under %s; the stage will run and dump.",
                self.print_mark,
                self.role,
                step,
                self._call_idx,
                self._step_dir(step),
            )
            return False
        if self._find_latest_step(step) != -1:
            return True
        logger.info(
            "%sNo cached %s output under %s; the stage will run and dump.",
            self.print_mark,
            self.role,
            self._role_dir(),
        )
        return False

    def warp_function(self, step: int, func: Callable, *args, **kwargs):
        if self.action == SkipAction.EMPTY:
            logger.info(
                "%s\033[33mSkipping stage '%s' (empty action) at step %s\033[0m",
                self.print_mark,
                self.role,
                step,
            )
            self._record_skip(step)
            return self.empty_result(*args, **kwargs)

        load_step = step if self.action == SkipAction.CACHE else self._find_latest_step(step)
        payload_path = self._payload_path(load_step, self._call_idx)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        logger.info(
            "%s\033[33mLoad cached %s output at step %s (request step %s, call %s) from %s\033[0m",
            self.print_mark,
            self.role,
            load_step,
            step,
            self._call_idx,
            payload_path,
        )
        self._record_skip(step)
        return payload["result"]

    def prepare_data(self, step: int, result, *args, **kwargs) -> None:
        step_dir = self._payload_dir(step, self._call_idx)
        tmp_path = step_dir / (PAYLOAD_NAME + ".tmp")
        try:
            step_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                torch.save({"result": result, "global_steps": step, "call_idx": self._call_idx}, f)
            # Atomic swap so meet_precondition never sees a partial dump.
            os.replace(tmp_path, step_dir / PAYLOAD_NAME)
            (step_dir / META_NAME).write_text(
                json.dumps({"global_steps": step, "call_idx": self._call_idx, "role": self.role})
            )
            logger.info(
                "%s\033[33mDump %s output at step %s (call %s) to %s\033[0m",
                self.print_mark,
                self.role,
                step,
                self._call_idx,
                step_dir,
            )
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            logger.warning(
                "%s\033[31mFailed to dump %s output at step %s (call %s) to %s: %s\033[0m",
                self.print_mark,
                self.role,
                step,
                self._call_idx,
                step_dir,
                e,
            )

    def empty_result(self, *args, **kwargs):
        """Return the stand-in value used when the stage is skipped via the ``empty`` action."""
        raise NotImplementedError(f"Stage '{self.role}' does not support the empty action")

    def _record_skip(self, step: int) -> None:
        self.skipped_calls += 1
        self.skipped_steps.add(step)

    def _advance_call_idx(self, step: int) -> None:
        """Advance the mini-batch ordinal: restart at 0 when the step changes."""
        if step != self._call_step:
            self._call_step = step
            self._call_idx = 0
        else:
            self._call_idx += 1

    def _role_dir(self) -> Path:
        dumped_dir = Path(self.dump_dir).expanduser().resolve()
        return dumped_dir.joinpath(f"{self.exp_name}_{self.project_name}", self.role).absolute()

    def _step_dir(self, step: int) -> Path:
        return self._role_dir().joinpath(str(step)).absolute()

    def _payload_dir(self, step: int, call_idx: int) -> Path:
        return self._step_dir(step).joinpath(str(call_idx)).absolute()

    def _payload_path(self, step: int, call_idx: int) -> Path:
        return self._payload_dir(step, call_idx).joinpath(PAYLOAD_NAME)

    def _step_cache_complete(self, step_dir: Path) -> bool:
        """All ``parameter_sync_step`` call slots of the step hold a dumped payload."""
        return all((step_dir / str(i) / PAYLOAD_NAME).is_file() for i in range(self.parameter_sync_step))

    def _available_steps(self) -> list[int]:
        result: list[int] = []
        role_dir = self._role_dir()
        if not role_dir.is_dir():
            return result
        for child in role_dir.iterdir():
            if child.is_dir() and self._step_cache_complete(child):
                try:
                    result.append(int(child.name))
                except ValueError:
                    continue
        return sorted(result)

    def _find_latest_step(self, step: int) -> int:
        """Prefer the exact step, else the closest dumped step below, else above; -1 if none."""
        if self._step_cache_complete(self._step_dir(step)):
            return step
        available = self._available_steps()
        smaller = [s for s in available if s < step]
        if smaller:
            return smaller[-1]
        larger = [s for s in available if s > step]
        if larger:
            return larger[0]
        return -1


@register_skip("reward")
class RewardSkip(StageSkip):
    """Cache/replay colocated reward-model scores (``omni_skip.reward``)."""

    role = "reward"


@register_skip("old_log_prob")
class OldLogProbSkip(StageSkip):
    """Cache/replay recomputed old log-probs (``omni_skip.old_log_prob``)."""

    role = "old_log_prob"


@register_skip("ref_log_prob")
class RefLogProbSkip(StageSkip):
    """Cache/replay reference-policy outputs: ref log-probs (policy gradient) or ref noise
    preds (direct preference), whichever the configured trainer computes (``omni_skip.ref_log_prob``)."""

    role = "ref_log_prob"


@register_skip("advantage")
class AdvantageSkip(StageSkip):
    """Cache/replay driver-computed advantages (``omni_skip.advantage``)."""

    role = "advantage"


@register_skip("actor_update")
class ActorUpdateSkip(StageSkip):
    """Skip the actor update (``omni_skip.actor_update``).

    ``empty`` no-ops the update and returns empty metrics so the metrics
    pipeline still runs; ``cache``/``repeat`` replay the metrics of an earlier
    real update. No checkpoint is written from a skipped update.
    """

    role = "actor_update"
    support_actions = [SkipAction.CACHE, SkipAction.REPEAT, SkipAction.EMPTY]

    def empty_result(self, *args, **kwargs) -> DataProto:
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {}})


@register_skip("weight_sync")
class WeightSyncSkip(StageSkip):
    """Skip the mid-cycle rollout-weight refresh (``omni_skip.weight_sync``).

    ``empty`` only: lifecycle syncs (step end, validation) always run, so
    replicas wake up but keep pre-update weights until the next sync.
    """

    role = "weight_sync"
    support_actions = [SkipAction.EMPTY]

    def empty_result(self, *args, **kwargs) -> None:
        return None


@register_skip("rollout")
class DiffusionRolloutSkip(RolloutSkip):
    """Driver-side rollout replay for the v0 diffusion trainer.

    The diffusion agent loop echoes the prompt batch's ``non_tensor_batch``
    (uid, dataset metadata) into its output and the trainer unions output with
    the fresh batch, so a cached generation is re-stamped with the current
    call's non-tensor fields before it is returned.
    """

    print_mark = "[DiffusionRolloutSkip()] "

    def warp_function(self, step: int, func: Callable, *args, **kwargs) -> Any:
        result = super().warp_function(step, func, *args, **kwargs)
        gen_batch = kwargs.get("prompts")
        if gen_batch is None and len(args) > 1:
            gen_batch = args[1]
        if gen_batch is not None and hasattr(result, "non_tensor_batch"):
            for key, value in gen_batch.non_tensor_batch.items():
                result.non_tensor_batch[key] = value
        return result


def skip_stage_metrics() -> dict[str, int]:
    """End-of-run counters: calls skipped and distinct steps skipped per stage."""
    metrics: dict[str, int] = {}
    for role in STAGE_SKIP_ROLES:
        instance = SkipManager.skip_instances.get(role)
        if instance is None or not instance.is_enabled():
            continue
        metrics[f"skip/{role}/calls"] = instance.skipped_calls
        metrics[f"skip/{role}/steps"] = len(instance.skipped_steps)
    return metrics
