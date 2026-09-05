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
"""Pure, deterministic distillation trainer control plane.

The control plane receives an immutable :class:`DistillationPlan` and talks to a
single :class:`DistillationPhaseExecutor`. It never imports a model pipeline,
manipulates latents, selects a PEFT adapter, or computes a DMD loss. Keeping the
controller free of Ray types makes its state machine testable in a CPU process
(RFC §13.1).

The cycle state machine follows RFC §14. Worker allocation, role binding, and
checkpoint restore are deliberately delegated to the bound phase executor; this module
only controls validated phase execution:

- Optional fake/discriminator warmup cycles: emit fake-only ``UpdateCycle``
  requests, advance fake/discriminator optimizer counters, never ``global_step``.
- For each ``global_step``: one student phase then ``K`` fake phases; increment
  ``global_step`` after all required phases complete; checkpoint/export/validate
  when due.

Invariants enforced here (RFC §14):

- a completed student phase reports exactly one student optimizer step;
- a skipped student phase reports no student optimizer step and cannot advance
  ``global_step``;
- a partially completed cycle is fail-fast and never retried in-process;
- a zero-progress cycle raises rather than being silently skipped;
- ``after_completed_step`` runs only after ``global_step`` is incremented.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from verl_omni.trainer.diffusion.distillation.contracts import (
    DistillationPlan,
    PhaseRequest,
    PhaseResult,
    TrainerCounters,
    UpdateCycle,
)

__all__ = [
    "DistillationTrainerControlPlane",
    "BatchProvider",
    "DistillationTrainerHooks",
    "DistillationPhaseExecutor",
    "FakePhaseExecutor",
    "FakeBatchProvider",
    "FakeDistillationHooks",
]


@runtime_checkable
class BatchProvider(Protocol):
    """Supplies per-phase input batches to the executor."""

    def next(self, phase: PhaseRequest) -> Any: ...


@runtime_checkable
class DistillationTrainerHooks(Protocol):
    """Receives post-completed-step callbacks with counters, metrics, and executor.

    ``after_completed_step`` is the sanctioned place to schedule checkpoint,
    validation, and export behavior without introducing model-specific or
    Ray-specific control flow into the generic trainer. It runs only after
    ``global_step`` is incremented, so it observes the new counter value.
    """

    def after_completed_step(self, counters: TrainerCounters, metrics: dict, executor: Any) -> None: ...


class DistillationTrainerControlPlane:
    """Pure driver over a plan and an executor. No Ray, model, or FSDP types."""

    def __init__(
        self,
        plan: DistillationPlan,
        executor: DistillationPhaseExecutor,
        batch_provider: BatchProvider,
        hooks: Optional[DistillationTrainerHooks] = None,
    ) -> None:
        self.plan = plan
        self.executor = executor
        self.batch_provider = batch_provider
        self.hooks = hooks
        self.counters = TrainerCounters()
        self._metrics: dict[str, dict] = {}
        self._failed = False

    def run(self, num_cycles: int) -> None:
        """Drive ``num_cycles`` update cycles through the executor."""
        for _ in range(num_cycles):
            self.run_cycle()

    def run_cycle(self) -> UpdateCycle:
        """Run one cycle transactionally and become terminal after a failure."""
        if self._failed:
            raise RuntimeError(
                "This control plane previously failed during a cycle and cannot be retried in-process; "
                "restore the last completed-cycle checkpoint into a new driver."
            )

        before_counters = TrainerCounters(
            global_step=self.counters.global_step,
            optimizer_steps=dict(self.counters.optimizer_steps),
            completed_cycles=self.counters.completed_cycles,
        )
        before_metrics = dict(self._metrics)
        try:
            cycle = self.plan.update_schedule.next_cycle(self.counters)
            student_step_reported = self.drive_requests(cycle.requests)

            if cycle.requires_student_update:
                if not student_step_reported:
                    raise ValueError(
                        "Cycle requires a student update but no student optimizer step was reported; "
                        "global_step must not advance."
                    )
                self.counters.increment_global()

            self.assert_progress(before_counters)
            self.counters.completed_cycles += 1

            if cycle.requires_student_update and self.hooks is not None:
                self.hooks.after_completed_step(self.counters, self.metrics, self.executor)
            return cycle
        except Exception:
            self.counters = before_counters
            self._metrics = before_metrics
            self._failed = True
            raise

    def drive_requests(self, requests: tuple[PhaseRequest, ...]) -> bool:
        """Execute phases in order and validate each executor result fail closed."""
        student_step_reported = False
        for request in requests:
            batch = self.batch_provider.next(request)
            result = self.executor.execute_phase(request, batch)
            self.validate_result(result, request)
            if request.kind == "student" and not result.optimizer_steps:
                return False
            self.accumulate_result(result, request)
            if request.kind == "student":
                student_step_reported = True
        return student_step_reported

    @staticmethod
    def validate_result(result: PhaseResult, request: PhaseRequest) -> None:
        """Require exactly one step for every role declared by a completed phase."""
        if not isinstance(result, PhaseResult):
            raise TypeError(f"execute_phase must return PhaseResult, got {type(result)}.")
        if not result.optimizer_steps and request.kind == "student":
            return
        expected_roles = set(request.trainable_roles)
        reported_roles = set(result.optimizer_steps)
        if reported_roles != expected_roles:
            raise ValueError(
                f"Phase {request.kind!r} must report optimizer steps for exactly {sorted(expected_roles)}, "
                f"got {sorted(reported_roles)}."
            )
        invalid_steps = {
            role: steps
            for role, steps in result.optimizer_steps.items()
            if isinstance(steps, bool) or not isinstance(steps, int) or steps != 1
        }
        if invalid_steps:
            raise ValueError(f"Each completed phase role must report exactly one optimizer step, got {invalid_steps}.")

    def accumulate_result(self, result: PhaseResult, request: PhaseRequest) -> None:
        """Record validated role counters and the latest phase metrics."""
        for role, steps in result.optimizer_steps.items():
            self.counters.optimizer_steps[role] = self.counters.optimizer_steps.get(role, 0) + steps
        self._metrics[request.kind] = dict(result.metrics)

    def assert_progress(self, before: TrainerCounters) -> None:
        """A cycle must advance global_step or at least one role optimizer counter."""
        if self.counters.global_step != before.global_step:
            return
        if self.counters.optimizer_steps != before.optimizer_steps:
            return
        raise ValueError("Zero-progress cycle: it advanced neither global_step nor any role optimizer counter.")

    @property
    def metrics(self) -> dict[str, dict]:
        """Metrics recorded for the most recent phase of each kind."""
        return self._metrics

    def state_dict(self) -> dict[str, Any]:
        """Return completed-cycle driver state for atomic checkpointing."""
        if self._failed:
            raise RuntimeError("Cannot checkpoint a failed control plane; restore the last completed cycle instead.")
        return {
            "global_step": self.counters.global_step,
            "optimizer_steps": dict(self.counters.optimizer_steps),
            "completed_cycles": self.counters.completed_cycles,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore validated counters into a fresh control plane."""
        if self._failed:
            raise RuntimeError("Cannot restore into a failed control plane; construct a new driver.")
        required = {"global_step", "optimizer_steps", "completed_cycles"}
        if set(state) != required:
            raise ValueError(f"Control-plane state must contain exactly {sorted(required)}, got {sorted(state)}.")
        global_step = state["global_step"]
        completed_cycles = state["completed_cycles"]
        optimizer_steps = state["optimizer_steps"]
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError(f"global_step must be a non-negative integer, got {global_step!r}.")
        if isinstance(completed_cycles, bool) or not isinstance(completed_cycles, int) or completed_cycles < 0:
            raise ValueError(f"completed_cycles must be a non-negative integer, got {completed_cycles!r}.")
        if not isinstance(optimizer_steps, dict) or any(
            not isinstance(role, str) or isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
            for role, steps in optimizer_steps.items()
        ):
            raise ValueError("optimizer_steps must map role names to non-negative integer counters.")
        trainable_roles = {binding.role for binding in self.plan.role_layout.bindings if binding.trainable}
        unknown_roles = set(optimizer_steps) - trainable_roles
        if unknown_roles:
            raise ValueError(f"Checkpoint contains unknown optimizer roles: {sorted(unknown_roles)}.")
        if optimizer_steps.get("student", 0) != global_step:
            raise ValueError("The student optimizer counter must equal global_step.")
        if completed_cycles < global_step:
            raise ValueError("completed_cycles cannot be less than global_step.")
        self.counters = TrainerCounters(
            global_step=global_step,
            optimizer_steps=dict(optimizer_steps),
            completed_cycles=completed_cycles,
        )
        self._metrics = {}

    def reset(self) -> None:
        """Reset a healthy driver; failed drivers must be reconstructed from checkpoint."""
        if self._failed:
            raise RuntimeError("A failed control plane cannot be reset in-process; construct a new driver.")
        self.counters = TrainerCounters()
        self._metrics = {}


@runtime_checkable
class DistillationPhaseExecutor(Protocol):
    """Executes one update phase and returns a :class:`PhaseResult`."""

    def execute_phase(self, request: PhaseRequest, batch: Any) -> PhaseResult:
        """Run the phase requested by ``request`` on ``batch`` and return metrics/steps."""
        ...


class FakeBatchProvider:
    """Minimal batch provider that yields synthetic batches on request."""

    def __init__(self, num_batches: int = 1, batch_size: int = 1) -> None:
        self._num = num_batches
        self._batch_size = batch_size
        self._sent = 0

    def next(self, request: PhaseRequest) -> Any:
        """Return a synthetic batch for the requested phase, advancing a counter."""
        if self._sent >= self._num:
            raise StopIteration("No more batches.")
        self._sent += 1
        return {"phase_kind": request.kind, "global_step": request.global_step, "repeat": request.repeat_index}


class FakePhaseExecutor:
    """Deterministic in-process fake executor used for CPU control-plane tests.

    It confirms the phase kind equals ``PhaseRequest.kind`` (fail-fast on a
    misordered phase), emits a deterministic metric and optimizer-step record, and
    can be configured to skip or fail a student phase for testing.
    """

    def __init__(self, skip_student: bool = False, fail_on: str | None = None) -> None:
        self._skip_student = skip_student
        self._fail_on = fail_on
        self.executed: list[PhaseRequest] = []

    def execute_phase(self, request: PhaseRequest, batch: Any) -> PhaseResult:
        """Return synthetic phase results or a configured failure."""
        if batch.get("phase_kind") != request.kind:
            raise ValueError(f"Batch phase {batch.get('phase_kind')!r} does not match request {request.kind!r}.")
        self.executed.append(request)
        if request.kind == self._fail_on:
            raise RuntimeError(f"FakePhaseExecutor failed on phase {request.kind} (global_step={request.global_step}).")
        if request.kind == "student" and self._skip_student:
            return PhaseResult(metrics={"fake/student": float(request.global_step)}, optimizer_steps={})
        metrics = {f"fake/{request.kind}": float(request.global_step)}
        optimizer_steps = {role: 1 for role in request.trainable_roles}
        return PhaseResult(metrics=metrics, optimizer_steps=optimizer_steps)


class FakeDistillationHooks:
    """Collects ``after_completed_step`` callbacks for deterministic validation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def after_completed_step(self, counters, metrics, executor) -> None:
        """Record the completed-step hook arguments."""
        self.calls.append({"global_step": counters.global_step, "metrics": dict(metrics), "executor": executor})
