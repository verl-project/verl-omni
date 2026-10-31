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

"""Compare offline microbatch candidates using a trusted single-process worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _keys(value: Any, expected: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"expected exactly these keys: {sorted(expected)}")
    return value


def _integer(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("expected a positive integer")
    return value


def _number(value: Any, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError("expected a finite nonnegative number" + (" greater than zero" if positive else ""))
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError("number is not representable as a finite float") from error
    if not math.isfinite(converted) or (positive and converted <= 0) or converted < 0:
        raise ValueError("expected a finite nonnegative number" + (" greater than zero" if positive else ""))
    return converted


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=reject_constant)


@dataclass(frozen=True)
class WorkloadIdentity:
    """Canonical string fields that bind every trial to one offline workload."""

    fields: tuple[tuple[str, str], ...]

    @classmethod
    def parse(cls, value: Any) -> WorkloadIdentity:
        """Validate an identity map and sort its fields for stable hashing.

        Args:
            value: Nonempty JSON object with string keys and values.

        Returns:
            Canonical workload identity.
        """
        if (
            type(value) is not dict
            or not value
            or any(type(k) is not str or not k or type(v) is not str for k, v in value.items())
        ):
            raise ValueError("workload identity must be a nonempty string map")
        return cls(tuple(sorted(value.items())))

    def mapping(self) -> dict[str, str]:
        """Return the identity as JSON-ready string fields."""
        return dict(self.fields)

    def digest(self) -> str:
        """Return the SHA256 digest of canonical JSON identity bytes."""
        return hashlib.sha256(_json(self.mapping()).encode()).hexdigest()


@dataclass(frozen=True)
class SearchPlan:
    """Finite candidate and numerical-alignment policy for one offline search."""

    effective_batch: int
    candidates: tuple[int, ...]
    baseline: int
    max_trials: int
    steps: int
    atol: float
    rtol: float

    @classmethod
    def parse(cls, value: Any) -> SearchPlan:
        """Validate the exact plan schema and all divisibility constraints.

        Args:
            value: JSON plan with candidates, baseline, budget and tolerances.

        Returns:
            Validated search plan.
        """
        value = _keys(value, {"effective_batch", "candidates", "baseline", "max_trials", "steps", "atol", "rtol"})
        batch = _integer(value["effective_batch"])
        raw = value["candidates"]
        if type(raw) is not list or not raw:
            raise ValueError("candidates must be a nonempty list")
        candidates = tuple(_integer(item) for item in raw)
        if len(set(candidates)) != len(candidates) or any(batch % item for item in candidates):
            raise ValueError("candidates must be unique divisors of effective_batch")
        baseline = _integer(value["baseline"])
        if baseline not in candidates:
            raise ValueError("baseline must be a candidate")
        return cls(
            batch,
            candidates,
            baseline,
            _integer(value["max_trials"]),
            _integer(value["steps"]),
            _number(value["atol"]),
            _number(value["rtol"]),
        )

    def ordered(self) -> tuple[int, ...]:
        """Place the baseline first while retaining other candidate order."""
        return (self.baseline, *(candidate for candidate in self.candidates if candidate != self.baseline))


def _vector(value: Any) -> list[float]:
    if type(value) is not list or not value:
        raise ValueError("witness vector must be nonempty")
    converted = []
    for item in value:
        if type(item) not in (int, float):
            raise ValueError("witness vector must contain finite numbers")
        try:
            number = float(item)
        except OverflowError as error:
            raise ValueError("witness number is not representable as a finite float") from error
        if not math.isfinite(number):
            raise ValueError("witness vector must contain finite numbers")
        converted.append(number)
    return converted


def _witness(value: Any, steps: int) -> list[dict[str, list[float]]]:
    if type(value) is not list or len(value) != steps:
        raise ValueError("wrong witness update count")
    return [
        {key: _vector(_keys(step, {"loss", "grads", "params"})[key]) for key in ("loss", "grads", "params")}
        for step in value
    ]


def _result(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    value = _keys(
        value,
        {
            "trial_id",
            "microbatch",
            "accumulation",
            "steps",
            "workload",
            "workload_digest",
            "status",
            "reason",
            "initial_state_digest",
            "update_seconds",
            "witnesses",
        },
    )
    for key in ("trial_id", "microbatch", "accumulation", "steps", "workload", "workload_digest"):
        if type(value[key]) is not type(request[key]) or value[key] != request[key]:
            raise ValueError(f"result {key} mismatch")
    if value["status"] not in ("ok", "oom", "failure") or type(value["status"]) is not str:
        raise ValueError("invalid status")
    if type(value["reason"]) is not str:
        raise ValueError("invalid reason")
    if value["status"] == "ok":
        if value["reason"] or type(value["initial_state_digest"]) is not str or not value["initial_state_digest"]:
            raise ValueError("invalid successful metadata")
        if type(value["update_seconds"]) is not list or len(value["update_seconds"]) != request["steps"]:
            raise ValueError("wrong timing count")
        value["update_seconds"] = [_number(item, positive=True) for item in value["update_seconds"]]
        value["witnesses"] = _witness(value["witnesses"], request["steps"])
    elif value["update_seconds"] != [] or value["witnesses"] != [] or value["initial_state_digest"] is not None:
        raise ValueError("failed result has successful data")
    return value


def _aligned(reference: dict[str, Any], trial: dict[str, Any], plan: SearchPlan) -> bool:
    if reference["initial_state_digest"] != trial["initial_state_digest"]:
        return False
    for left, right in zip(reference["witnesses"], trial["witnesses"], strict=True):
        for key in ("loss", "grads", "params"):
            if len(left[key]) != len(right[key]):
                return False
            if any(
                not math.isclose(a, b, abs_tol=plan.atol, rel_tol=plan.rtol)
                for a, b in zip(left[key], right[key], strict=True)
            ):
                return False
    return True


def _settle_child(child: subprocess.Popen[bytes]) -> None:
    """Settle only the owned direct child; the one-second bound applies to shutdown, not work."""
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if child.poll() is None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
                child.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


def run(plan: SearchPlan, workload: WorkloadIdentity, argv: list[str], output: Path) -> dict[str, Any]:
    """Run bounded trials in fresh direct worker processes and write records.

    Args:
        plan: Validated candidate and comparison policy.
        workload: Canonical identity shared by every candidate.
        argv: Trusted single-process worker command without request paths.
        output: New directory owned exclusively by this call.

    Returns:
        Aggregate trial evaluations and optional recommendation.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("controller must run on the main thread for exact child interrupt cleanup")
    if type(plan) is not SearchPlan or type(workload) is not WorkloadIdentity:
        raise ValueError("plan and workload must use the declared dataclasses")
    checked_plan = SearchPlan.parse({**asdict(plan), "candidates": list(plan.candidates)})
    if checked_plan != plan or WorkloadIdentity.parse(workload.mapping()) != workload:
        raise ValueError("noncanonical or invalid plan or workload")
    if not argv or any(type(arg) is not str or not arg for arg in argv):
        raise ValueError("worker argv must be a nonempty string list")
    output.mkdir(parents=True, exist_ok=False)
    source_plan = asdict(plan)
    source_plan["candidates"] = list(plan.candidates)
    source = {"plan": source_plan, "workload": workload.mapping(), "workload_digest": workload.digest(), "argv": argv}
    _write(output / "source.json", source)
    ordered = plan.ordered()
    records: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    best: tuple[float, int] | None = None
    for index, microbatch in enumerate(ordered[: plan.max_trials]):
        trial_dir = output / f"trial-{index:03d}"
        trial_dir.mkdir()
        request = {
            "trial_id": index,
            "microbatch": microbatch,
            "accumulation": plan.effective_batch // microbatch,
            "steps": plan.steps,
            "effective_batch": plan.effective_batch,
            "workload": workload.mapping(),
            "workload_digest": workload.digest(),
        }
        _write(trial_dir / "request.json", request)
        launch_error = ""
        with (trial_dir / "stdout.txt").open("x") as stdout, (trial_dir / "stderr.txt").open("x") as stderr:
            try:
                child = subprocess.Popen(
                    [*argv, "--request", str(trial_dir / "request.json"), "--result", str(trial_dir / "result.json")],
                    stdout=stdout,
                    stderr=stderr,
                )
            except OSError as error:
                launch_error = f"worker launch failed: {error}"
            else:
                try:
                    code = child.wait()
                except BaseException:
                    _settle_child(child)
                    raise
        reason = launch_error
        result = None
        latency = None
        try:
            if launch_error:
                raise ValueError(launch_error)
            if code:
                raise ValueError(f"worker exited {code}")
            result = _result(_read(trial_dir / "result.json"), request)
            if result["status"] != "ok":
                raise ValueError(f"declared {result['status']}: {result['reason']}")
            latency = sum(result["update_seconds"]) / plan.steps
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("invalid derived per-update latency")
            if index == 0:
                baseline = result
            elif baseline is None:
                raise ValueError("baseline unavailable")
            elif not _aligned(baseline, result, plan):
                raise ValueError("numerical or initial-state mismatch")
        except (OSError, ValueError, TypeError, KeyError) as error:
            reason = str(error)
        if not reason and result is not None:
            if best is None or latency < best[0]:
                best = (latency, microbatch)
        record = {
            "trial_id": index,
            "microbatch": microbatch,
            "status": "valid" if not reason else "excluded",
            "reason": reason,
            "per_update_seconds": None if reason else latency,
        }
        records.append(record)
        _write(trial_dir / "evaluation.json", record)
    aggregate = {
        "source": source,
        "trials": records,
        "budget_excluded": list(ordered[plan.max_trials :]),
        "recommendation": None if best is None else {"microbatch": best[1], "per_update_seconds": best[0]},
        "exploratory_only": True,
    }
    _write(output / "aggregate.json", aggregate)
    return aggregate


def main() -> None:
    """Parse a plan, identity and trusted worker command for one offline search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("worker", nargs=argparse.REMAINDER, help="worker argv after --")
    args = parser.parse_args()
    argv = args.worker[1:] if args.worker[:1] == ["--"] else args.worker
    aggregate = run(SearchPlan.parse(_read(args.plan)), WorkloadIdentity.parse(_read(args.workload)), argv, args.output)
    print(_json(aggregate))


if __name__ == "__main__":
    main()
