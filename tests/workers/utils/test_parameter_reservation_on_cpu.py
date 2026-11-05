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

"""Local version-fencing contracts; these are not distributed RL tests."""

import threading

import pytest

from verl_omni.workers.utils.parameter_revision import ParameterRevision


def test_reservation_fences_parameter_operations_until_matching_abort():
    tracker = ParameterRevision()
    version = tracker.version()
    tracker.reserve_update(version, "first")
    assert tracker.version() == version
    with pytest.raises(RuntimeError, match="reserved"):
        tracker.reserve_update(version, "other")
    for operation in (
        tracker.mutation("load"),
        tracker.mutation("optimizer"),
        tracker.access(),
        tracker.access(internal=True),
        tracker.device_transition(),
        tracker.preflight_update(version, "first"),
        tracker.update(version, "first"),
    ):
        with pytest.raises(RuntimeError, match="reserved"):
            with operation:
                pytest.fail("A reserved operation must not begin")
    with pytest.raises(ValueError, match="does not match"):
        tracker.abort_reserved_update(version, "other")
    with pytest.raises(RuntimeError, match="reserved"):
        with tracker.mutation("load"):
            pytest.fail("A mismatched abort must not clear the reservation")
    tracker.abort_reserved_update(version, "first")
    with tracker.mutation("load"):
        pass
    assert tracker.version().revision == version.revision + 1


def test_commit_is_exactly_once_and_invalid_reservations_do_not_poison():
    tracker = ParameterRevision()
    version = tracker.version()
    tracker.reserve_update(version, "first")
    with pytest.raises(ValueError, match="does not match"):
        with tracker.commit_reserved_update(version, "wrong"):
            pytest.fail("Wrong cohort must not enter")
    tracker.abort_reserved_update(version, "first")
    tracker.reserve_update(version, "first")
    with tracker.commit_reserved_update(version, "first"):
        with tracker.mutation("optimizer"):
            pass
    assert tracker.version().revision == version.revision + 1
    with pytest.raises(ValueError, match="Stale"):
        tracker.reserve_update(version, "second")
    with pytest.raises(ValueError, match="consumed"):
        tracker.reserve_update(tracker.version(), "first")
    assert tracker.version().revision == version.revision + 1


def test_failed_or_incomplete_commit_poison_the_engine():
    tracker = ParameterRevision()
    version = tracker.version()
    tracker.reserve_update(version, "first")
    with pytest.raises(RuntimeError, match="exactly one"):
        with tracker.commit_reserved_update(version, "first"):
            pass
    with pytest.raises(RuntimeError, match="Partial"):
        tracker.version()


def test_reservation_blocks_another_thread_without_silently_expiring():
    tracker = ParameterRevision()
    version = tracker.version()
    tracker.reserve_update(version, "first")
    result = []

    def writer():
        try:
            with tracker.mutation("load"):
                result.append("entered")
        except RuntimeError as exc:
            result.append(str(exc))

    thread = threading.Thread(target=writer)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result == ["An update is reserved"]
    tracker.abort_reserved_update(version, "first")
    assert tracker.version() == version
