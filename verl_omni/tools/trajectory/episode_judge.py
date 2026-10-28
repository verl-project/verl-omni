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

"""Compatibility latch operations on the current episode owner."""

from .episode import current_episode


def set_good_enough_yes_reached(reached: bool) -> None:
    state = current_episode(create=True)
    latest = state.latest_generation
    if latest is not None:
        state.record_judgment(latest.call_id, good_enough=reached)
    state.set_good_enough(reached)


def get_good_enough_yes_reached() -> bool:
    state = current_episode()
    return state.good_enough if state else False


def clear_good_enough_yes_reached() -> None:
    state = current_episode()
    if state is not None:
        state.set_good_enough(False)
