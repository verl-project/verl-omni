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

"""The shared worker accepts only explicit, qualified engine capabilities."""

from types import SimpleNamespace

import pytest

from verl_omni.workers.engine.megatron.omni_impl import OmniMegatronEngine
from verl_omni.workers.engine_workers import TrainingWorker


@pytest.mark.parametrize("capability", [None, False, 1, lambda: False, lambda: 1])
def test_shared_worker_rejects_missing_or_unqualified_capability(capability):
    worker = SimpleNamespace(engine=SimpleNamespace(supports_detached_training_output=capability))
    with pytest.raises(ValueError, match="does not support"):
        TrainingWorker._check_training_output_support(worker)


def test_shared_worker_accepts_explicit_capability_without_model_type_dependency():
    worker = SimpleNamespace(engine=SimpleNamespace(supports_detached_training_output=lambda: True))
    TrainingWorker._check_training_output_support(worker)


@pytest.mark.parametrize(
    "forward_only,dynamic_cp,expected", [(False, False, True), (True, False, False), (False, True, False)]
)
def test_omni_megatron_capability_checks_actual_engine_configuration(forward_only, dynamic_cp, expected):
    engine = object.__new__(OmniMegatronEngine)
    engine.engine_config = SimpleNamespace(forward_only=forward_only, dynamic_context_parallel=dynamic_cp)
    assert engine.supports_detached_training_output() is expected


def test_omni_megatron_subclass_must_explicitly_qualify_itself():
    class UnqualifiedEngine(OmniMegatronEngine):
        pass

    engine = object.__new__(UnqualifiedEngine)
    engine.engine_config = SimpleNamespace(forward_only=False, dynamic_context_parallel=False)
    assert engine.supports_detached_training_output() is False
