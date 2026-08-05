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
"""Stable public API for generic multi-turn visual-reflection data."""

from .contracts import (
    AtomicSFTExample,
    DataManifest,
    ImageRef,
    ReflectionStep,
    RejectionReason,
    SourceSplitAssignment,
    SplitProvenance,
    VisualReflectionDataError,
    VisualReflectionTrajectory,
    validate_trajectory,
)
from .images import ImageResolver, LocalImageResolver
from .partition import assign_source_splits, deduplicate_source_records, make_split_provenance
from .planner import plan_atomic_examples, validate_atomic_example
from .protocol import format_reflection_response, parse_reflection_response
from .provenance import RejectionLedger, build_data_manifest, derive_manifest_id

__all__ = [
    "AtomicSFTExample",
    "DataManifest",
    "ImageRef",
    "ImageResolver",
    "LocalImageResolver",
    "ReflectionStep",
    "RejectionLedger",
    "RejectionReason",
    "SourceSplitAssignment",
    "SplitProvenance",
    "VisualReflectionDataError",
    "VisualReflectionTrajectory",
    "assign_source_splits",
    "build_data_manifest",
    "deduplicate_source_records",
    "derive_manifest_id",
    "format_reflection_response",
    "make_split_provenance",
    "parse_reflection_response",
    "plan_atomic_examples",
    "validate_atomic_example",
    "validate_trajectory",
]
