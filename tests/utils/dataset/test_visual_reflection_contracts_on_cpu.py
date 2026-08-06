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
"""CPU tests for generic visual-reflection contracts and text protocol."""

import json

import pytest

from verl_omni.utils.dataset.visual_reflection import (
    RejectionReason,
    VisualReflectionDataError,
    format_reflection_response,
    parse_reflection_response,
    validate_trajectory,
)
from verl_omni.utils.dataset.visual_reflection.contracts import (
    FINGERPRINT_VERSION,
    SCHEMA_VERSION,
    canonical_json,
    derive_pair_source_dedup_key,
    derive_prompt_source_dedup_key,
    derive_trajectory_id,
    stable_digest,
    validate_image_ref,
    validate_reflection_step,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
EXPECTED_PUBLIC_API = {
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
}


def test_schema_identity_is_model_agnostic():
    assert SCHEMA_VERSION == "visual_reflection_v1"


def test_package_root_exports_only_stable_generic_api():
    import verl_omni.utils.dataset.visual_reflection as visual_reflection

    assert set(visual_reflection.__all__) == EXPECTED_PUBLIC_API


def test_pair_source_dedup_key_uses_source_neutral_digest_domain():
    expected = stable_digest(
        "image_pair_source_dedup_v1",
        {
            "version": FINGERPRINT_VERSION,
            "prompt": "draw one object.",
            "reference_sha256": SHA_A,
        },
    )

    actual = derive_pair_source_dedup_key(
        "  Draw ONE object.  ",
        {"uri": "images/reference.png", "sha256": SHA_A},
    )

    assert actual == expected


def test_prompt_source_dedup_key_uses_source_neutral_digest_domain():
    expected = stable_digest(
        "prompt_only_source_dedup_v1",
        {
            "version": FINGERPRINT_VERSION,
            "prompt": "draw one object.",
        },
    )

    assert derive_prompt_source_dedup_key("  Draw ONE object.  ") == expected


def _trajectory(**overrides):
    value = {
        "trajectory_id": "traj-1",
        "manifest_id": "manifest-1",
        "source_dataset": "dataset",
        "source_record_id": "row-1",
        "pipeline_variant": "direct_parse",
        "prompt": "Draw one object.",
        "images": [{"uri": "images/a.png", "sha256": SHA_A}],
        "steps": [{"reflection": "The image is correct.", "action": "stop", "edit": ""}],
    }
    value.update(overrides)
    if "trajectory_id" not in overrides:
        value["trajectory_id"] = derive_trajectory_id(
            manifest_id=value["manifest_id"],
            source_dataset=value["source_dataset"],
            source_record_id=value["source_record_id"],
            pipeline_variant=value["pipeline_variant"],
            prompt=value["prompt"],
            images=value["images"],
            steps=value["steps"],
        )
    return value


def test_contracts_round_trip_through_canonical_json():
    trajectory = validate_trajectory(_trajectory())

    restored = json.loads(canonical_json(trajectory))

    assert validate_trajectory(restored) == trajectory


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, ""])
def test_image_ref_rejects_noncanonical_digest(digest):
    with pytest.raises(VisualReflectionDataError) as error:
        validate_image_ref({"uri": "image.png", "sha256": digest})

    assert error.value.reason is RejectionReason.INVALID_IMAGE_HASH


@pytest.mark.parametrize(
    ("step", "reason"),
    [
        ({"reflection": "", "action": "stop", "edit": ""}, RejectionReason.EMPTY_REFLECTION),
        (
            {"reflection": "Needs work.", "action": "continue", "edit": "  "},
            RejectionReason.EMPTY_CONTINUE_EDIT,
        ),
        (
            {"reflection": "Done.", "action": "stop", "edit": "Change it."},
            RejectionReason.NONEMPTY_STOP_EDIT,
        ),
    ],
)
def test_reflection_step_fail_closed(step, reason):
    with pytest.raises(VisualReflectionDataError) as error:
        validate_reflection_step(step)

    assert error.value.reason is reason


@pytest.mark.parametrize(
    "trajectory",
    [
        _trajectory(images=[], steps=[]),
        _trajectory(images=[{"uri": "a", "sha256": SHA_A}], steps=[]),
        _trajectory(
            images=[{"uri": "a", "sha256": SHA_A}, {"uri": "b", "sha256": SHA_B}],
            steps=[
                {"reflection": "Stopped too early.", "action": "stop", "edit": ""},
                {"reflection": "Done.", "action": "stop", "edit": ""},
            ],
        ),
        _trajectory(steps=[{"reflection": "Not done.", "action": "continue", "edit": "Fix it."}]),
    ],
)
def test_trajectory_rejects_invalid_state_machine(trajectory):
    with pytest.raises(VisualReflectionDataError):
        validate_trajectory(trajectory)


def test_trajectory_rejects_forged_content_id():
    with pytest.raises(VisualReflectionDataError):
        validate_trajectory(_trajectory(trajectory_id="traj-forged"))


@pytest.mark.parametrize(
    "step",
    [
        {"reflection": "The hand is distorted.", "action": "continue", "edit": "Fix only the hand."},
        {"reflection": "The image now matches the prompt.", "action": "stop", "edit": ""},
        {
            "reflection": "The subject is correct.\nThe background is too dark.",
            "action": "continue",
            "edit": "Brighten the background.\nPreserve the subject.",
        },
    ],
)
def test_protocol_formatter_parser_round_trip(step):
    text = format_reflection_response(step)

    assert parse_reflection_response(text) == step
    assert not text.endswith("<eos>")


def test_protocol_normalizes_crlf_and_terminal_shape():
    response = "Reflection: Looks good.\r\nAction: stop\r\nEdit:\r\n"

    parsed = parse_reflection_response(response)

    assert parsed == {"reflection": "Looks good.", "action": "stop", "edit": ""}
    assert format_reflection_response(parsed) == "Reflection: Looks good.\nAction: stop\nEdit:"


@pytest.mark.parametrize(
    "response",
    [
        "Action: stop\nReflection: Done.\nEdit:",
        "Reflection: Done.\nAction: STOP\nEdit:",
        "Reflection: Done.\nAction: wait\nEdit:",
        "Reflection: Done.\nAction: continue\nEdit:",
        "Reflection: Done.\nAction: stop\nEdit: change it",
        "Reflection: One.\nReflection: Two.\nAction: stop\nEdit:",
        "```\nReflection: Done.\nAction: stop\nEdit:\n```",
    ],
)
def test_protocol_rejects_malformed_responses(response):
    with pytest.raises(VisualReflectionDataError) as error:
        parse_reflection_response(response)

    assert error.value.reason is RejectionReason.PROTOCOL_PARSE_ERROR


def test_canonical_step_rejects_ambiguous_reserved_header_inside_content():
    step = {
        "reflection": "First observation.\nAction: this line is ambiguous.",
        "action": "stop",
        "edit": "",
    }

    with pytest.raises(VisualReflectionDataError) as error:
        validate_reflection_step(step)

    assert error.value.reason is RejectionReason.PROTOCOL_PARSE_ERROR
    with pytest.raises(VisualReflectionDataError):
        format_reflection_response(step)
